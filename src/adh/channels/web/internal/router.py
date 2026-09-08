"""CrawlRouter — the internal dispatcher (design-doc §2.3, component map).

The doc separates *Browser Mgmt* (``CrawlService``) from *Dispatching*
(``CrawlRouter``: "Queue management, async result routing"). The router turns one seed
:class:`CrawlRequest` into a bounded deep-crawl *job*: a **best-first** frontier drained
by a worker pool, until the page budget is hit.

The router is a pure mechanism. *Every* judgement — what is a duplicate, which links are
eligible, how to prioritise them, whether a fetched page is on-topic — is delegated to a
:class:`FrontierPolicy` (see :mod:`policy`). Nothing site-specific (versioning, locales,
keyword lists) is baked in here; the default policy is site-agnostic and callers compose
stronger strategies.

Control surface (doc §2.3):

* **Queue Flush (primary):** the per-page ``on_result`` hook is the saturation stop —
  when the agent's coverage map says "saturated", the hook returns ``False`` and the
  router drains the remaining frontier *without* crawling it.
* **Per-job cancellation (optional):** :meth:`cancel` / :meth:`flush_aspect` set a job's
  cancel flag from any thread, killing its pending frontier by id / aspect.

It runs its coordinator coroutine on the *service's* event loop (one loop, one browser,
one semaphore) via ``service.run_coroutine`` — never its own.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from .contracts import CrawlRequest, CrawlResponse
from .policy import FrontierPolicy

logger = logging.getLogger(__name__)

# Returns True to keep crawling, False to request a Queue Flush for this job.
OnResult = Callable[[CrawlResponse], bool]

# Priority given to the seed so it is always crawled before any scored outlink.
_SEED_PRIORITY = -2.0


class _ServiceLike(Protocol):
    """The slice of CrawlService the router depends on (keeps it test-fakeable)."""

    async def acrawl(self, request: CrawlRequest) -> CrawlResponse: ...
    def run_coroutine(self, coro): ...


@dataclass
class _JobHandle:
    """Live state for one in-flight deep-crawl job."""

    job_id: str
    aspect_id: str | None
    cancelled: bool = field(default=False)


@dataclass
class _Job:
    """Per-job working state, threaded through the worker coroutines."""

    seed: CrawlRequest
    max_pages: int
    on_result: OnResult | None
    policy: FrontierPolicy
    handle: _JobHandle
    frontier: "asyncio.PriorityQueue[tuple[float, int, CrawlRequest]]"
    seen: set[str]
    # Post-fetch dedup on the *resolved* URL: two requested URLs can redirect to the
    # same page, which pre-fetch dedup (requested URL) cannot see.
    seen_final: set[str] = field(default_factory=set)
    # Fetch budget: max_pages caps *kept* results, but a content judge that drops
    # many pages would otherwise keep fetching unboundedly. None == no budget.
    max_fetches: int | None = None
    # Per-job depth bound (falls back to the router's default when not overridden).
    max_depth: int = 2
    fetches: int = 0
    results: list[CrawlResponse] = field(default_factory=list)
    _counter: int = 0
    on_event: Callable[[str, str, dict], None] | None = None

    def next_seq(self) -> int:
        """Monotonic tiebreaker so equal-priority items keep FIFO order."""
        self._counter += 1
        return self._counter


class CrawlRouter:
    """Dispatches deep-crawl jobs best-first over a worker pool, with flush/cancel."""

    def __init__(self, service: _ServiceLike, *, workers: int = 4, max_depth: int = 2) -> None:
        self._service = service
        self._workers = max(1, workers)
        self._max_depth = max(0, max_depth)
        self._jobs: dict[str, _JobHandle] = {}
        self._jobs_lock = threading.Lock()  # cancel() may be called cross-thread
        self._counter = 0

    # ------------------------------------------------------------------ sync API
    def run_job(
        self,
        seed: CrawlRequest,
        *,
        max_pages: int = 1,
        max_fetches: int | None = None,
        max_depth: int | None = None,
        on_result: OnResult | None = None,
        policy: FrontierPolicy | None = None,
        timeout: float | None = None,
        on_event: Callable[[str, str, dict], None] | None = None,
    ) -> list[CrawlResponse]:
        """Run one deep-crawl job to completion (or flush) and return its pages.

        Synchronous: bridges onto the service loop and blocks until the frontier is
        drained. ``policy`` supplies dedup/filter/scoring/relevance (default: a fully
        generic, no-pruning policy); ``on_result`` is the saturation hook.
        ``max_fetches`` bounds pages *fetched* (kept or judged-off-topic); ``None``
        leaves only the ``max_pages`` kept-results cap. ``max_depth`` overrides the
        router's default depth bound for this job.
        """
        future = self._service.run_coroutine(
            self._run_job(
                seed,
                max_pages=max_pages,
                max_fetches=max_fetches,
                max_depth=self._max_depth if max_depth is None else max(0, max_depth),
                on_result=on_result,
                policy=policy or FrontierPolicy(),
                on_event=on_event,
            )
        )
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------ control API
    def cancel(self, job_id: str) -> bool:
        """Cancel a job's pending frontier by id. Returns False if unknown/done."""
        with self._jobs_lock:
            handle = self._jobs.get(job_id)
            if handle is None:
                return False
            handle.cancelled = True
            return True

    def flush_aspect(self, aspect_id: str) -> int:
        """Cancel every in-flight job serving ``aspect_id``. Returns how many."""
        with self._jobs_lock:
            matched = [h for h in self._jobs.values() if h.aspect_id == aspect_id]
            for h in matched:
                h.cancelled = True
        return len(matched)

    def flush(self) -> int:
        """Cancel all in-flight jobs (hard stop). Returns how many."""
        with self._jobs_lock:
            handles = list(self._jobs.values())
            for h in handles:
                h.cancelled = True
        return len(handles)

    # ------------------------------------------------------------------ async core
    async def _run_job(
        self,
        seed: CrawlRequest,
        *,
        max_pages: int,
        max_fetches: int | None,
        max_depth: int,
        on_result: OnResult | None,
        policy: FrontierPolicy,
        on_event: Callable[[str, str, dict], None] | None = None,
    ) -> list[CrawlResponse]:
        job = _Job(
            seed=seed,
            max_pages=max_pages,
            max_fetches=max_fetches,
            max_depth=max_depth,
            on_result=on_result,
            policy=policy,
            handle=self._register(seed),
            frontier=asyncio.PriorityQueue(),
            seen={policy.dedup_key(seed.url)},
            on_event=on_event,
        )
        job.frontier.put_nowait((_SEED_PRIORITY, job.next_seq(), seed))

        async def worker() -> None:
            while True:
                _, _, req = await job.frontier.get()
                try:
                    await self._process(job, req)
                except Exception as exc:  # noqa: BLE001 - one bad page must not wedge join()
                    logger.warning("router worker error on %s: %s", req.url, exc)
                finally:
                    job.frontier.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(self._workers)]
        try:
            await job.frontier.join()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._unregister(job.handle.job_id)
        return job.results[: job.max_pages]

    async def _process(self, job: _Job, req: CrawlRequest) -> None:
        """Crawl one frontier item, gate it on relevance, and expand its outlinks."""
        # Queue Flush: once cancelled or a budget is met, drain without crawling.
        if job.handle.cancelled or len(job.results) >= job.max_pages:
            return
        if job.max_fetches is not None and job.fetches >= job.max_fetches:
            logger.debug("fetch budget reached (%d); draining frontier", job.max_fetches)
            return
        job.fetches += 1
        if job.on_event:
            job.on_event("running", req.url, {"depth": req.depth})
        import time

        fetched_at = time.time()
        resp = await self._service.acrawl(req)
        resp = replace(resp, depth=req.depth, fetched_at=fetched_at)
        if resp.error is not None:
            logger.debug("crawl error for %s: %s", req.url, resp.error)
            if job.on_event:
                job.on_event("failed", req.url, {"error": resp.error, "depth": req.depth})
            return
        if not resp.markdown.strip():
            if job.on_event:
                job.on_event("pruned", req.url, {"reason": "empty content", "depth": req.depth})
            return
        # Redirect dedup: drop a page whose resolved URL we already fetched (and stop
        # links to it from re-entering the frontier). Runs before the judge so
        # duplicates never cost a judge call.
        final_key = job.policy.dedup_key(resp.final_url or resp.url)
        if final_key in job.seen_final:
            logger.debug("dropped redirect-duplicate of %s: %s", resp.final_url, req.url)
            if job.on_event:
                job.on_event(
                    "pruned",
                    req.url,
                    {
                        "reason": "redirect duplicate",
                        "final_url": resp.final_url,
                        "depth": req.depth,
                    },
                )
            return
        job.seen_final.add(final_key)
        job.seen.add(final_key)
        relevance = 1.0
        # Content relevance gate (the seed is always kept; it was chosen deliberately).
        # Judging may block (a network LLM call), so run it off the event loop to avoid
        # stalling the shared browser; the judge's tracker is thread-safe.
        if req.depth > 0 and job.policy.content_threshold > 0.0:
            relevance = await asyncio.get_running_loop().run_in_executor(
                None, job.policy.judge.score, job.policy.query, resp.markdown
            )
            if relevance < job.policy.content_threshold:
                logger.debug(
                    "dropped off-topic page (%.2f<%.2f): %s",
                    relevance,
                    job.policy.content_threshold,
                    req.url,
                )
                if job.on_event:
                    judge_name = (
                        "bm25"
                        if "Bm25" in job.policy.judge.__class__.__name__
                        else "llm"
                        if "Llm" in job.policy.judge.__class__.__name__
                        else "keep_all"
                    )
                    job.on_event(
                        "dropped",
                        req.url,
                        {
                            "judge": judge_name,
                            "score": relevance,
                            "threshold": job.policy.content_threshold,
                            "depth": req.depth,
                        },
                    )
                return  # drop the page and do not expand its links
        resp = replace(resp, relevance_score=relevance)
        if len(job.results) >= job.max_pages:  # a peer worker may have filled it
            return
        job.results.append(resp)
        if job.on_event:
            job.on_event(
                "kept",
                req.url,
                {"score": relevance, "depth": req.depth, "license": resp.license_hint},
            )
        keep = job.on_result(resp) if job.on_result is not None else True
        if not keep:
            job.handle.cancelled = True  # saturation -> flush this job's frontier
            if job.on_event:
                job.on_event(
                    "saturated",
                    resp.final_url or resp.url,
                    {"reason": "coverage saturated", "depth": req.depth},
                )
            return
        if req.depth < job.max_depth and len(job.results) < job.max_pages:
            self._enqueue_outlinks(job, req, resp)

    # ------------------------------------------------------------------ frontier
    def _enqueue_outlinks(self, job: _Job, parent: CrawlRequest, resp: CrawlResponse) -> None:
        """Score each outlink; enqueue the eligible ones, best (highest score) first."""
        policy = job.policy
        for link in resp.links:
            key = policy.dedup_key(link.url)
            if key in job.seen:
                continue
            if not policy.link_filter(link.url, job.seed):
                if job.on_event:
                    reason = "domain filter"
                    if job.seed.allowed_locales:
                        import re
                        from urllib.parse import urlparse

                        segments = [s for s in urlparse(link.url).path.split("/") if s]
                        if segments:
                            lead = segments[0].lower()
                            if re.match(r"^[a-z]{2}(-[a-z]{2,4})?$", lead) and lead not in [
                                loc.lower() for loc in job.seed.allowed_locales
                            ]:
                                reason = "locale filter"
                    job.on_event("pruned", link.url, {"reason": reason, "depth": parent.depth + 1})
                continue
            score = policy.url_scorer.score(link.url, anchor=link.text)
            if score < policy.url_threshold:
                if job.on_event:
                    job.on_event(
                        "pruned",
                        link.url,
                        {
                            "reason": "url scorer threshold",
                            "score": score,
                            "threshold": policy.url_threshold,
                            "depth": parent.depth + 1,
                        },
                    )
                continue
            job.seen.add(key)
            job.frontier.put_nowait(
                (-score, job.next_seq(), replace(job.seed, url=link.url, depth=parent.depth + 1))
            )

    # ------------------------------------------------------------------ registry
    def _register(self, seed: CrawlRequest) -> _JobHandle:
        with self._jobs_lock:
            self._counter += 1
            job_id = f"job-{self._counter}"
            handle = _JobHandle(job_id=job_id, aspect_id=seed.aspect_id)
            self._jobs[job_id] = handle
            return handle

    def _unregister(self, job_id: str) -> None:
        with self._jobs_lock:
            self._jobs.pop(job_id, None)
