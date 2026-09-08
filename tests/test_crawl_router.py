"""Tests for the CrawlRouter (deep-crawl frontier + Queue Flush) and the live
WebChannel wiring. All deterministic and offline: the async core is driven through
a synchronous fake service, and the end-to-end path uses an injected fake crawler
(no crawl4ai / no browser)."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from datetime import datetime, timezone
from types import SimpleNamespace

from adh.channels import FetchContext, WebChannel
from adh.channels.web import Crawl4AIContentFetcher
from adh.channels.web.channel import _response_to_page
from adh.channels.web.internal.contracts import CrawlRequest, CrawlResponse, Outlink
from adh.channels.web.internal.policy import (
    FrontierPolicy,
    collapsed_version_key,
    compose_filters,
    domain_filter,
    make_locale_filter,
)
from adh.channels.web.internal.relevance import Bm25RelevanceJudge, LlmRelevanceJudge
from adh.channels.web.internal.router import CrawlRouter
from adh.channels.web.internal.scoring import KeywordRelevanceScorer
from adh.channels.web.internal.service import CrawlService, ServiceConfig
from adh.contract import JobSpec, SourceChannel
from adh.corpus import LocalCorpusStore
from adh.discovery.models import CandidateSource, RankedSource, SourceEvaluation


def _as_outlink(link) -> Outlink:
    """Accept a bare URL or an (url, anchor) pair in test page maps."""
    if isinstance(link, Outlink):
        return link
    if isinstance(link, tuple):
        return Outlink(url=link[0], text=link[1] if len(link) > 1 else "")
    return Outlink(url=link)


# --------------------------------------------------------------------- fakes
class _SyncFakeService:
    """Implements the slice CrawlRouter needs; runs the coordinator inline.

    ``run_coroutine`` executes the coroutine to completion on a throwaway loop and
    returns an already-resolved Future, so ``router.run_job`` (the public API) is
    exercised with no threads and no flakiness.
    """

    def __init__(self, pages: dict[str, tuple[str, list[str]]]):
        self.pages = pages
        self.crawled: list[str] = []

    async def acrawl(self, request: CrawlRequest) -> CrawlResponse:
        self.crawled.append(request.url)
        entry = self.pages.get(request.url, ("", []))
        md, links = entry[0], entry[1]
        final = entry[2] if len(entry) > 2 else None  # optional redirect target
        if not md:
            return CrawlResponse(url=request.url, error="not found", aspect_id=request.aspect_id)
        return CrawlResponse(
            url=request.url,
            markdown=md,
            final_url=final,
            links=tuple(_as_outlink(link) for link in links),
            aspect_id=request.aspect_id,
            depth=request.depth,
        )

    def run_coroutine(self, coro) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(asyncio.run(coro))
        except Exception as exc:  # pragma: no cover - surfaced to caller
            fut.set_exception(exc)
        return fut


class _FakeMarkdown:
    def __init__(self, md: str):
        self.fit_markdown = md


class _FakeResult:
    def __init__(self, url: str, md: str, links: list[str], *, final: str | None = None):
        self.url = final or url
        self.markdown = _FakeMarkdown(md)
        self.status_code = 200
        self.links = {"internal": [{"href": h} for h in links]}


class LinkedFakeCrawler:
    """AsyncWebCrawler double returning content + outlinks per URL."""

    def __init__(self, pages: dict[str, tuple[str, list[str]]]):
        self.pages = pages
        self.calls: list[str] = []

    async def start(self):
        return None

    async def close(self):
        return None

    async def arun(self, url: str, config=None):
        self.calls.append(url)
        md, links = self.pages.get(url, (f"# {url}", []))
        return _FakeResult(url, md, links)


def _seed(url="https://example.com/root", **kw) -> CrawlRequest:
    return CrawlRequest(url=url, max_pages=kw.pop("max_pages", 10), aspect_id="a0", **kw)


def _ranked(locator: str, *, aspect="a0", license_=None) -> RankedSource:
    cand = CandidateSource(
        source_id=f"{aspect}:web:{locator}",
        channel=SourceChannel.WEB,
        locator=locator,
        title=locator,
        aspect_id=aspect,
        license=license_,
        discovered_via="q",
    )
    ev = SourceEvaluation(
        source_id=cand.source_id,
        shape_match=0.8,
        quality=0.8,
        license_ok=bool(license_),
        freshness_ok=True,
        estimated_yield=10,
        score=0.8,
    )
    return RankedSource(candidate=cand, evaluation=ev)


def _ctx(minimal_spec_dict, tmp_path, *, governor=None, max_items=10) -> FetchContext:
    kw = {} if governor is None else {"governor": governor}
    return FetchContext(
        job_id="j1",
        store=LocalCorpusStore(tmp_path),
        spec=JobSpec.model_validate(minimal_spec_dict),
        max_items_per_source=max_items,
        clock=lambda: datetime.now(timezone.utc),
        **kw,
    )


# --------------------------------------------------------------------- router: frontier
def test_single_seed_no_links():
    svc = _SyncFakeService({"https://example.com/root": ("root body", [])})
    router = CrawlRouter(svc, workers=4)
    pages = router.run_job(_seed(), max_pages=10)
    assert [p.url for p in pages] == ["https://example.com/root"]
    assert svc.crawled == ["https://example.com/root"]


def test_deep_crawl_expands_frontier():
    svc = _SyncFakeService(
        {
            "https://example.com/root": (
                "root",
                ["https://example.com/a", "https://example.com/b"],
            ),
            "https://example.com/a": ("a body", []),
            "https://example.com/b": ("b body", []),
        }
    )
    router = CrawlRouter(svc, workers=4, max_depth=2)
    pages = router.run_job(_seed(), max_pages=10)
    assert {p.url for p in pages} == {
        "https://example.com/root",
        "https://example.com/a",
        "https://example.com/b",
    }


def test_max_pages_caps_frontier():
    links = [f"https://example.com/{i}" for i in range(5)]
    pages_map = {"https://example.com/root": ("root", links)}
    pages_map.update({u: (f"body {u}", []) for u in links})
    svc = _SyncFakeService(pages_map)
    router = CrawlRouter(svc, workers=4)
    pages = router.run_job(_seed(), max_pages=2)
    assert len(pages) == 2  # exact cap, no overshoot
    assert "https://example.com/root" in {p.url for p in pages}


def test_dedup_does_not_recrawl():
    svc = _SyncFakeService(
        {
            "https://example.com/root": (
                "root",
                ["https://example.com/a", "https://example.com/root", "https://example.com/a"],
            ),
            "https://example.com/a": ("a body", ["https://example.com/root"]),
        }
    )
    router = CrawlRouter(svc, workers=4)
    pages = router.run_job(_seed(), max_pages=10)
    assert sorted(svc.crawled) == ["https://example.com/a", "https://example.com/root"]
    assert len(pages) == 2


def test_domain_filter_excludes_offsite_links():
    svc = _SyncFakeService(
        {
            "https://example.com/root": (
                "root",
                ["https://example.com/a", "https://evil.com/x"],
            ),
            "https://example.com/a": ("a body", []),
            "https://evil.com/x": ("evil body", []),
        }
    )
    router = CrawlRouter(svc, workers=4)
    pages = router.run_job(_seed(include_domains=("example.com",)), max_pages=10)
    assert "https://evil.com/x" not in svc.crawled
    assert {p.url for p in pages} == {"https://example.com/root", "https://example.com/a"}


def test_max_depth_bounds_expansion():
    svc = _SyncFakeService(
        {
            "https://example.com/root": ("root", ["https://example.com/a"]),
            "https://example.com/a": ("a body", ["https://example.com/b"]),
            "https://example.com/b": ("b body", []),
        }
    )
    router = CrawlRouter(svc, workers=4, max_depth=1)
    pages = router.run_job(_seed(), max_pages=10)
    assert "https://example.com/b" not in svc.crawled  # depth 2 not reached
    assert {p.url for p in pages} == {"https://example.com/root", "https://example.com/a"}


# --------------------------------------------------------------------- router: flush / cancel
def test_on_result_flush_stops_crawl():
    svc = _SyncFakeService(
        {
            "https://example.com/root": (
                "root",
                ["https://example.com/a", "https://example.com/b"],
            ),
            "https://example.com/a": ("a body", []),
            "https://example.com/b": ("b body", []),
        }
    )
    router = CrawlRouter(svc, workers=4)
    # Stop immediately after the seed: its outlinks must never be crawled.
    pages = router.run_job(_seed(), max_pages=10, on_result=lambda resp: False)
    assert svc.crawled == ["https://example.com/root"]
    assert len(pages) == 1


def test_raising_on_result_does_not_wedge_job():
    """A throwing hook must not hang frontier.join() (worker stays alive)."""
    svc = _SyncFakeService(
        {
            "https://example.com/root": ("root", ["https://example.com/a"]),
            "https://example.com/a": ("a body", []),
        }
    )
    router = CrawlRouter(svc, workers=2)

    def boom(_resp):
        raise RuntimeError("hook blew up")

    pages = router.run_job(_seed(), max_pages=10, on_result=boom)
    # The seed was crawled and recorded before the hook ran; the job still returns.
    assert [p.url for p in pages] == ["https://example.com/root"]


def test_cancel_unknown_job_returns_false():
    svc = _SyncFakeService({})
    router = CrawlRouter(svc, workers=2)
    assert router.cancel("job-does-not-exist") is False
    assert router.flush_aspect("nope") == 0
    assert router.flush() == 0


# --------------------------------------------------------------------- service bridge integration
def test_router_drives_real_service_bridge():
    """End-to-end through the real background-loop service + a fake crawler."""
    crawler = LinkedFakeCrawler(
        {
            "https://example.com/root": ("root", ["https://example.com/a"]),
            "https://example.com/a": ("a body", []),
        }
    )
    svc = CrawlService(
        ServiceConfig(),
        crawler_factory=lambda: crawler,
        run_config_factory=lambda req: None,
    )
    router = CrawlRouter(svc, workers=2, max_depth=2)
    try:
        pages = router.run_job(_seed(), max_pages=10)
        assert {p.url for p in pages} == {"https://example.com/root", "https://example.com/a"}
        assert crawler.calls.count("https://example.com/root") == 1  # singleton reused
    finally:
        svc.close()


# --------------------------------------------------------------------- relevance scoring
def test_keyword_scorer_ranks_on_topic_above_off_topic_and_kills_off_host():
    seed = "https://docs.djangoproject.com/en/6.0/topics/http/views/"
    scorer = KeywordRelevanceScorer.from_seed(seed, ("views view class-based generic mixin",))

    on_topic = scorer.score("https://docs.djangoproject.com/en/6.0/ref/views/")
    sibling = scorer.score(
        "https://docs.djangoproject.com/en/6.0/topics/http/shortcuts/", anchor="render"
    )
    off_topic = scorer.score("https://docs.djangoproject.com/en/6.0/topics/forms/", anchor="Forms")
    off_host = scorer.score("https://www.djangoproject.com/fundraising/", anchor="Fundraising")

    assert off_host == 0.0  # different host -> never crawled (the reported drift)
    assert on_topic > off_topic  # 'views' keyword beats an unrelated section
    assert sibling > off_topic  # path affinity boosts a page under the seed's dir
    assert off_topic < 0.3  # below a sane prune threshold
    assert on_topic >= 0.3  # survives it


def test_relevance_threshold_prunes_off_topic_and_off_host_frontier():
    svc = _SyncFakeService(
        {
            "https://docs.dj.com/topics/views/": (
                "views overview",
                [
                    ("https://docs.dj.com/topics/views/detail/", "Detail views"),  # on-topic
                    ("https://www.dj.com/fundraising/", "Fundraising"),  # off-host drift
                    ("https://docs.dj.com/topics/forms/", "Forms"),  # off-topic
                ],
            ),
            "https://docs.dj.com/topics/views/detail/": ("detail body", []),
            "https://docs.dj.com/topics/forms/": ("forms body", []),
            "https://www.dj.com/fundraising/": ("donate", []),
        }
    )
    router = CrawlRouter(svc, workers=4, max_depth=2)
    seed = CrawlRequest(
        url="https://docs.dj.com/topics/views/",
        aspect_id="views",
        keywords=("views view detail",),
    )
    scorer = KeywordRelevanceScorer.from_seed(seed.url, seed.keywords)
    policy = FrontierPolicy(url_scorer=scorer, url_threshold=0.3)
    pages = router.run_job(seed, max_pages=10, policy=policy)
    urls = {p.url for p in pages}

    assert "https://docs.dj.com/topics/views/" in urls
    assert "https://docs.dj.com/topics/views/detail/" in urls  # relevant -> kept
    assert "https://www.dj.com/fundraising/" not in urls  # off-host -> pruned
    assert "https://docs.dj.com/topics/forms/" not in urls  # off-topic -> pruned
    assert "https://www.dj.com/fundraising/" not in svc.crawled  # never even fetched


def _versioned_service() -> _SyncFakeService:
    return _SyncFakeService(
        {
            "https://docs.dj.com/en/6.0/topics/views/": (
                "views",
                [
                    "https://docs.dj.com/en/5.1/topics/views/",  # same page, older version
                    "https://docs.dj.com/en/dev/topics/views/",  # same page, dev version
                    "https://docs.dj.com/en/6.0/ref/views/",  # genuinely different page
                ],
            ),
            "https://docs.dj.com/en/5.1/topics/views/": ("views (5.1)", []),
            "https://docs.dj.com/en/dev/topics/views/": ("views (dev)", []),
            "https://docs.dj.com/en/6.0/ref/views/": ("ref body", []),
        }
    )


def test_default_policy_does_not_collapse_versions():
    """Version-collapse is NOT hardcoded: the generic default treats twins as distinct."""
    svc = _versioned_service()
    router = CrawlRouter(svc, workers=4, max_depth=2)
    router.run_job(CrawlRequest(url="https://docs.dj.com/en/6.0/topics/views/"), max_pages=10)
    assert len(svc.crawled) == 4  # seed + 2 version twins + ref page (no collapsing)


def test_version_collapse_is_opt_in_policy():
    """With the opt-in collapsed_version_key, version twins dedupe to one crawl."""
    svc = _versioned_service()
    router = CrawlRouter(svc, workers=4, max_depth=2)
    seed = CrawlRequest(url="https://docs.dj.com/en/6.0/topics/views/")
    router.run_job(seed, max_pages=10, policy=FrontierPolicy(dedup_key=collapsed_version_key))
    assert sorted(svc.crawled) == [
        "https://docs.dj.com/en/6.0/ref/views/",
        "https://docs.dj.com/en/6.0/topics/views/",
    ]


def test_locale_filter_is_opt_in_policy():
    """With the opt-in locale filter, other-language twins are pruned for an en dataset."""
    svc = _SyncFakeService(
        {
            "https://docs.dj.com/en/topics/views/": (
                "views",
                [
                    "https://docs.dj.com/fr/topics/views/",  # French twin
                    "https://docs.dj.com/zh-hans/topics/views/",  # Chinese twin
                    "https://docs.dj.com/en/ref/views/",  # English, different page
                ],
            ),
            "https://docs.dj.com/en/ref/views/": ("ref", []),
        }
    )
    router = CrawlRouter(svc, workers=4, max_depth=2)
    seed = CrawlRequest(url="https://docs.dj.com/en/topics/views/", allowed_locales=("en",))
    policy = FrontierPolicy(link_filter=compose_filters(domain_filter, make_locale_filter()))
    router.run_job(seed, max_pages=10, policy=policy)
    assert sorted(svc.crawled) == [
        "https://docs.dj.com/en/ref/views/",
        "https://docs.dj.com/en/topics/views/",
    ]


def test_best_first_crawls_most_relevant_within_budget():
    svc = _SyncFakeService(
        {
            "https://docs.dj.com/views/": (
                "seed",
                [
                    ("https://docs.dj.com/misc/", "Misc"),  # low score
                    ("https://docs.dj.com/views/detail/", "Detail views"),  # high score
                ],
            ),
            "https://docs.dj.com/views/detail/": ("detail", []),
            "https://docs.dj.com/misc/": ("misc", []),
        }
    )
    router = CrawlRouter(svc, workers=1, max_depth=2)  # 1 worker -> strict best-first
    seed = CrawlRequest(url="https://docs.dj.com/views/", keywords=("views view detail",))
    scorer = KeywordRelevanceScorer.from_seed(seed.url, seed.keywords)
    # Budget of 2: seed + the single most relevant link, not the misc one.
    pages = router.run_job(seed, max_pages=2, policy=FrontierPolicy(url_scorer=scorer))
    urls = [p.url for p in pages]
    assert urls == ["https://docs.dj.com/views/", "https://docs.dj.com/views/detail/"]


# --------------------------------------------------------------------- content relevance (BM25 / LLM)
_VIEWS_QUERY = "django class-based views dispatch request response generic mixin handler"
_ON_TOPIC = (
    "A view is a Python function that takes a request and returns a response.\n"
    "Class-based views let you structure views and reuse code via mixins and inheritance.\n"
    "The view dispatches the request to a handler method based on the HTTP method.\n"
    "Generic editing views handle forms while generic display views render objects.\n"
    "Each handler returns an HttpResponse or a subclass like JsonResponse."
)
_OFF_TOPIC = (
    "The Django Software Foundation is an independent non-profit organization.\n"
    "Your donation directly funds development sprints and community outreach.\n"
    "Become an individual or corporate member to support the project.\n"
    "Fundraising campaigns run throughout the year and sponsors get recognition.\n"
    "Read the latest weblog posts about upcoming conferences worldwide."
)


def test_bm25_judge_scores_on_topic_above_off_topic():
    judge = Bm25RelevanceJudge()
    assert judge.score(_VIEWS_QUERY, _ON_TOPIC) > judge.score(_VIEWS_QUERY, _OFF_TOPIC)
    assert judge.score(_VIEWS_QUERY, _ON_TOPIC) >= 0.4
    assert judge.score(_VIEWS_QUERY, _OFF_TOPIC) < 0.4
    assert judge.score("", _OFF_TOPIC) == 1.0  # no query -> no gating
    assert judge.score(_VIEWS_QUERY, "") == 0.0  # no content -> not relevant


class _FakeChatModel:
    """Minimal LangChain-style chat model: invoke(prompt) -> message with .content.

    Carries ``usage_metadata`` like a real AIMessage so token counting is exercised.
    """

    def __init__(self, reply: str, *, in_tokens: int = 0, out_tokens: int = 0):
        self.reply = reply
        self.prompts: list[str] = []
        self._usage = {"input_tokens": in_tokens, "output_tokens": out_tokens}

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply, usage_metadata=self._usage)


def test_llm_judge_parses_clamps_and_truncates():
    model = _FakeChatModel("Relevance: 1.7")  # out of range, with prose
    judge = LlmRelevanceJudge(model=model, max_chars=5)
    assert judge.score("views", "abcdefghij") == 1.0  # parsed 1.7 -> clamped to 1.0
    assert "abcde" in model.prompts[0] and "abcdef" not in model.prompts[0]  # truncated
    assert judge.score("", "anything") == 1.0  # empty query -> no model call
    assert len(model.prompts) == 1


def test_llm_judge_counts_tokens():
    model = _FakeChatModel("0.9", in_tokens=120, out_tokens=4)
    judge = LlmRelevanceJudge(model=model)
    judge.score("views", "on-topic content one")
    judge.score("views", "on-topic content two")
    assert judge.usage.calls == 2
    assert judge.usage.total.input_tokens == 240
    assert judge.usage.total.output_tokens == 8
    assert judge.usage.total.total_tokens == 248
    judge.score("", "skipped")  # empty query -> no model call -> not counted
    assert judge.usage.calls == 2


def test_llm_judge_shared_tracker_aggregates_across_judges():
    from adh.llm import UsageTracker

    tracker = UsageTracker()
    j1 = LlmRelevanceJudge(model=_FakeChatModel("0.8", in_tokens=10, out_tokens=2), usage=tracker)
    j2 = LlmRelevanceJudge(model=_FakeChatModel("0.7", in_tokens=30, out_tokens=5), usage=tracker)
    j1.score("q", "a")
    j2.score("q", "b")
    assert tracker.calls == 2
    assert tracker.total.total_tokens == 47  # 10+2 + 30+5


def test_llm_judge_from_model_id_delegates_to_shared_factory(monkeypatch):
    import adh.llm

    captured = {}

    def fake_build(model_id, **kw):
        captured["model_id"], captured["kw"] = model_id, kw
        return _FakeChatModel("0.7")

    monkeypatch.setattr(adh.llm, "build_chat_model", fake_build)
    judge = LlmRelevanceJudge.from_model_id(
        "anthropic:claude-haiku-4-5", model_kwargs={"temperature": 0}, max_chars=12
    )
    assert captured == {"model_id": "anthropic:claude-haiku-4-5", "kw": {"temperature": 0}}
    assert judge.max_chars == 12  # judge kwargs routed to the judge
    assert judge.score("views", "some on-topic views content") == 0.7  # uses the built model


def test_content_judge_drops_off_topic_pages_and_their_links():
    cbv = (
        "Class-based views let you structure views and reuse code through inheritance.\n"
        "A class-based view dispatches the request to a handler method based on the HTTP verb.\n"
        "Generic display views render objects and generic editing views process forms.\n"
        "Mixins compose reusable behaviour across multiple views.\n"
        "Each handler returns an HttpResponse to the caller."
    )
    mixins = (
        "Mixins provide reusable methods for class-based views.\n"
        "A mixin can override dispatch to customise how a view handles each request.\n"
        "ContextMixin supplies get_context_data for views that render templates.\n"
        "SingleObjectMixin and MultipleObjectMixin back the generic display views.\n"
        "Combining mixins lets you build custom views with shared handler logic."
    )
    donate = _OFF_TOPIC  # clearly off-topic content
    svc = _SyncFakeService(
        {
            "https://site.com/views/": (
                _ON_TOPIC,
                [("https://site.com/cbv/", "More views"), ("https://site.com/donate/", "Donate")],
            ),
            "https://site.com/cbv/": (cbv, ["https://site.com/cbv/mixins/"]),
            "https://site.com/cbv/mixins/": (mixins, []),
            "https://site.com/donate/": (donate, ["https://site.com/donate/thanks/"]),
            "https://site.com/donate/thanks/": ("Thank you for your generous donation today.", []),
        }
    )
    router = CrawlRouter(svc, workers=1, max_depth=3)
    seed = CrawlRequest(url="https://site.com/views/", keywords=(_VIEWS_QUERY,))
    policy = FrontierPolicy(judge=Bm25RelevanceJudge(), content_threshold=0.3, query=_VIEWS_QUERY)
    pages = router.run_job(seed, max_pages=10, policy=policy)
    urls = {p.url for p in pages}
    assert "https://site.com/cbv/" in urls  # on-topic kept
    assert "https://site.com/cbv/mixins/" in urls  # expanded from a kept page
    assert "https://site.com/donate/" not in urls  # off-topic dropped
    assert "https://site.com/donate/thanks/" not in svc.crawled  # its links not followed


def test_redirect_duplicates_are_dropped_post_fetch():
    """Two requested URLs resolving to the same final page yield ONE result."""
    svc = _SyncFakeService(
        {
            "https://d.com/root/": (
                "root body",
                ["https://d.com/stable/x/", "https://d.com/latest/x/"],
            ),
            # both redirect to the same resolved page
            "https://d.com/stable/x/": ("x body", [], "https://d.com/6.0/x/"),
            "https://d.com/latest/x/": ("x body", [], "https://d.com/6.0/x/"),
        }
    )
    router = CrawlRouter(svc, workers=1, max_depth=2)
    pages = router.run_job(CrawlRequest(url="https://d.com/root/"), max_pages=10)
    finals = [p.final_url or p.url for p in pages]
    assert finals.count("https://d.com/6.0/x/") == 1  # duplicate dropped
    assert len(pages) == 2  # root + one copy of x


def test_max_fetches_bounds_a_high_drop_judge_crawl():
    """A judge that drops everything must not keep fetching unboundedly."""
    # A long chain: every page links to the next; the judge rejects all of them.
    chain = {
        f"https://site.com/p{i}/": (f"irrelevant filler {i}", [f"https://site.com/p{i + 1}/"])
        for i in range(50)
    }
    svc = _SyncFakeService(chain)

    class _RejectAll:
        def score(self, query, content):
            return 0.0

    router = CrawlRouter(svc, workers=1, max_depth=100)
    seed = CrawlRequest(url="https://site.com/p0/", max_pages=5)
    policy = FrontierPolicy(judge=_RejectAll(), content_threshold=0.5, query="anything")
    pages = router.run_job(seed, max_pages=5, max_fetches=8, policy=policy)
    assert len(pages) <= 1  # only the ungated seed can be kept
    assert len(svc.crawled) <= 8  # the fetch budget held


# --------------------------------------------------------------------- provenance mapping
def test_response_to_page_prefers_final_url():
    redirected = CrawlResponse(url="https://x/req", final_url="https://x/final", markdown="body")
    assert _response_to_page(redirected).url == "https://x/final"
    plain = CrawlResponse(url="https://x/req", markdown="body")
    assert _response_to_page(plain).url == "https://x/req"  # falls back to requested


# --------------------------------------------------------------------- WebChannel live wiring
def test_web_channel_live_path_deep_crawls(minimal_spec_dict, tmp_path):
    """Proves the channel->fetcher->router->service path works (was a hard crash)."""
    fetcher = Crawl4AIContentFetcher(
        crawler_factory=lambda: LinkedFakeCrawler(
            {
                "https://docs.rs/root": ("# Async macros big content", ["https://docs.rs/a"]),
                "https://docs.rs/a": ("# more async content here", []),
            }
        ),
        run_config_factory=lambda req: None,
    )
    channel = WebChannel(fetcher=fetcher)
    try:
        chunks = channel.fetch(_ranked("https://docs.rs/root"), _ctx(minimal_spec_dict, tmp_path))
        assert {c.source_url for c in chunks} == {"https://docs.rs/root", "https://docs.rs/a"}
        assert all(c.source_channel == SourceChannel.WEB for c in chunks)
    finally:
        channel.close()


def test_web_channel_live_path_flushes_on_saturation(minimal_spec_dict, tmp_path):
    """The governor's stop signal flushes the live frontier mid-crawl."""

    class _StopAfterFirst:
        def __init__(self):
            self.observed: list[str] = []

        def observe(self, aspect_id, text):
            self.observed.append(text)

        def should_stop(self, aspect_id):
            return len(self.observed) >= 1  # saturated after the very first page

    fetcher = Crawl4AIContentFetcher(
        crawler_factory=lambda: LinkedFakeCrawler(
            {
                "https://docs.rs/root": (
                    "root content",
                    ["https://docs.rs/a", "https://docs.rs/b"],
                ),
                "https://docs.rs/a": ("a content", []),
                "https://docs.rs/b": ("b content", []),
            }
        ),
        run_config_factory=lambda req: None,
    )
    governor = _StopAfterFirst()
    channel = WebChannel(fetcher=fetcher)
    try:
        chunks = channel.fetch(
            _ranked("https://docs.rs/root"),
            _ctx(minimal_spec_dict, tmp_path, governor=governor),
        )
        # Saturated after the seed -> outlinks were flushed, never fetched.
        assert {c.source_url for c in chunks} == {"https://docs.rs/root"}
        assert governor.observed == ["root content"]
    finally:
        channel.close()
