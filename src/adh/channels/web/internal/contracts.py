"""Internal contracts for the Crawl4AI service bridge.

These models define the communication between the synchronous WebChannel
and the asynchronous CrawlService.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Outlink:
    """A link discovered on a crawled page, with its anchor text.

    Anchor text ("Class-based views" vs "Fundraising") is a strong relevance
    signal for the frontier scorer, so we carry it alongside the href.
    """

    url: str
    text: str = ""


@dataclass(frozen=True)
class CrawlRequest:
    """A request for a specific URL to be crawled.

    This is the internal 'job' object the :class:`CrawlRouter` dispatches to the
    :class:`CrawlService`. One seed request becomes a deep-crawl *job*; outlinks
    discovered while crawling re-enter the frontier as derived requests that
    inherit the seed's ``aspect_id``, constraints, and relevance vocabulary.
    """

    url: str
    max_pages: int = 1
    # The aspect this URL serves. Carried end-to-end so the router can flush /
    # cancel an entire aspect's pending frontier when the agent signals
    # saturation (Engineering Review B2 / design-doc §2 saturation loop).
    aspect_id: str | None = None
    # Depth of this URL in the deep-crawl frontier (0 == seed). Used to bound
    # how far outlink expansion travels from the seed.
    depth: int = 0
    # Constraints projected from the JobSpec's DiscoveryConstraints. Enforced at
    # fetch time by the service's guards (defense-in-depth on top of discovery).
    respect_robots_txt: bool = True
    include_domains: tuple[str, ...] | None = None
    exclude_domains: tuple[str, ...] | None = None
    # The dataset's languages (JobSpec.languages). Data for an opt-in locale filter
    # (see policy.make_locale_filter) — the router itself stays language-agnostic.
    allowed_locales: tuple[str, ...] = ()
    # The aspect vocabulary (title / aspect id / snippet from discovery). Data for the
    # relevance strategies — the URL scorer and the content judge's query. Thresholds
    # and which strategies to apply live in the FrontierPolicy, not on the request.
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class CrawlResponse:
    """The raw result from the CrawlService before it is mapped to FetchedPage.

    This carries the raw markdown and metadata needed for provenance. Required
    fields (no defaults) must precede optional ones — ``markdown`` defaults to ""
    so an error response can be built as ``CrawlResponse(url=u, error=...)``.
    """

    url: str
    markdown: str = ""
    final_url: str | None = (
        None  # redirect target — provenance records where content actually came from
    )
    license_hint: str | None = None
    status: int | None = None
    error: str | None = None
    # Outlinks discovered on the page, fed back into the deep-crawl frontier.
    links: tuple[Outlink, ...] = field(default_factory=tuple)
    # The aspect / depth this response belongs to (mirrored from its request so
    # the router can route results and the frontier can track depth).
    aspect_id: str | None = None
    depth: int = 0
    relevance_score: float | None = None
    fetched_at: float | None = None
