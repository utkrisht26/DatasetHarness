"""Frontier policy — the pluggable strategies the generic router runs.

The :class:`CrawlRouter` is a mechanism (frontier + workers + flush). *What* counts as
a duplicate, which links are eligible, how links are prioritised, and whether a fetched
page is on-topic are all **policy** — supplied here, never hardcoded in the router or
service. Defaults are deliberately site-agnostic:

* ``dedup_key`` — exact-URL identity.
* ``link_filter`` — scheme + the request's own domain allow/deny (generic).
* ``url_scorer`` — none (plain breadth-first).
* ``judge`` — keep everything.

Niche, site-specific behaviours live here as **opt-in** strategies a caller composes:
:func:`collapsed_version_key` (versioned docs) and :func:`make_locale_filter`
(i18n path sites). They are not the default, because not every site has them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

from adh.discovery.junk import is_junk_url

from .contracts import CrawlRequest
from .relevance import KeepAllJudge, RelevanceJudge
from .scoring import KeywordRelevanceScorer, NullScorer, UrlScorer

# A dedup key collapses URLs that should count as the same page.
DedupKey = Callable[[str], str]
# A link filter decides whether a candidate URL is eligible for the frontier.
LinkFilter = Callable[[str, CrawlRequest], bool]

_VERSION_RE = re.compile(r"^(v?\d+(\.\d+)*|dev|stable|latest)$")
_LOCALE_RE = re.compile(r"^[a-z]{2}(-[a-z]{2,4})?$")


# --------------------------------------------------------------------- dedup keys
def _canon(url: str, *, collapse_versions: bool = False) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    raw = [s for s in parsed.path.split("/") if s]
    segments = ["_v" if (collapse_versions and _VERSION_RE.match(s.lower())) else s for s in raw]
    path = ("/" + "/".join(segments)).rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{host}{path}{query}"


def exact_url_key(url: str) -> str:
    """Default dedup: exact URL (host + path + query), fragment/trailing-slash normalised."""
    return _canon(url)


def collapsed_version_key(url: str) -> str:
    """Opt-in dedup for versioned docs: ``/en/6.0/x`` and ``/en/5.1/x`` -> one key.

    Stops a version selector from fanning the frontier into the same page across every
    documentation version. Use only for sites with a ``/<version>/`` path segment.
    """
    return _canon(url, collapse_versions=True)


# --------------------------------------------------------------------- link filters
def domain_filter(url: str, seed: CrawlRequest) -> bool:
    """Generic eligibility: http(s) scheme + the request's domain allow/deny lists."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    excl = seed.exclude_domains or ()
    if any(host == d.lower() or host.endswith("." + d.lower()) for d in excl):
        return False
    incl = seed.include_domains or ()
    if incl and not any(host == d.lower() or host.endswith("." + d.lower()) for d in incl):
        return False
    if is_junk_url(url, keywords=seed.keywords):
        return False
    return True


def make_locale_filter() -> LinkFilter:
    """Opt-in filter for i18n path sites: keep only the request's ``allowed_locales``.

    A leading path segment that *looks* like a locale code (``/fr/``, ``/zh-hans/``) and
    is not in ``seed.allowed_locales`` is rejected; non-localised paths pass untouched.
    Composed with :func:`domain_filter` by the adapter when the seed declares languages.
    """

    def _filter(url: str, seed: CrawlRequest) -> bool:
        locales = seed.allowed_locales or ()
        if not locales:
            return True
        segments = [s for s in urlparse(url).path.split("/") if s]
        if not segments:
            return True
        lead = segments[0].lower()
        if not _LOCALE_RE.match(lead):
            return True  # not a locale-structured path; don't second-guess it
        return any(lead == loc.lower() for loc in locales)

    return _filter


def compose_filters(*filters: LinkFilter) -> LinkFilter:
    """AND several link filters into one."""

    def _filter(url: str, seed: CrawlRequest) -> bool:
        return all(f(url, seed) for f in filters)

    return _filter


# --------------------------------------------------------------------- the policy
@dataclass
class FrontierPolicy:
    """How the router dedups, filters, prioritises, and gates a deep crawl.

    Every field has a site-agnostic default; callers opt into stronger steering.
    """

    dedup_key: DedupKey = exact_url_key
    link_filter: LinkFilter = domain_filter
    # Pre-fetch URL/anchor scoring: orders the frontier and prunes below url_threshold.
    url_scorer: UrlScorer = field(default_factory=NullScorer)
    url_threshold: float = 0.0
    # Post-fetch content relevance: a page scoring below content_threshold (against
    # ``query``) is dropped and its links are not expanded.
    judge: RelevanceJudge = field(default_factory=KeepAllJudge)
    content_threshold: float = 0.0
    query: str = ""


def assemble_policy(
    request: CrawlRequest,
    *,
    judge: RelevanceJudge | None = None,
    content_threshold: float = 0.0,
    score_urls_by_keyword: bool = False,
    url_threshold: float = 0.0,
    collapse_versions: bool = False,
    pin_locales: bool = False,
) -> FrontierPolicy:
    """Compose a :class:`FrontierPolicy` for one seed from strategy toggles.

    The single assembly point shared by the in-process adapter
    (:class:`Crawl4AIContentFetcher`) and the crawl microservice, so the two can
    never drift. The request contributes its data (keywords, allowed locales);
    the toggles pick which strategies to apply.
    """
    link_filter = domain_filter
    if pin_locales:
        link_filter = compose_filters(domain_filter, make_locale_filter())
    scorer: UrlScorer = (
        KeywordRelevanceScorer.from_seed(request.url, request.keywords)
        if score_urls_by_keyword
        else NullScorer()
    )
    return FrontierPolicy(
        dedup_key=collapsed_version_key if collapse_versions else exact_url_key,
        link_filter=link_filter,
        url_scorer=scorer,
        url_threshold=url_threshold,
        judge=judge or KeepAllJudge(),
        content_threshold=content_threshold,
        query=" ".join(request.keywords),
    )
