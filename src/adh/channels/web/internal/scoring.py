"""Relevance scoring for the deep-crawl frontier (the "intelligent" in the crawler).

Blind breadth-first crawling drifts off-topic: from a Django *views* page at depth 3
the frontier wanders into ``/download/``, ``/fundraising/``, the whole site TOC. The
fix (RESEARCH §APIs — BestFirstCrawlingStrategy / ``url_scorer`` / ``score_threshold``)
is to score every candidate link for topical relevance and crawl best-first, pruning
links below a threshold.

This module is deliberately dependency-free (no crawl4ai, no ML): a transparent,
offline-testable heuristic over the signals available *before* fetching a page —
the link's URL, its anchor text, and the seed it descended from. A scorer is a plain
:class:`UrlScorer`; swap in a BM25 / embedding / LLM scorer later without touching the
router.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from urllib.parse import urlparse

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens of length >= 2 (drops separators and single chars)."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2}


def _stem(token: str) -> str:
    """Crude singularizer so ``views`` and ``view`` match."""
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


class UrlScorer(abc.ABC):
    """Scores a candidate outlink's topical relevance in ``[0.0, 1.0]`` (pre-fetch)."""

    @abc.abstractmethod
    def score(self, url: str, *, anchor: str = "") -> float:
        """Higher == more relevant. ``0.0`` means "do not crawl"."""
        raise NotImplementedError


class NullScorer(UrlScorer):
    """Scores everything 1.0 — plain breadth-first, no pruning (back-compat default)."""

    def score(self, url: str, *, anchor: str = "") -> float:
        return 1.0


@dataclass
class KeywordRelevanceScorer(UrlScorer):
    """Heuristic relevance over URL tokens + anchor text vs an aspect vocabulary.

    Combines three cheap, explainable signals:

    * **host affinity** — links off the seed's host (e.g. ``www.`` vs ``docs.``)
      are heavily penalised; cross-host drift is the main cause of ``/fundraising/``.
    * **keyword hits** — aspect keywords (stemmed) appearing in the URL path or
      anchor text. The dominant signal: a ``/ref/views/`` link reads as relevant
      even though it lives in a different doc section than the seed.
    * **path affinity** — how many leading path segments the link shares with the
      seed, so siblings under the seed's directory get a boost.
    """

    keywords: frozenset[str]
    seed_host: str
    seed_segments: tuple[str, ...]
    keyword_weight: float = 0.7
    path_weight: float = 0.3
    off_host_factor: float = 0.0  # 0 == never leave the seed host

    @classmethod
    def from_seed(
        cls,
        seed_url: str,
        keywords: "tuple[str, ...] | frozenset[str]" = (),
        **kw,
    ) -> "KeywordRelevanceScorer":
        parsed = urlparse(seed_url)
        seed_segments = tuple(s for s in parsed.path.lower().split("/") if s)
        # Vocabulary = explicit aspect keywords + the seed's *leaf* path segment
        # (the topic, e.g. "views"), stemmed. We deliberately exclude the structural
        # prefix ("en", "6.0", "topics", "http") — those are scaffolding shared by
        # every page, so treating them as keywords would match unrelated sections.
        # Structural proximity is captured separately by path affinity below.
        vocab = {_stem(t) for t in tokenize(" ".join(keywords))}
        if seed_segments:
            vocab |= {_stem(t) for t in tokenize(seed_segments[-1])}
        return cls(
            keywords=frozenset(v for v in vocab if v),
            seed_host=(parsed.hostname or "").lower(),
            seed_segments=seed_segments,
            **kw,
        )

    def score(self, url: str, *, anchor: str = "") -> float:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return 0.0
        host = (parsed.hostname or "").lower()
        host_factor = 1.0 if host == self.seed_host else self.off_host_factor
        if host_factor == 0.0:
            return 0.0

        segments = tuple(s for s in parsed.path.lower().split("/") if s)
        # Keyword match on the URL path only. Anchor text is captured (and a richer
        # scorer may use it) but it carries site-name noise — a footer "Django docs"
        # link would match the framework name and read as on-topic — so the default
        # scorer keeps to the path, which is reliably topical in documentation sites.
        tokens = {_stem(t) for t in tokenize(" ".join(segments))}

        keyword_score = 0.0
        if self.keywords:
            hits = len(tokens & self.keywords)
            keyword_score = min(1.0, hits / 2.0)  # 2+ matching terms == full marks

        shared = 0
        for seed_seg, seg in zip(self.seed_segments, segments):
            if seed_seg == seg:
                shared += 1
            else:
                break
        path_score = shared / max(len(self.seed_segments), 1)

        relevance = self.keyword_weight * keyword_score + self.path_weight * path_score
        return host_factor * min(1.0, relevance)
