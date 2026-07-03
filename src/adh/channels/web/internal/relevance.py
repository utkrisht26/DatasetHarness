"""Content-relevance judges — generic, site-agnostic drift control.

URL/anchor heuristics (:mod:`scoring`) are cheap but brittle: they assume the site
encodes topic in the path. A :class:`RelevanceJudge` instead scores a page's *fetched
content* against the topic, so it generalises to any site. The router applies it after
fetching: a page below the threshold is dropped and its links are not expanded.

Implementations:

* :class:`Bm25RelevanceJudge` — BM25 via ``rank_bm25`` (the library Crawl4AI already
  bundles; reuse, not reinvention). The page is chunked into sentences (the BM25
  corpus); relevance = the saturating mean of the top-k chunk scores for the query.
  BM25 needs a corpus, so it is meaningful on real multi-chunk pages and weak on a
  couple of sentences — fine for the live channel, which fetches whole pages.
* :class:`LlmRelevanceJudge` — provider-generic via LangChain ``init_chat_model``
  (``"anthropic:…"``, ``"openai:…"``, …). Most reliable; costs a model call per page.
* :class:`KeepAllJudge` — the default (no gating).

All judges subclass :class:`RelevanceJudge` and return a relevance in ``[0, 1]``.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any

from adh.llm.prompts import RELEVANCE_JUDGE
from adh.llm.usage import UsageTracker

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CHUNK_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_FLOAT_RE = re.compile(r"\d*\.?\d+")


def _stem(token: str) -> str:
    """Crude singularizer so ``views``/``view`` and ``mixins``/``mixin`` match."""
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]


def _chunks(content: str) -> list[list[str]]:
    """Split content into sentence/line chunks of terms (the BM25 corpus)."""
    return [terms for terms in (_terms(c) for c in _CHUNK_RE.split(content)) if terms]


class RelevanceJudge(abc.ABC):
    """Scores a fetched page's topical relevance to ``query`` in ``[0.0, 1.0]``."""

    @abc.abstractmethod
    def score(self, query: str, content: str) -> float:
        """Higher == more relevant. Empty query should return 1.0 (no gating)."""
        raise NotImplementedError


class KeepAllJudge(RelevanceJudge):
    """Default judge: every page is relevant (no content gating)."""

    def score(self, query: str, content: str) -> float:
        return 1.0


@dataclass
class Bm25RelevanceJudge(RelevanceJudge):
    """BM25-weighted topic coverage via ``rank_bm25`` (Crawl4AI's bundled BM25).

    Interpretable and calibrated: the score is the fraction of the query vocabulary
    the page covers, where each term is weighted by the fitted BM25 model's IDF
    (rare terms count most; a query term absent from the page costs the max observed
    IDF) and each term's presence is its best TF-saturated hit across the page's
    sentence chunks, capped at 1. A page covering the whole topic vocabulary scores
    ~1; a page sharing only generic words scores low. Live calibration on
    djangoproject.com: on-topic 0.49-0.70, off-topic (/foundation, /fundraising,
    trac) 0.10-0.30 — so ~0.35 is a sound default threshold.

    Empty query == 1.0 (no gating); no page content == 0.0.
    """

    def score(self, query: str, content: str) -> float:
        q = list(dict.fromkeys(_terms(query)))  # distinct query terms, stemmed
        if not q:
            return 1.0
        chunks = _chunks(content)
        if not chunks:
            return 0.0
        from rank_bm25 import BM25Okapi  # bundled with crawl4ai (the `web` extra)

        bm = BM25Okapi(chunks)
        max_idf = max(bm.idf.values()) if bm.idf else 1.0
        weight = {t: bm.idf.get(t, max_idf) for t in q}
        total = sum(weight.values()) or 1.0
        presence = dict.fromkeys(q, 0.0)  # per-term best TF-saturated hit on the page
        for freqs, dl in zip(bm.doc_freqs, bm.doc_len):
            for t in q:
                f = freqs.get(t, 0)
                if not f:
                    continue
                tf_sat = (f * (bm.k1 + 1)) / (f + bm.k1 * (1 - bm.b + bm.b * dl / bm.avgdl))
                presence[t] = max(presence[t], min(1.0, tf_sat))
        return sum(weight[t] * presence[t] for t in q) / total


@dataclass
class LlmRelevanceJudge(RelevanceJudge):
    """Relevance via a LangChain chat model — provider-generic and testable.

    ``model`` is any LangChain ``BaseChatModel`` (or any object with ``invoke(str)``
    returning a message whose ``.content`` is the text). Build one for any provider
    with :meth:`from_model_id` (``"anthropic:claude-…"``, ``"openai:gpt-…"``, …), or
    inject a fake in tests. Content is truncated to ``max_chars`` to bound cost. Each
    call's token usage is recorded into ``usage`` (share one tracker across judges to
    get a run total; read ``usage.total`` for cost reporting / ``max_llm_tokens``).
    """

    model: Any
    max_chars: int = 4000
    prompt: str = RELEVANCE_JUDGE
    usage: UsageTracker = field(default_factory=UsageTracker)

    @classmethod
    def from_model_id(
        cls, model_id: str, *, model_kwargs: dict[str, Any] | None = None, **judge_kwargs: Any
    ) -> "LlmRelevanceJudge":
        """Build a judge over a provider-generic model (``"anthropic:…"``, ``"openai:…"``).

        Uses the shared :func:`adh.llm.build_chat_model` factory so model construction
        is identical everywhere. ``model_kwargs`` configure the model (``temperature``,
        …); remaining kwargs (``max_chars``, ``prompt``) configure the judge.
        """
        from adh.llm import build_chat_model

        model = build_chat_model(model_id, **(model_kwargs or {}))
        return cls(model=model, **judge_kwargs)

    def score(self, query: str, content: str) -> float:
        if not query.strip():
            return 1.0
        message = self.model.invoke(self.prompt.format(query=query, content=content[: self.max_chars]))
        self.usage.add_message(message)  # count tokens for this call
        text = getattr(message, "content", message)
        return _parse_score(str(text))


def _parse_score(text: str) -> float:
    """Extract the first number from a model reply and clamp to ``[0, 1]``."""
    match = _FLOAT_RE.search(text)
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group())))
