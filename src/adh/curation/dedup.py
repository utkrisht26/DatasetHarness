"""Near-duplicate detection (OUTLINE §2.4) — a first-class subsystem, not a footnote.

Assigns every candidate a ``dedup_cluster_id`` and an ``is_duplicate`` flag (the first
member of a cluster is the representative; the rest are duplicates). ``minhash`` uses
MinHash + LSH banding (``datasketch`` when installed, a pure-Python MinHash fallback
otherwise) at the job's ``jaccard`` threshold; ``exact`` clusters by normalized-text
hash. Scope defaults to per-shard — global MinHash can delete higher-quality recurring
content (RESEARCH §QC), so :func:`dedupe` warns when asked to run globally.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from adh.provenance import sha256_hex

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class DedupAssignment:
    cluster_id: str
    is_duplicate: bool


@dataclass
class Deduper:
    """Clusters near-duplicate texts; order-stable (first seen is the representative)."""

    method: str = "minhash"
    ngram: int = 5
    jaccard: float = 0.8
    num_perm: int = 128

    def _norm(self, text: str) -> str:
        return " ".join((text or "").lower().split())

    def _shingles(self, text: str) -> frozenset[str]:
        tokens = _WORD_RE.findall(text)
        if len(tokens) < self.ngram:
            return frozenset({" ".join(tokens)}) if tokens else frozenset()
        return frozenset(
            " ".join(tokens[i : i + self.ngram]) for i in range(len(tokens) - self.ngram + 1)
        )

    def cluster(self, texts: list[str]) -> list[DedupAssignment]:
        index = self.index()
        return [index.add(t) for t in texts]

    def index(self) -> DedupIndex:
        """An empty incremental index applying this deduper's policy."""
        return DedupIndex(self)

    def cluster_per_split(self, items: list[tuple[str, str]]) -> list[DedupAssignment]:
        """Cluster ``(split, text)`` pairs independently within each split.

        A duplicate never spans shards: identical text landing in two splits yields
        two representatives (one per shard), so eval splits keep their own copies of
        recurring content (RESEARCH §QC). Cluster ids carry a ``"<split>:"`` prefix;
        input order is preserved in the output.
        """
        by_split: dict[str, list[int]] = {}
        for i, (split, _text) in enumerate(items):
            by_split.setdefault(split, []).append(i)
        out: list[DedupAssignment | None] = [None] * len(items)
        for split, idxs in by_split.items():
            for i, assign in zip(idxs, self.cluster([items[i][1] for i in idxs])):
                out[i] = DedupAssignment(f"{split}:{assign.cluster_id}", assign.is_duplicate)
        return [a for a in out if a is not None]


class DedupIndex:
    """Incremental near-dedup: ``add`` texts one at a time, first seen is the representative.

    Batch :meth:`Deduper.cluster` is this index fed in order, so a running count of
    distinct texts always agrees with what a later batch clustering would keep.
    """

    def __init__(self, deduper: Deduper) -> None:
        self._d = deduper
        self._n = 0
        self.n_distinct = 0
        self._exact_seen: dict[str, str] = {}
        self._lsh = None
        self._cluster_of: dict[int, str] = {}
        self._reps: list[tuple[frozenset[str], str]] = []  # pure-Python fallback
        if deduper.method != "exact":
            try:
                from datasketch import MinHashLSH

                self._lsh = MinHashLSH(threshold=deduper.jaccard, num_perm=deduper.num_perm)
            except ImportError:
                logger.info("datasketch not installed; using pure-Python near-dedup fallback")

    def add(self, text: str) -> DedupAssignment:
        if self._d.method == "exact":
            assignment = self._add_exact(text)
        elif self._lsh is not None:
            assignment = self._add_datasketch(text)
        else:
            assignment = self._add_pure(text)
        self._n += 1
        self.n_distinct += not assignment.is_duplicate
        return assignment

    def _add_exact(self, text: str) -> DedupAssignment:
        digest = hashlib.sha256(self._d._norm(text).encode("utf-8")).hexdigest()
        cid = self._exact_seen.get(digest)
        if cid is not None:
            return DedupAssignment(cid, True)
        cid = f"clu-{digest[:12]}"
        self._exact_seen[digest] = cid
        return DedupAssignment(cid, False)

    def _add_datasketch(self, text: str) -> DedupAssignment:
        from datasketch import MinHash

        shingles = self._d._shingles(self._d._norm(text))
        if not shingles:
            return DedupAssignment(f"clu-{sha256_hex(text)[:12]}", False)
        m = MinHash(num_perm=self._d.num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        matches = self._lsh.query(m)
        if matches:
            assignment = DedupAssignment(self._cluster_of[matches[0]], True)
        else:
            assignment = DedupAssignment(f"clu-{sha256_hex(text)[:12]}", False)
        self._lsh.insert(self._n, m)
        self._cluster_of[self._n] = assignment.cluster_id
        return assignment

    def _add_pure(self, text: str) -> DedupAssignment:
        shingles = self._d._shingles(self._d._norm(text))
        for rep_sh, cid in self._reps:
            if shingles and rep_sh and _jaccard(shingles, rep_sh) >= self._d.jaccard:
                return DedupAssignment(cid, True)
        cid = f"clu-{sha256_hex(text)[:12]}"
        self._reps.append((shingles, cid))
        return DedupAssignment(cid, False)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dedupe(texts: list[str], *, policy) -> list[DedupAssignment]:
    """Cluster ``texts`` per a contract ``DedupPolicy``; warn on global scope."""
    if getattr(policy, "scope", "per-shard") == "global":
        logger.warning(
            "dedup scope='global' can delete higher-quality recurring content; "
            "per-shard is recommended (RESEARCH §QC)"
        )
    deduper = Deduper(
        method=getattr(policy, "method", "minhash"),
        ngram=getattr(policy, "ngram", 5),
        jaccard=getattr(policy, "jaccard", 0.8),
    )
    return deduper.cluster(texts)
