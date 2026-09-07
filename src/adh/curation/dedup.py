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
        if self.method == "exact":
            return self._exact([self._norm(t) for t in texts])
        return self._minhash(texts)

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

    # ------------------------------------------------------------------ exact
    def _exact(self, normalized: list[str]) -> list[DedupAssignment]:
        seen: dict[str, str] = {}
        out: list[DedupAssignment] = []
        for text in normalized:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cid = seen.get(digest)
            if cid is None:
                cid = f"clu-{digest[:12]}"
                seen[digest] = cid
                out.append(DedupAssignment(cid, False))
            else:
                out.append(DedupAssignment(cid, True))
        return out

    # ------------------------------------------------------------------ minhash
    def _minhash(self, texts: list[str]) -> list[DedupAssignment]:
        shingles = [self._shingles(self._norm(t)) for t in texts]
        try:
            return self._minhash_datasketch(texts, shingles)
        except ImportError:
            logger.info("datasketch not installed; using pure-Python near-dedup fallback")
            return self._minhash_pure(texts, shingles)

    def _minhash_datasketch(self, texts, shingles) -> list[DedupAssignment]:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=self.jaccard, num_perm=self.num_perm)
        cluster_of: dict[int, str] = {}
        out: list[DedupAssignment] = []
        for i, sh in enumerate(shingles):
            if not sh:
                cid = f"clu-{sha256_hex(texts[i])[:12]}"
                out.append(DedupAssignment(cid, False))
                continue
            m = MinHash(num_perm=self.num_perm)
            for s in sh:
                m.update(s.encode("utf-8"))
            matches = lsh.query(m)
            if matches:
                cid = cluster_of[matches[0]]
                out.append(DedupAssignment(cid, True))
            else:
                cid = f"clu-{sha256_hex(texts[i])[:12]}"
                out.append(DedupAssignment(cid, False))
            lsh.insert(i, m)
            cluster_of[i] = cid
        return out

    def _minhash_pure(self, texts, shingles) -> list[DedupAssignment]:
        reps: list[tuple[int, frozenset[str], str]] = []  # (idx, shingles, cluster_id)
        out: list[DedupAssignment] = []
        for i, sh in enumerate(shingles):
            matched: str | None = None
            for _, rep_sh, cid in reps:
                if sh and rep_sh and _jaccard(sh, rep_sh) >= self.jaccard:
                    matched = cid
                    break
            if matched is not None:
                out.append(DedupAssignment(matched, True))
            else:
                cid = f"clu-{sha256_hex(texts[i])[:12]}"
                reps.append((i, sh, cid))
                out.append(DedupAssignment(cid, False))
        return out


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
