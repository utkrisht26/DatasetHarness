"""Provenance & lineage helpers (Phase 3.3 — minimal slice for Engine A auditability).

Per Phase 0 open decision §0.8.2 (recommendation adopted): per-record/per-chunk
provenance is carried as flat columns now, plus a ``lineage.json`` export shaped as a
W3C PROV graph (Entity / Activity / Agent + ``wasGeneratedBy`` / ``used`` /
``wasDerivedFrom``). MLMD is deferred until cross-run artifact tracking is needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # avoid a runtime import cycle (corpus -> provenance -> corpus)
    from adh.corpus.models import CorpusChunk


def sha256_hex(data: str | bytes) -> str:
    """Hex SHA-256 of text/bytes (Croissant ``FileObject.sha256`` convention)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def sha256_prefixed(data: str | bytes) -> str:
    """``sha256:<hex>`` form used for ``prompt_hash`` / ``system_prompt_hash``."""
    return f"sha256:{sha256_hex(data)}"


def approx_token_count(text: str) -> int:
    """Whitespace-token approximation. Deliberately simple; replace with a real

    tokenizer when token-budget accounting must be exact (accounting only).
    """
    return len(text.split())


def build_lineage(
    job_id: str,
    chunks: Iterable["CorpusChunk"],
    *,
    exit_reason: str | None = None,
) -> dict:
    """Build a PROV-style lineage graph for an Engine A corpus.

    * each corpus chunk -> a PROV *entity*
    * each distinct source -> a PROV *entity* the chunk ``wasDerivedFrom``
    * each channel -> a PROV *agent* that the fetch activity ``wasAssociatedWith``
    """
    entities: dict[str, dict] = {}
    agents: dict[str, dict] = {}
    activities: dict[str, dict] = {}

    for ch in chunks:
        agent_id = f"agent:channel:{ch.source_channel.value}"
        agents.setdefault(agent_id, {"id": agent_id, "type": "channel"})

        activity_id = f"activity:fetch:{ch.source_channel.value}:{ch.aspect_id}"
        activities.setdefault(
            activity_id,
            {
                "id": activity_id,
                "type": "acquisition",
                "wasAssociatedWith": agent_id,
                "aspect_id": ch.aspect_id,
            },
        )

        source_entity_id = None
        if ch.source_url:
            source_entity_id = f"entity:source:{ch.source_sha256[:16]}"
            entities.setdefault(
                source_entity_id,
                {
                    "id": source_entity_id,
                    "type": "source",
                    "url": ch.source_url,
                    "license": ch.source_license,
                    "sha256": ch.source_sha256,
                },
            )

        entities[ch.chunk_id] = {
            "id": ch.chunk_id,
            "type": "corpus_chunk",
            "job_id": ch.job_id,
            "wasGeneratedBy": activity_id,
            "wasAttributedTo": agent_id,
            "wasDerivedFrom": ([source_entity_id] if source_entity_id else list(ch.derived_from)),
            "content_uri": ch.content_uri,
            "source_channel": ch.source_channel.value,
            "n_tokens": ch.n_tokens,
            "generatedAtTime": ch.fetched_at.isoformat(),
        }

    return {
        "job_id": job_id,
        "prov": "http://www.w3.org/ns/prov#",
        "exit_reason": exit_reason,
        "agents": list(agents.values()),
        "activities": list(activities.values()),
        "entities": list(entities.values()),
        "counts": {
            "entities": len(entities),
            "agents": len(agents),
            "activities": len(activities),
        },
    }


def write_lineage(path: str | Path, lineage: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lineage, indent=2, sort_keys=False), encoding="utf-8")
    return p
