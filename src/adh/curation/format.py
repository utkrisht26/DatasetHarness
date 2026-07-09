"""Label / structure / format (OUTLINE §2.5, PHASE0-CONTRACT §0.5).

Assembles a verified+curated candidate into the canonical :class:`~adh.contract.Record`,
assigns its split, and validates it at the emission boundary against both the base
Record model and the per-job JSON Schema. Invalid records are **quarantined** to a
dead-letter store with the error attached — never silently dropped. Split assignment
is a stable hash of the record id (reproducible regardless of order/count), and a
``contamination_flag`` record is kept out of any eval split.
"""

from __future__ import annotations

import hashlib

from adh.contract import JobSpec, Record, RecordValidationError, validate_record
from adh.generation import Candidate
from adh.verification import Verdict

from .dedup import DedupAssignment

EVAL_SPLIT_NAMES = frozenset({"validation", "test", "val", "dev", "eval"})


def assign_split(record_id: str, splits: dict, *, contaminated: bool) -> str:
    """Deterministically bucket a record into a split; contaminated -> never eval."""
    names = list(splits) or ["train"]
    total = sum(splits.values()) or 1.0
    h = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16) / 0x1_0000_0000
    cum = 0.0
    chosen = names[-1]
    for name in names:
        cum += splits[name] / total
        if h < cum:
            chosen = name
            break
    if contaminated and chosen.lower() in EVAL_SPLIT_NAMES:
        return "train" if "train" in splits else names[0]
    return chosen


def build_record(
    candidate: Candidate,
    *,
    spec: JobSpec,
    schema: dict | None,
    verdict: Verdict,
    dedup: DedupAssignment,
    contaminated: bool,
) -> Record:
    """Assemble + validate the final Record (raises RecordValidationError -> dead-letter)."""
    record_id = candidate.candidate_id
    split = assign_split(record_id, spec.splits, contaminated=contaminated)
    payload = {
        "record_id": record_id,
        "job_id": candidate.job_id,
        "split": split,
        **candidate.base_record_fields(),
        "verifier_result": verdict.verifier_result,
        "judge_scores": verdict.judge_scores,
        "judge_models": list(verdict.judge_models),
        "dedup_cluster_id": dedup.cluster_id,
        "is_duplicate": dedup.is_duplicate,
        "contamination_flag": contaminated,
    }
    return validate_record(payload, schema=schema)


def dead_letter_row(candidate: Candidate, error: RecordValidationError) -> dict:
    """A quarantine row: the offending payload + why it failed the contract."""
    return {
        "candidate_id": candidate.candidate_id,
        "job_id": candidate.job_id,
        "aspect_id": candidate.aspect_id,
        "error": str(error),
        "input": candidate.input[:500],
        "output": candidate.output[:500],
    }
