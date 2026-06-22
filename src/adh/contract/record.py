"""The Record Schema — one emitted record, payload + full lineage.

Faithful port of PHASE0-CONTRACT.md §0.4.2. Provenance fields map onto W3C PROV
(Entity/Activity/Agent) and MLMD (Artifact/Execution/Context). Validation runs at
the emission boundary; invalid records are quarantined to a dead-letter store with
the ``ValidationError`` attached, never silently dropped (§0.5).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import AnyUrl, BaseModel, ConfigDict, Field

from .enums import SourceChannel, VerifiabilityTier


class VerifierResult(BaseModel):
    """Outcome of a Tier-1/3 deterministic verifier."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    detail: str | None = None


class Record(BaseModel):
    """One JSONL line / Parquet row.

    The ``input``/``output`` payload shape is the Phase 0 canonical pair (§0.8.1).
    ``source_url`` is ``None`` for synthesis-channel records.
    """

    model_config = ConfigDict(extra="forbid")

    # --- identity ---
    record_id: str
    job_id: str
    split: str

    # --- payload ---
    input: str
    output: str

    # --- source lineage (PROV wasDerivedFrom / used) ---
    source_channel: SourceChannel = SourceChannel.WEB
    source_url: AnyUrl | None = None  # null for synthesis
    source_license: str = "unknown"
    source_sha256: str | None = None
    derived_from: list[str] = Field(default_factory=list)

    # --- generation lineage (PROV wasGeneratedBy / wasAttributedTo) ---
    model_id: str
    prompt_hash: str
    system_prompt_hash: str | None = None
    generation_params: dict = Field(default_factory=dict)
    generated_at: datetime
    pipeline_step: str

    # --- quality + dedup ---
    verifiability_tier: VerifiabilityTier
    verifier_result: VerifierResult | None = None  # required for tier 1/3
    judge_scores: dict[str, float] | None = None  # required for tier 2/3
    judge_models: list[str] = Field(default_factory=list)
    dedup_cluster_id: str | None = None
    is_duplicate: bool = False
    contamination_flag: bool = False

    # --- accounting ---
    input_tokens: int = 0
    output_tokens: int = 0
    language: str = "en"
    task_category: str | None = None
    cost_usd: float = 0.0
