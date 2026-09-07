"""The Job Spec — what a requester submits (PHASE0-CONTRACT.md §0.3).

This is a faithful, machine-enforced port of the LOCKED contract. The hard,
*structural* cross-field invariants from §0.3 are enforced here at construction
time (fail-fast). Invariants that need filesystem/schema access (e.g. the Tier-1
``x-verifier`` hook) live in :mod:`adh.contract.validation` and run at job-load time.
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import DiscoveryMode, SourceChannel, VerifiabilityTier
from .families import distinct_families
from .licenses import incompatible_licenses

# Splits that must never contain contaminated data — decontamination is mandatory
# before anything enters them (PHASE0-CONTRACT.md §0.3 invariant).
EVAL_SPLIT_NAMES = frozenset({"test", "validation"})


class DiscoveryConstraints(BaseModel):
    """Optional guardrails — NOT a list of where to look."""

    model_config = ConfigDict(extra="forbid")

    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    recency_after: str | None = None
    respect_robots_txt: bool = True

    # --- media acquisition filters (media modalities: video/image); the
    # authoritative integrity bounds live in the record schema's x-verifier hook ---
    media_min_duration_s: float | None = None
    media_max_duration_s: float | None = None
    media_orientation: str | None = None  # landscape | portrait | square
    # Requested asset types ("video", "image", "gif"). None = infer from the spec's
    # modality via resolved_media_types(); an explicit list overrides that default.
    media_types: list[str] | None = None


# Asset types implied by each media modality when constraints.media_types is unset.
_MEDIA_TYPES_BY_MODALITY: dict[str, frozenset[str]] = {
    "video": frozenset({"video"}),
    "image": frozenset({"image", "gif"}),
}


def resolved_media_types(
    modality: str, *, constraints: DiscoveryConstraints | None = None
) -> frozenset[str]:
    """Which asset types a job wants from the MEDIA channel.

    Defaults infer from the modality (video→{video}, image→{image, gif}, others→∅);
    an explicit ``constraints.media_types`` overrides the inference verbatim
    (normalized lowercase). Old specs without the field keep their exact meaning.
    """
    explicit = constraints.media_types if constraints is not None else None
    if explicit:
        return frozenset(t.strip().lower() for t in explicit if t.strip())
    return _MEDIA_TYPES_BY_MODALITY.get((modality or "").strip().lower(), frozenset())


class Discovery(BaseModel):
    """WHERE to get the data — the system figures this out (§0.3 'Discovery-first')."""

    model_config = ConfigDict(extra="forbid")

    mode: DiscoveryMode = DiscoveryMode.AUTO
    seeds: list[str] = Field(default_factory=list)  # OPTIONAL hints, never required
    approval_gate: bool = True  # review ranked sources before fetch (False = autonomous)
    channels: list[SourceChannel] = Field(
        default_factory=lambda: [
            SourceChannel.DATASETS,
            SourceChannel.WEB,
            SourceChannel.SYNTHESIS,
        ]
    )
    constraints: DiscoveryConstraints = Field(default_factory=DiscoveryConstraints)
    # Two-layer license policy. Acquisition defaults to known-free licenses; this
    # flag opts a job into open-web sources whose license is unknown/absent (hits
    # are recorded as "unknown", never invented). Distribution consent is separate:
    # publishing publicly while unknown-license assets are present still requires
    # its own explicit override at publish time.
    allow_unknown_licenses: bool = False

    @model_validator(mode="after")
    def _check_mode_seeds(self) -> Discovery:
        # §0.3: fixed mode needs the explicit sources.
        if self.mode == DiscoveryMode.FIXED and not self.seeds:
            raise ValueError("discovery.mode='fixed' requires a non-empty discovery.seeds list")
        if not self.channels:
            raise ValueError("discovery.channels must list at least one channel")
        return self

    @property
    def is_synthesis_only(self) -> bool:
        """§0.3: channels == ['synthesis'] => no acquisition runs; corpus stays empty."""
        return self.channels == [SourceChannel.SYNTHESIS]


class JudgeConfig(BaseModel):
    """LLM-jury configuration. MUST be >=2 disjoint families for Tier 2/3 (PoLL)."""

    model_config = ConfigDict(extra="forbid")

    models: list[str]
    rubric_ref: str | None = None
    min_score: float = Field(3.5, ge=0, le=5)
    position_swap: bool = True


class Limits(BaseModel):
    """Circuit-breaker floor. Coverage-saturation is the *primary* stop signal."""

    model_config = ConfigDict(extra="forbid")

    max_pages_scraped: int = Field(5_000, ge=0)
    max_llm_tokens: int = Field(50_000_000, ge=0)
    max_storage_mb: int = Field(20_000, ge=0)
    max_runtime_seconds: int = Field(28_800, ge=0)
    max_cost_usd: float = Field(250.0, ge=0)


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "minhash"  # minhash | exact | semantic
    ngram: int = Field(5, ge=1)
    jaccard: float = Field(0.8, ge=0, le=1)
    scope: str = "per-shard"  # per-shard (recommended) | global


class TargetSize(BaseModel):
    """Either a record count or a token budget (§0.3 ``target_size``)."""

    model_config = ConfigDict(extra="forbid")

    records: int | None = Field(default=None, ge=1)
    tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> TargetSize:
        if (self.records is None) == (self.tokens is None):
            raise ValueError("target_size must set exactly one of {records, tokens}")
        return self


class JobSpec(BaseModel):
    """A request to build a dataset. The requester gives a goal — not sources."""

    model_config = ConfigDict(extra="forbid")

    job_id: str | None = None
    name: str
    description: str
    motivation: str
    intended_uses: list[str]
    out_of_scope_uses: list[str] = Field(default_factory=list)

    # Task & content shape
    verifiability_tier: VerifiabilityTier
    task_categories: list[str]
    languages: list[str]
    modality: str = "text"
    target_size: TargetSize
    record_schema_ref: str

    # Discovery (sources are DISCOVERED, not required from the user)
    discovery: Discovery = Field(default_factory=Discovery)
    allowed_licenses: list[str]
    # None = contract default ("redact", unenforced-with-loud-degradation when the
    # presidio extra is missing); an explicit value fail-closes instead. Explicitness
    # must live in the VALUE (not pydantic's model_fields_set) because specs are
    # persisted via a complete model_dump() — a round-trip would otherwise mark every
    # defaulted field as user-configured.
    pii_policy: Literal["redact", "drop", "allow"] | None = None

    # Generation config
    generator_models: list[str]
    judge_config: JudgeConfig | None = None
    generation_params: dict = Field(default_factory=dict)
    diversity_strategy: str = "persona+evol"
    dedup_policy: DedupPolicy = Field(default_factory=DedupPolicy)
    decontaminate_against: list[str] = Field(default_factory=list)

    # Output & distribution
    output_format: str = "parquet"  # jsonl | parquet
    # Trainer-facing bundle layout, resolved against the delivery-layout registry
    # (adh.delivery.layouts): "hf" by default, "ltx" for LTX-Video-style trainers;
    # validated in check_job_spec.
    delivery_layout: str = "hf"
    splits: dict = Field(default_factory=lambda: {"train": 0.9, "validation": 0.05, "test": 0.05})
    output_license: str
    distribution_target: str | None = None
    maintainer: str | None = None
    emit_datasheet: bool = True

    limits: Limits = Field(default_factory=Limits)

    # ----- hard cross-field invariants (PHASE0-CONTRACT.md §0.3) -----
    @model_validator(mode="after")
    def _enforce_invariants(self) -> JobSpec:
        tier = self.verifiability_tier

        # tier in {2,3} => judge panel from >=2 distinct model families (PoLL)
        if tier in (VerifiabilityTier.SOFT, VerifiabilityTier.HYBRID):
            if self.judge_config is None:
                raise ValueError(
                    f"verifiability_tier={int(tier)} requires judge_config (LLM-jury)"
                )
            families = distinct_families(self.judge_config.models)
            if len(families) < 2:
                raise ValueError(
                    f"verifiability_tier={int(tier)} requires a judge panel from >=2 distinct "
                    f"model families (PoLL); got {sorted(families) or 'none'}"
                )

        # output_license must be compatible with every retained license
        if self.output_license:
            bad = incompatible_licenses(self.allowed_licenses, self.output_license)
            if bad:
                raise ValueError(
                    f"output_license={self.output_license!r} is incompatible with "
                    f"allowed_licenses {bad}; remove them or change output_license"
                )

        # any eval split => decontamination is mandatory
        if self._has_eval_split() and not self.decontaminate_against:
            raise ValueError(
                "splits include an eval split (test/validation) so decontaminate_against "
                "must be non-empty (decontamination is mandatory before any eval split)"
            )

        # splits must form a sane partition
        self._check_splits()

        # global dedup scope => warn (can delete higher-quality recurring content)
        if self.dedup_policy.scope == "global":
            warnings.warn(
                "dedup_policy.scope='global' can delete higher-quality recurring content "
                "(FineWeb 490B->31B); per-shard is recommended.",
                stacklevel=2,
            )

        return self

    def _has_eval_split(self) -> bool:
        return any(
            name in EVAL_SPLIT_NAMES and float(frac) > 0 for name, frac in self.splits.items()
        )

    def _check_splits(self) -> None:
        if not self.splits:
            raise ValueError("splits must be non-empty")
        try:
            total = sum(float(v) for v in self.splits.values())
        except (TypeError, ValueError) as exc:
            raise ValueError("splits values must be numeric fractions") from exc
        if any(float(v) < 0 for v in self.splits.values()):
            raise ValueError("splits fractions must be non-negative")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits fractions must sum to 1.0 (got {total})")
