"""Generation domain models (OUTLINE §2.1/§2.2).

The seed engine turns aspects + corpus grounding into a structured sampling space of
:class:`SeedPoint` s (persona × difficulty × grounding); the generator turns each
seed into one or more :class:`Candidate` s — pre-verification records carrying their
payload, source/generation lineage, and (for Tier-1 tasks) the oracle inputs
(``reference`` / ``tests``) that are consumed by verification and never shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from adh.contract import MediaRef, SourceChannel, VerifiabilityTier
from adh.provenance import sha256_hex
from adh.verifiers import VerifierRequest

# The structured sampling axes that fight mode collapse (OUTLINE §2.1). Small,
# opinionated defaults; a job can override them via the seed engine.
DEFAULT_PERSONAS = (
    "a curious beginner",
    "an expert practitioner",
    "a meticulous reviewer",
    "a time-pressed engineer",
)
DEFAULT_DIFFICULTIES = ("basic", "intermediate", "advanced")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SeedPoint:
    """One point in the sampling space — what to generate, conditioned and grounded."""

    seed_id: str
    job_id: str
    aspect_id: str
    topic: str
    keywords: tuple[str, ...]
    persona: str
    difficulty: str
    source_channel: SourceChannel
    grounding_uri: str | None = None  # corpus content_uri to retrieve full grounding text
    grounding_preview: str = ""  # context-safe excerpt (used when no store/uri)
    source_url: str | None = None
    source_license: str = "unknown"
    source_sha256: str | None = None
    derived_from: tuple[str, ...] = ()
    # diversity-strategy metadata (OUTLINE §2.1, additive — downstream may ignore):
    # ``evolution_op`` names the evol_instruct operator that produced this seed;
    # ``directive`` is a strategy-specific instruction-shaping hint a Generator MAY
    # fold into its prompt (empty for plain persona-grid seeds).
    evolution_op: str = ""
    directive: str = ""


@dataclass(frozen=True)
class Candidate:
    """A generated, pre-verification record.

    ``reference`` / ``tests`` are oracle inputs for Tier-1 verification; they are
    dropped when the final :class:`~adh.contract.Record` is assembled.
    """

    candidate_id: str
    job_id: str
    aspect_id: str
    verifiability_tier: VerifiabilityTier
    input: str
    output: str
    reference: str | None = None
    tests: str | None = None
    # source lineage
    source_channel: SourceChannel = SourceChannel.SYNTHESIS
    source_url: str | None = None
    source_license: str = "unknown"
    source_sha256: str | None = None
    derived_from: tuple[str, ...] = ()
    # generation lineage
    model_id: str = "heuristic:adh"
    prompt_hash: str = ""
    system_prompt_hash: str | None = None
    generation_params: dict = field(default_factory=dict)
    generated_at: datetime = field(default_factory=_utcnow)
    pipeline_step: str = "text-generation"
    # accounting
    input_tokens: int = 0
    output_tokens: int = 0
    language: str = "en"
    task_category: str | None = None
    cost_usd: float = 0.0
    # media payload (modality extensions): ``media`` ships inside the record;
    # ``staged_path`` is a local-only absolute path consumed by verification and
    # delivery layouts — it is never serialized into shipped artifacts.
    media: MediaRef | None = None
    staged_path: str | None = None

    def to_verifier_request(self) -> VerifierRequest:
        # Media candidates thread their staged clip through so deterministic
        # probes (e.g. adh.verifiers.media:check_media) can re-hash/re-probe it.
        extras: dict = {}
        if self.media is not None:
            extras["media"] = self.media.model_dump(mode="json")
            extras["media_abs_path"] = self.staged_path
        return VerifierRequest(
            input=self.input,
            output=self.output,
            reference=self.reference,
            tests=self.tests,
            extras=extras,
        )

    def to_dict(self) -> dict:
        """JSON form for the durable verified-candidate store (resume)."""
        return {
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "aspect_id": self.aspect_id,
            "verifiability_tier": int(self.verifiability_tier),
            "input": self.input,
            "output": self.output,
            "reference": self.reference,
            "tests": self.tests,
            "source_channel": self.source_channel.value,
            "source_url": self.source_url,
            "source_license": self.source_license,
            "source_sha256": self.source_sha256,
            "derived_from": list(self.derived_from),
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "system_prompt_hash": self.system_prompt_hash,
            "generation_params": dict(self.generation_params),
            "generated_at": self.generated_at.isoformat(),
            "pipeline_step": self.pipeline_step,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "language": self.language,
            "task_category": self.task_category,
            "cost_usd": self.cost_usd,
            "media": self.media.model_dump(mode="json") if self.media is not None else None,
            "staged_path": self.staged_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        # Tolerant defaults keep pre-media verified.jsonl resumes parseable.
        return cls(
            candidate_id=data["candidate_id"],
            job_id=data["job_id"],
            aspect_id=data["aspect_id"],
            verifiability_tier=VerifiabilityTier(data["verifiability_tier"]),
            input=data["input"],
            output=data["output"],
            reference=data.get("reference"),
            tests=data.get("tests"),
            source_channel=SourceChannel(data.get("source_channel", "synthesis")),
            source_url=data.get("source_url"),
            source_license=data.get("source_license", "unknown"),
            source_sha256=data.get("source_sha256"),
            derived_from=tuple(data.get("derived_from", [])),
            model_id=data.get("model_id", "heuristic:adh"),
            prompt_hash=data.get("prompt_hash", ""),
            system_prompt_hash=data.get("system_prompt_hash"),
            generation_params=dict(data.get("generation_params", {})),
            generated_at=datetime.fromisoformat(data["generated_at"]),
            pipeline_step=data.get("pipeline_step", "text-generation"),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            language=data.get("language", "en"),
            task_category=data.get("task_category"),
            cost_usd=float(data.get("cost_usd", 0.0)),
            media=MediaRef(**data["media"]) if data.get("media") else None,
            staged_path=data.get("staged_path"),
        )

    def base_record_fields(self) -> dict:
        """Generation-time record fields (quality/dedup/split are added downstream)."""
        return {
            "input": self.input,
            "output": self.output,
            "source_channel": self.source_channel,
            "source_url": self.source_url,
            "source_license": self.source_license,
            "source_sha256": self.source_sha256,
            "derived_from": list(self.derived_from),
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "system_prompt_hash": self.system_prompt_hash,
            "generation_params": dict(self.generation_params),
            "generated_at": self.generated_at,
            "pipeline_step": self.pipeline_step,
            "verifiability_tier": self.verifiability_tier,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "language": self.language,
            "task_category": self.task_category,
            "cost_usd": self.cost_usd,
            "media": self.media,
        }


def make_candidate_id(job_id: str, aspect_id: str, input_text: str, output_text: str) -> str:
    """Deterministic, content-addressed candidate id (same payload -> same id)."""
    digest = sha256_hex(f"{input_text}\x00{output_text}")[:12]
    return f"{job_id}:{aspect_id}:{digest}"
