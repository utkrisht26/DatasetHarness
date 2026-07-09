"""Engine B driver — corpus + hand-offs -> verified, deduplicated dataset (OUTLINE §2).

Mirrors the Engine A driver's discipline (resumable, metered, limit-bounded) for the
synthesis + curation half:

    load Engine A corpus/hand-offs (§1) -> seed/taxonomy (§2.1) -> generate (§2.2)
      -> novelty gate + tiered verify (§2.3) -> near-dedup + decontaminate (§2.4)
      -> format/validate + dead-letter (§2.5) -> shards + datasheet/croissant (§4.1)

Every model call is metered into a :class:`UsageTracker`; the circuit breaker gates
generation on ``max_llm_tokens`` / ``max_cost_usd`` / ``max_runtime_seconds``; and
verified candidates are journaled to ``engine_b_verified.jsonl`` so a killed run
resumes without regenerating. A coverage report against the taxonomy flags thin
aspects — the signal Engine A uses to re-discover (the coverage-gap loop).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from adh.contract import (
    JobSpec,
    Record,
    RecordValidationError,
    get_verifier_hook,
    load_record_schema,
)
from adh.corpus import CorpusChunk, CorpusStore, LocalCorpusStore, SynthesisHandoff
from adh.curation import Decontaminator, Deduper, build_record, dead_letter_row
from adh.discovery.models import Aspect
from adh.generation import (
    Candidate,
    CoverageReport,
    NoveltyGate,
    SeedEngine,
    build_generator,
    coverage_report,
)
from adh.generation.generator import Generator
from adh.delivery import DeliveryResult, emit_docs, write_dataset
from adh.llm import UsageTracker
from adh.verification import JudgePanel, VerificationRouter, Verdict, build_judge

from .limits import CircuitBreaker, RunMetrics
from .run_store import load_plan
from .state import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass
class EngineBResult:
    job_id: str
    records: list[Record]
    metrics: RunMetrics
    coverage: CoverageReport
    delivery: DeliveryResult
    docs: list[Path]
    exit_reason: str
    work_dir: Path
    dataset_dir: Path
    dead_letter_path: Path
    summary_path: Path
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return len(self.records)


def run_engine_b(
    spec: JobSpec,
    *,
    work_dir: str | Path,
    generator: Generator | None = None,
    judge: JudgePanel | None = None,
    store: CorpusStore | None = None,
    usage: UsageTracker | None = None,
    seed_engine: SeedEngine | None = None,
    n_seeds: int | None = None,
    resume: bool = True,
    schema_base_dir: Path | None = None,
    clock_seconds: Callable[[], float] = time.monotonic,
    wallclock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> EngineBResult:
    """Build a verified dataset from an Engine A run directory."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    job_id = spec.job_id or spec.name
    dataset_dir = work / "dataset"
    store = store or LocalCorpusStore(work / "corpus")
    usage = usage or UsageTracker()
    breaker = CircuitBreaker(spec.limits)
    notes: list[str] = []

    verified_path = work / "engine_b_verified.jsonl"
    dead_letter_path = work / "engine_b_dead_letter.jsonl"
    summary_path = work / "engine_b_summary.json"
    if not resume:
        for p in (verified_path, dead_letter_path, _bstate_path(work)):
            p.unlink(missing_ok=True)

    # ---- record schema + Tier-1 verifier hook ----
    schema, hook = _load_schema(spec, schema_base_dir, notes)

    # ---- Engine A outputs ----
    plan = load_plan(work)
    aspects: tuple[Aspect, ...] = plan.aspects if plan else ()
    chunks, handoffs = load_corpus(work, store)

    # ---- collaborators (offline heuristics unless a model is configured) ----
    generator = generator or build_generator(spec, usage=usage)
    judge = judge or build_judge(spec, usage=usage)
    router = VerificationRouter(spec=spec, verifier_hook=hook, judge=judge)
    novelty = NoveltyGate()

    # ---- seed the sampling space ----
    target = _target_records(spec)
    seeds = (seed_engine or SeedEngine(seed=int(spec.generation_params.get("seed", 42)))).plan(
        job_id, aspects=aspects, chunks=chunks, handoffs=handoffs, n_seeds=n_seeds or target
    )

    # ---- durable state + prior verified candidates ----
    state = _load_bstate(work, job_id, resume)
    kept: list[tuple[Candidate, Verdict]] = _load_verified(verified_path) if resume else []
    for cand, _ in kept:
        novelty.is_novel(cand.input)  # re-prime so resume keeps the diversity gate honest
    if kept:
        notes.append(f"resumed: {len(kept)} verified candidate(s) from prior run")

    # ---- generate -> verify (metered, limit-bounded, resumable) ----
    exit_reason = _generate_and_verify(
        seeds,
        spec=spec,
        generator=generator,
        router=router,
        novelty=novelty,
        store=store,
        state=state,
        kept=kept,
        verified_path=verified_path,
        usage=usage,
        breaker=breaker,
        work=work,
        clock_seconds=clock_seconds,
    )
    notes.extend(state.notes)

    # ---- curate (dedup + decontaminate) + format/validate -> records ----
    records, accepted_aspects, dead, dup_count = _curate(
        kept, spec=spec, schema=schema, schema_base_dir=schema_base_dir
    )
    _write_dead_letter(dead_letter_path, dead)

    if exit_reason is None:
        exit_reason = "completed" if records else "no_records"

    # ---- deliver: shards + manifest + documentation-as-byproduct ----
    delivery = write_dataset(records, spec=spec, out_dir=dataset_dir, exit_reason=exit_reason)
    notes.extend(delivery.notes)
    coverage = coverage_report(accepted_aspects, aspects, target)

    counts = _counts(state, records=records, duplicates=dup_count, dead=len(dead), coverage=coverage)
    docs: list[Path] = []
    if spec.emit_datasheet:
        docs = emit_docs(
            spec,
            stats={**counts, "counts_by_split": delivery.counts_by_split},
            files=[f.name for f in delivery.files],
            out_dir=dataset_dir,
        )

    metrics = _metrics_now(state.base_metrics, usage)
    _write_summary(summary_path, spec, exit_reason, metrics, counts, delivery, coverage, notes)

    return EngineBResult(
        job_id=job_id,
        records=records,
        metrics=metrics,
        coverage=coverage,
        delivery=delivery,
        docs=docs,
        exit_reason=exit_reason,
        work_dir=work,
        dataset_dir=dataset_dir,
        dead_letter_path=dead_letter_path,
        summary_path=summary_path,
        counts=counts,
        notes=notes,
    )


# --------------------------------------------------------------------- generation loop
def _generate_and_verify(
    seeds,
    *,
    spec,
    generator,
    router,
    novelty,
    store,
    state,
    kept,
    verified_path,
    usage,
    breaker,
    work,
    clock_seconds,
) -> str | None:
    start = clock_seconds()
    for seed in seeds:
        if seed.seed_id in state.completed_seed_ids:
            continue
        metrics = _metrics_now(state.base_metrics, usage)
        trip = breaker.check(
            metrics, storage_mb=_storage_mb(store, seed.job_id), elapsed_seconds=clock_seconds() - start
        )
        if trip is not None:
            state.notes.append(f"circuit breaker: {trip}")
            return trip.reason

        try:
            cands = generator.generate(seed, spec=spec, store=store)
        except Exception as exc:  # noqa: BLE001 - a failed seed must not kill the run
            logger.warning("generation failed for seed %s: %s", seed.seed_id, exc)
            state.counters["generation_errors"] += 1
            state.completed_seed_ids.add(seed.seed_id)
            continue
        for cand in cands:
            state.counters["generated"] += 1
            if not novelty.is_novel(cand.input):
                state.counters["dropped_novelty"] += 1
                continue
            verdict = router.verify(cand)
            if verdict.dropped:
                state.counters[f"dropped_{verdict.stage}"] += 1
                continue
            kept.append((cand, verdict))
            _append_verified(verified_path, cand, verdict)
        state.completed_seed_ids.add(seed.seed_id)
        _save_bstate(work, state)
    return None


# --------------------------------------------------------------------- curation stage
def _curate(kept, *, spec, schema, schema_base_dir):
    texts = [c.output for c, _ in kept]
    assigns = Deduper(
        method=spec.dedup_policy.method,
        ngram=spec.dedup_policy.ngram,
        jaccard=spec.dedup_policy.jaccard,
    ).cluster(texts)
    decon = Decontaminator.from_paths(spec.decontaminate_against, base_dir=schema_base_dir)

    records: list[Record] = []
    accepted_aspects: list[str] = []
    dead: list[dict] = []
    duplicates = 0
    for (cand, verdict), assign in zip(kept, assigns):
        if assign.is_duplicate:
            duplicates += 1
            continue  # deduplicated dataset: keep the representative, drop near-dups
        contaminated = decon.is_contaminated(cand.output) or decon.is_contaminated(cand.input)
        try:
            record = build_record(
                cand, spec=spec, schema=schema, verdict=verdict, dedup=assign, contaminated=contaminated
            )
        except RecordValidationError as exc:
            dead.append(dead_letter_row(cand, exc))
            continue
        records.append(record)
        accepted_aspects.append(cand.aspect_id)
    return records, accepted_aspects, dead, duplicates


# --------------------------------------------------------------------- loaders
def load_corpus(
    work: Path, store: CorpusStore
) -> tuple[list[CorpusChunk], list[SynthesisHandoff]]:
    """Reconstruct Engine A's corpus chunks + synthesis hand-offs from its manifest."""
    manifest = Path(work) / "manifest.jsonl"
    chunks: list[CorpusChunk] = []
    handoffs: list[SynthesisHandoff] = []
    if not manifest.exists():
        return chunks, handoffs
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        kind = row.get("kind")
        if kind == "chunk":
            preview = ""
            uri = row.get("content_uri", "")
            if uri and store.exists(uri):
                preview = store.get(uri).strip()[:280]
            chunks.append(CorpusChunk.from_manifest_row(row, text_preview=preview))
        elif kind == "handoff":
            handoffs.append(
                SynthesisHandoff(
                    job_id=row["job_id"],
                    aspect_id=row["aspect_id"],
                    topic=row.get("topic", ""),
                    keywords=tuple(row.get("keywords", [])),
                )
            )
    return chunks, handoffs


def _load_schema(spec, schema_base_dir, notes) -> tuple[dict | None, dict | None]:
    try:
        schema = load_record_schema(spec.record_schema_ref, base_dir=schema_base_dir)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        notes.append(f"record schema unreadable ({exc}); skipping per-job JSON-Schema validation")
        return None, None
    return schema, get_verifier_hook(schema)


# --------------------------------------------------------------------- durable state
@dataclass
class _BState:
    job_id: str
    completed_seed_ids: set[str] = field(default_factory=set)
    base_metrics: RunMetrics = field(default_factory=RunMetrics)  # spend carried across resumes
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "completed_seed_ids": sorted(self.completed_seed_ids),
            "metrics": self.base_metrics.to_dict(),
            "counters": dict(self.counters),
            "notes": self.notes,
        }


def _bstate_path(work: Path) -> Path:
    return Path(work) / "engine_b_state.json"


def _load_bstate(work: Path, job_id: str, resume: bool) -> _BState:
    p = _bstate_path(work)
    if resume and p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("job_id") == job_id:
                st = _BState(
                    job_id=job_id,
                    completed_seed_ids=set(data.get("completed_seed_ids", [])),
                    base_metrics=RunMetrics.from_dict(data.get("metrics", {})),
                    notes=list(data.get("notes", [])),
                )
                st.counters = defaultdict(int, data.get("counters", {}))
                return st
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return _BState(job_id=job_id)


def _save_bstate(work: Path, state: _BState) -> None:
    atomic_write_text(_bstate_path(work), json.dumps(state.to_dict(), indent=2))


def _append_verified(path: Path, cand: Candidate, verdict: Verdict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"candidate": cand.to_dict(), "verdict": verdict.to_dict()}) + "\n")


def _load_verified(path: Path) -> list[tuple[Candidate, Verdict]]:
    out: list[tuple[Candidate, Verdict]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append((Candidate.from_dict(row["candidate"]), Verdict.from_dict(row["verdict"])))
    return out


def _write_dead_letter(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------- small helpers
def _target_records(spec: JobSpec) -> int:
    if spec.target_size.records:
        return spec.target_size.records
    if spec.target_size.tokens:
        return max(1, spec.target_size.tokens // 512)
    return 0


def _metrics_now(base: RunMetrics, usage: UsageTracker) -> RunMetrics:
    total = usage.total
    return base.add_tokens(total.total_tokens).add_cost(total.cost_usd)


def _storage_mb(store: CorpusStore, job_id: str) -> float:
    size = getattr(store, "size_mb", None)
    return size(job_id) if callable(size) else 0.0


def _counts(state, *, records, duplicates, dead, coverage) -> dict:
    generated = state.counters.get("generated", 0)
    accepted = len(records)
    return {
        "generated": generated,
        "accepted": accepted,
        "accept_rate": (accepted / generated) if generated else 0.0,
        "dropped_novelty": state.counters.get("dropped_novelty", 0),
        "dropped_quality": state.counters.get("dropped_quality", 0),
        "dropped_verifier": state.counters.get("dropped_verifier", 0),
        "dropped_jury": state.counters.get("dropped_jury", 0),
        "duplicates": duplicates,
        "dead_lettered": dead,
        "contaminated": sum(1 for r in records if r.contamination_flag),
        "total_records": accepted,
        "thin_aspects": list(coverage.thin_aspects),
    }


def _write_summary(path, spec, exit_reason, metrics, counts, delivery, coverage, notes) -> None:
    summary = {
        "job_id": spec.job_id or spec.name,
        "engine": "B",
        "exit_reason": exit_reason,
        "metrics": metrics.to_dict(),
        "counts": counts,
        "output_format": delivery.format,
        "counts_by_split": delivery.counts_by_split,
        "coverage": coverage.to_dict(),
        "notes": notes,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
