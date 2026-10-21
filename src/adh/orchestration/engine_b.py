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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from adh.contract import (
    JobSpec,
    Record,
    RecordValidationError,
    check_job_spec,
    get_verifier_hook,
    is_media_modality,
    load_record_schema,
)
from adh.corpus import CorpusChunk, CorpusStore, LocalCorpusStore, SynthesisHandoff
from adh.curation import (
    Decontaminator,
    Deduper,
    assign_split,
    build_record,
    dead_letter_row,
    resolve_redactor,
)
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
from adh.delivery import (
    DeliveryResult,
    DatasetQualityReport,
    emit_docs,
    evaluate_dataset,
    write_dataset,
)
from adh.llm import UsageTracker
from adh.verification import JudgePanel, VerificationRouter, Verdict, build_judge

from .limits import CircuitBreaker, RunMetrics
from .run_store import load_plan
from .state import atomic_write_text

logger = logging.getLogger(__name__)

# Seeds planned per target record. Live runs keep ~4 in 10 seeds (junk grounding is
# skipped, the jury rejects some), so one seed per record falls far short. Generation
# stops once the target is verified, so unneeded seeds cost nothing.
SEED_OVERSAMPLE = 3


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
    quality_report: DatasetQualityReport | None = None  # offline quality card (§4.2)

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
    # Validate before constructing models, consuming tokens, or resetting any
    # prior build artifacts. A missing Tier-1 oracle would reject every candidate.
    check_job_spec(spec, base_dir=schema_base_dir).raise_if_errors()
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
    schema, hook = _load_schema(spec, schema_base_dir)

    # ---- Engine A outputs ----
    plan = load_plan(work)
    aspects: tuple[Aspect, ...] = plan.aspects if plan else ()
    chunks, handoffs = load_corpus(work, store)

    # ---- collaborators (offline heuristics unless a model is configured) ----
    generator = generator or build_generator(spec, usage=usage)
    judge = judge or build_judge(spec, usage=usage)
    router = VerificationRouter(spec=spec, verifier_hook=hook, judge=judge)
    novelty = NoveltyGate()

    # ---- seed the sampling space (diversity_strategy honored from the spec) ----
    target = _target_records(spec)
    seeds = (
        seed_engine
        or SeedEngine(
            seed=int(spec.generation_params.get("seed", 42)),
            diversity_strategy=spec.diversity_strategy,
        )
    ).plan(
        job_id,
        aspects=aspects,
        chunks=chunks,
        handoffs=handoffs,
        n_seeds=n_seeds or target * SEED_OVERSAMPLE,
        spec=spec,
    )

    # ---- durable state + prior verified candidates ----
    state = _load_bstate(work, job_id, resume)
    kept: list[tuple[Candidate, Verdict]] = _load_verified(verified_path) if resume else []
    progress = _TargetProgress(spec, aspects, target)
    for cand, _ in kept:
        novelty.is_novel(_novelty_key(cand))  # re-prime so resume keeps the diversity gate honest
        progress.add(cand)
    if kept:
        notes.append(f"resumed: {len(kept)} verified candidate(s) from prior run")
    if resume and not kept and state.completed_seed_ids and not state.counters.get("generated"):
        # A prior build marked seeds complete but generated nothing — the signature of a
        # transient provider outage swallowed inside the generator. Bookkeeping without
        # work must not strand the run: clear it so this build regenerates.
        stranded = len(state.completed_seed_ids)
        state.completed_seed_ids.clear()
        notes.append(
            f"prior build marked {stranded} seed(s) complete but generated nothing "
            "(likely a transient provider outage) — clearing seed bookkeeping to regenerate"
        )
        _save_bstate(work, state)

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
        progress=progress,
    )
    notes.extend(state.notes)
    exhausted = getattr(generator, "exhausted", 0)
    if exhausted:
        notes.append(
            f"generator exhausted {exhausted}/{len(seeds)} seed(s) across every configured "
            "tier — check provider health and ADH_GENERATOR_MODELS"
        )

    # ---- curate (PII policy + per-shard dedup + decontaminate) -> records ----
    records, accepted_aspects, dead, dup_count, pii_stats = _curate(
        kept, spec=spec, schema=schema, schema_base_dir=schema_base_dir, notes=notes
    )
    _write_dead_letter(dead_letter_path, dead)

    if exit_reason is None:
        exit_reason = "completed" if records else "no_records"

    metrics = _metrics_now(state.base_metrics, usage)

    # ---- deliver: shards + manifest + documentation-as-byproduct ----
    # media_staging holds the content-addressed clips Engine A downloaded; layouts
    # copy from there into the dataset. Absent for text jobs (no-op).
    delivery = write_dataset(
        records,
        spec=spec,
        out_dir=dataset_dir,
        exit_reason=exit_reason,
        staged_media_root=work / "media_staging",
    )
    notes.extend(delivery.notes)
    coverage = coverage_report(accepted_aspects, aspects, target)

    counts = _counts(
        state,
        records=records,
        duplicates=dup_count,
        dead=len(dead),
        coverage=coverage,
        pii=pii_stats,
        metrics=metrics,
    )
    docs: list[Path] = []
    if spec.emit_datasheet:
        docs = emit_docs(
            spec,
            stats={**counts, "counts_by_split": delivery.counts_by_split},
            files=_croissant_files(delivery, dataset_dir),
            out_dir=dataset_dir,
        )

    # ---- offline quality card (§4.2): markdown beside the shards + into the summary
    quality = evaluate_dataset(
        dataset_dir, schema=schema, references=_decontam_references(spec, schema_base_dir)
    )
    quality_md_path = dataset_dir / "quality_report.md"
    quality_md_path.write_text(quality.to_markdown(), encoding="utf-8")

    _write_summary(
        summary_path,
        spec,
        exit_reason,
        metrics,
        counts,
        delivery,
        coverage,
        notes,
        quality=_quality_summary(quality),
    )

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
        quality_report=quality,
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
    progress: _TargetProgress,
) -> str | None:
    start = clock_seconds()
    total = len(seeds)
    done_before = sum(1 for s in seeds if s.seed_id in state.completed_seed_ids)
    logger.info(
        "engine B: %d seed(s) to generate (%d already complete), stopping at %d distinct record(s)",
        total - done_before,
        done_before,
        progress.target,
    )
    for seed in seeds:
        if seed.seed_id in state.completed_seed_ids:
            continue
        if progress.reached():
            thin = progress.thin_aspects()
            if not thin:
                _note_once(
                    state,
                    f"target reached: {progress.distinct} distinct record(s) after "
                    f"{done_before}/{total} seed(s); the remaining seeds were not needed",
                )
                _save_bstate(work, state)
                return None
            if seed.aspect_id not in thin:
                continue  # left unmarked: a later build may still need it
            _note_once(
                state,
                f"target reached; generating only for thin aspect(s): {', '.join(thin)}",
            )
        metrics = _metrics_now(state.base_metrics, usage)
        trip = breaker.check(
            metrics,
            storage_mb=_storage_mb(store, seed.job_id),
            elapsed_seconds=clock_seconds() - start,
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
            _save_bstate(work, state)  # durable evidence even if every seed fails
            continue
        for cand in cands:
            state.counters["generated"] += 1
            if not novelty.is_novel(_novelty_key(cand)):
                state.counters["dropped_novelty"] += 1
                continue
            verdict = router.verify(cand)
            if verdict.dropped:
                state.counters[f"dropped_{verdict.stage}"] += 1
                logger.warning(
                    "candidate %s rejected at %s: %s",
                    cand.candidate_id,
                    verdict.stage,
                    verdict.reason,
                )
                # Keep a bounded sample of distinct reasons in the durable build
                # summary, without persisting rejected (potentially PII) content.
                note = f"verification rejection ({verdict.stage}): {verdict.reason[:500]}"
                if (
                    note not in state.notes
                    and sum(n.startswith("verification rejection (") for n in state.notes) < 10
                ):
                    state.notes.append(note)
                continue
            kept.append((cand, verdict))
            progress.add(cand)
            _append_verified(verified_path, cand, verdict)
        state.completed_seed_ids.add(seed.seed_id)
        _save_bstate(work, state)
        done_before += 1
        c = state.counters
        logger.info(
            "engine B progress: seed %d/%d (%s) — generated=%d kept=%d "
            "dropped_verifier=%d dropped_jury=%d errors=%d",
            done_before,
            total,
            seed.aspect_id,
            c.get("generated", 0),
            len(kept),
            c.get("dropped_verifier", 0),
            c.get("dropped_jury", 0),
            c.get("generation_errors", 0),
        )
    return None


# --------------------------------------------------------------------- curation stage
def _curate(kept, *, spec, schema, schema_base_dir, redactor=None, notes=None):
    """PII policy (§2.4) -> per-shard near-dedup + decontaminate (§2.4) -> format.

    Returns ``(records, accepted_aspects, dead_letter_rows, duplicate_count,
    pii_stats)``; ``pii_stats`` is ``{"pii_dropped": int, "pii_redacted": int}``.
    Dedup clusters live *inside* their split (identical text landing in two
    shards yields two representatives) so eval splits keep their own copies of
    recurring content. ``notes`` collects loud-degradation messages (e.g. an
    unenforceable PII default) for the run summary.
    """
    notes = notes if notes is not None else []
    redactor = resolve_redactor(spec, injected=redactor, notes=notes)
    pii_stats = {"pii_dropped": 0, "pii_redacted": 0}

    survivors: list[tuple[Candidate, Verdict]] = []
    dead: list[dict] = []
    for cand, verdict in kept:
        if redactor is None:
            survivors.append((cand, verdict))
            continue
        input_entities = redactor.analyze(cand.input)
        output_entities = redactor.analyze(cand.output)
        if not input_entities and not output_entities:
            survivors.append((cand, verdict))
            continue
        entity_types = sorted({e.entity_type for e in [*input_entities, *output_entities]})
        if spec.pii_policy == "drop":
            error = RecordValidationError(
                f"pii_policy='drop': candidate bears {', '.join(entity_types)}", record=None
            )
            dead.append(dead_letter_row(cand, error))
            pii_stats["pii_dropped"] += 1
            continue
        cand = replace(
            cand,
            input=redactor.redact(cand.input, input_entities),
            output=redactor.redact(cand.output, output_entities),
            generation_params={
                **cand.generation_params,
                "pii_redaction": {
                    "tool": redactor.name,
                    "policy": "redact",
                    "entity_types": entity_types,
                },
            },
        )
        pii_stats["pii_redacted"] += 1
        survivors.append((cand, verdict))

    decon = Decontaminator.from_paths(spec.decontaminate_against, base_dir=schema_base_dir)
    media_modality = is_media_modality(spec.modality)
    deduper = _job_deduper(spec)
    contaminated_flags = [
        decon.is_contaminated(c.output) or decon.is_contaminated(c.input) for c, _ in survivors
    ]
    # the split each candidate will land in — computed with the same deterministic
    # assign_split build_record uses, so clustering and record stamping agree
    split_of = [
        assign_split(c.candidate_id, spec.splits, contaminated=flag)
        for (c, _), flag in zip(survivors, contaminated_flags)
    ]
    assigns = deduper.cluster_per_split(
        list(zip(split_of, [_dedup_text(c, media_modality) for c, _ in survivors]))
    )

    records: list[Record] = []
    accepted_aspects: list[str] = []
    duplicates = 0
    for (cand, verdict), flag, assign in zip(survivors, contaminated_flags, assigns):
        if assign.is_duplicate:
            duplicates += 1
            continue  # deduplicated dataset: keep the shard's representative, drop near-dups
        try:
            record = build_record(
                cand, spec=spec, schema=schema, verdict=verdict, dedup=assign, contaminated=flag
            )
        except RecordValidationError as exc:
            dead.append(dead_letter_row(cand, exc))
            continue
        records.append(record)
        accepted_aspects.append(cand.aspect_id)
    return records, accepted_aspects, dead, duplicates, pii_stats


# --------------------------------------------------------------------- loaders
def load_corpus(work: Path, store: CorpusStore) -> tuple[list[CorpusChunk], list[SynthesisHandoff]]:
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
                try:
                    preview = store.get(uri).strip()[:280]
                except (OSError, UnicodeDecodeError, ValueError):
                    preview = ""  # binary/media URIs carry no text preview
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


def _load_schema(spec, schema_base_dir) -> tuple[dict, dict | None]:
    schema = load_record_schema(spec.record_schema_ref, base_dir=schema_base_dir)
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
def _novelty_key(cand: Candidate) -> str:
    """Video candidates collide on asset identity (media sha); text ones on payload."""
    return cand.media.sha256 if cand.media is not None else cand.input


def _note_once(state, note: str) -> None:
    if note not in state.notes:
        state.notes.append(note)


def _job_deduper(spec: JobSpec) -> Deduper:
    """The near-dedup curation applies; generation counts progress with the same one."""
    if is_media_modality(spec.modality):
        # MinHash degenerates on short captions, and identical captions over
        # different assets ARE training signal — cluster on asset identity instead.
        return Deduper(method="exact")
    return Deduper(
        method=spec.dedup_policy.method,
        ngram=spec.dedup_policy.ngram,
        jaccard=spec.dedup_policy.jaccard,
    )


class _TargetProgress:
    """Distinct verified records per aspect, measured against the build target.

    Counts what curation will ship (one per near-duplicate cluster), not raw verified
    candidates, so a build never stops short on duplicates. Clustering here is across
    splits, which is at least as strict as curation's per-split clustering.
    """

    def __init__(self, spec: JobSpec, aspects: tuple[Aspect, ...], target: int) -> None:
        self.target = target
        self._aspects = aspects
        self._media = is_media_modality(spec.modality)
        self._index = _job_deduper(spec).index()
        self._aspect_ids: list[str] = []

    def add(self, cand: Candidate) -> None:
        if not self._index.add(_dedup_text(cand, self._media)).is_duplicate:
            self._aspect_ids.append(cand.aspect_id)

    @property
    def distinct(self) -> int:
        return len(self._aspect_ids)

    def reached(self) -> bool:
        return bool(self.target) and self.distinct >= self.target

    def thin_aspects(self) -> tuple[str, ...]:
        return coverage_report(self._aspect_ids, self._aspects, self.target).thin_aspects


def _dedup_text(cand: Candidate, media_modality: bool) -> str:
    """The string dedup clusters on: media sha for media jobs, output text otherwise."""
    if media_modality and cand.media is not None:
        return cand.media.sha256
    return cand.output


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


def _counts(state, *, records, duplicates, dead, coverage, pii, metrics) -> dict:
    generated = state.counters.get("generated", 0)
    accepted = len(records)
    return {
        "generated": generated,
        "generation_errors": state.counters.get("generation_errors", 0),
        "accepted": accepted,
        "accept_rate": (accepted / generated) if generated else 0.0,
        "dropped_novelty": state.counters.get("dropped_novelty", 0),
        "dropped_quality": state.counters.get("dropped_quality", 0),
        "dropped_verifier": state.counters.get("dropped_verifier", 0),
        "dropped_jury": state.counters.get("dropped_jury", 0),
        "duplicates": duplicates,
        "dead_lettered": dead,
        "pii_dropped": pii["pii_dropped"],
        "pii_redacted": pii["pii_redacted"],
        "contaminated": sum(1 for r in records if r.contamination_flag),
        "total_records": accepted,
        # OUTLINE §3.4 efficiency per accepted record; cost stays None unless a
        # provider surfaced spend (decision 4 — no self-maintained price table).
        "tokens_per_accepted_record": (metrics.llm_tokens / accepted) if accepted else None,
        "cost_per_accepted_record": (
            round(metrics.cost_usd / accepted, 6)
            if accepted and metrics.cost_usd is not None
            else None
        ),
        "thin_aspects": list(coverage.thin_aspects),
    }


def _decontam_references(spec: JobSpec, schema_base_dir: Path | None) -> list[Path] | None:
    """Resolve ``decontaminate_against`` against the spec base dir (same rule as curation)."""
    out: list[Path] = []
    for ref in spec.decontaminate_against or []:
        p = Path(ref)
        if not p.is_absolute() and schema_base_dir is not None:
            p = Path(schema_base_dir) / p
        out.append(p)
    return out or None


def _croissant_files(delivery: DeliveryResult, dataset_dir: Path) -> list[str]:
    """Distribution ids for croissant: dataset-dir-relative POSIX paths.

    Layout assets (videos/, media/) keep their subdirectory instead of being
    flattened to a basename; anything outside the dataset dir falls back to its
    bare name.
    """
    names = []
    for f in delivery.files:
        if f.is_relative_to(dataset_dir):
            names.append(f.relative_to(dataset_dir).as_posix())
        else:
            names.append(f.name)
    return sorted(set(names))


def _quality_summary(q: DatasetQualityReport) -> dict:
    """The JSON-serializable slice of the quality report for the run summary."""
    return {
        "total_records": q.total_records,
        "counts_by_split": q.counts_by_split,
        "duplicate_rate": round(q.duplicate_rate, 4),
        "contamination_residual_count": q.contamination_residual_count,
        "license_coverage": round(q.license_coverage, 4),
        "schema_validation_pass_rate": round(q.schema_validation_pass_rate, 4),
        "pii_residual_count": q.pii_residual_count,
        "n_media_assets": q.n_media_assets,
        "total_media_seconds": round(q.total_media_seconds, 3),
        "warnings": list(q.warnings),
    }


def _write_summary(
    path, spec, exit_reason, metrics, counts, delivery, coverage, notes, *, quality=None
) -> None:
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
    if quality is not None:
        summary["quality"] = quality
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
