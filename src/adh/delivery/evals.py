"""Dataset-quality evaluation — the pre-publish gate (OUTLINE §4.2).

Builds an offline report card for a written dataset directory (shards + manifest as
produced by :func:`adh.delivery.write_dataset`): record counts and split balance,
payload length distributions, duplicate rate, contamination residuals, license
coverage, schema-validation pass rate, and PII residuals via an injected
:class:`PIIRedactor` port. Everything is computed from the emitted records — no
network, no model calls.

:func:`assert_publishable` turns a report into a hard gate before
:func:`adh.delivery.publish.publish_dataset`.

The downstream-lift probe (:class:`LiftProbe`, :class:`EmbeddingLiftProbe`) is a
**proxy**: it measures cosine separation between aspect clusters vs random pairs in
embedding space. A true fine-tune-lift probe — train on one aspect cluster, evaluate
against a matched control, measure the eval delta (OUTLINE §4.2: "the only evaluation
that ultimately matters") — is future work; this is the cheap geometric stand-in.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from adh.contract import Record
from adh.curation.decontam import Decontaminator
from adh.delivery.writer import decode_parquet_row

logger = logging.getLogger(__name__)

_UNKNOWN_LICENSES = {"", "unknown", "none", "null"}
_MAX_FAILURE_SAMPLES = 5


# --- ports --------------------------------------------------------------------


@runtime_checkable
class PIIRedactor(Protocol):
    """Port for PII scanning/redaction (injected; the harness ships none by default).

    ``redact`` returns the text with any PII removed/masked. A record counts as a
    *PII residual* when redaction would change its payload at all.
    """

    def redact(self, text: str) -> str: ...


# --- report model -------------------------------------------------------------


@dataclass(frozen=True)
class LengthStats:
    """Character-length distribution of one payload field across all records."""

    n: int
    minimum: int
    mean: float
    p50: float
    p95: float
    maximum: int


@dataclass(frozen=True)
class DatasetQualityReport:
    """Offline quality card for a dataset directory (renderable to markdown)."""

    dataset_dir: Path
    total_records: int
    counts_by_split: dict[str, int]
    split_fractions: dict[str, float]
    input_lengths: LengthStats
    output_lengths: LengthStats
    duplicate_records: int
    duplicate_rate: float
    contamination_residual_count: int
    license_coverage: float
    unknown_license_records: int
    schema_validation_pass_rate: float
    schema_failures: tuple[str, ...]
    pii_residual_count: int
    n_media_assets: int = 0
    total_media_seconds: float = 0.0
    #: assets whose provenance rows declare no usable license — the publish
    #: gate refuses PUBLIC uploads while this is positive (two-layer policy).
    unknown_license_assets: int = 0
    warnings: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        """Render the report as a markdown metric table."""
        rows: list[tuple[str, str]] = [
            ("total_records", str(self.total_records)),
            (
                "split_balance",
                "; ".join(
                    f"{s}: {self.counts_by_split[s]} ({self.split_fractions[s]:.1%})"
                    for s in sorted(self.counts_by_split)
                )
                or "n/a",
            ),
            (
                "input_chars",
                f"mean {self.input_lengths.mean:.1f} · p50 {self.input_lengths.p50:.0f} · "
                f"p95 {self.input_lengths.p95:.0f} · max {self.input_lengths.maximum}",
            ),
            (
                "output_chars",
                f"mean {self.output_lengths.mean:.1f} · p50 {self.output_lengths.p50:.0f} · "
                f"p95 {self.output_lengths.p95:.0f} · max {self.output_lengths.maximum}",
            ),
            ("duplicate_records", str(self.duplicate_records)),
            ("duplicate_rate", f"{self.duplicate_rate:.1%}"),
            ("contamination_residuals", str(self.contamination_residual_count)),
            ("license_coverage", f"{self.license_coverage:.1%}"),
            ("unknown_license_records", str(self.unknown_license_records)),
            ("schema_validation_pass_rate", f"{self.schema_validation_pass_rate:.1%}"),
            ("pii_residual_count", str(self.pii_residual_count)),
        ]
        if self.n_media_assets:
            rows += [
                ("n_media_assets", str(self.n_media_assets)),
                ("total_media_seconds", str(self.total_media_seconds)),
            ]
            if self.unknown_license_assets:
                rows.append(("unknown_license_assets", str(self.unknown_license_assets)))
        lines = [
            f"# Dataset quality report — `{self.dataset_dir}`",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *(f"| {k} | {v} |" for k, v in rows),
        ]
        if self.warnings:
            lines += ["", "## Warnings", *(f"- {w}" for w in self.warnings)]
        return "\n".join(lines) + "\n"


class NotPublishableError(ValueError):
    """Raised by :func:`assert_publishable`; ``reasons`` lists every failing check."""

    def __init__(self, reasons: list[str]):
        super().__init__("dataset is not publishable: " + "; ".join(reasons))
        self.reasons = reasons


def assert_publishable(
    report: DatasetQualityReport, *, min_records: int, max_contamination: int = 0
) -> None:
    """Gate publishing: raise :class:`NotPublishableError` listing every reason not to."""
    reasons: list[str] = []
    if report.total_records < min_records:
        reasons.append(
            f"records {report.total_records} < required minimum {min_records}"
        )
    if report.contamination_residual_count > max_contamination:
        reasons.append(
            f"contamination residuals {report.contamination_residual_count} > allowed "
            f"{max_contamination}"
        )
    if report.schema_validation_pass_rate < 1.0:
        reasons.append(
            f"schema validation pass rate {report.schema_validation_pass_rate:.1%} < 100% "
            f"({len(report.schema_failures)}+ failures)"
        )
    if reasons:
        raise NotPublishableError(reasons)


# --- shard reading ------------------------------------------------------------


def _iter_shard_rows(dataset_dir: Path) -> tuple[list[tuple[str, dict]], tuple[str, ...]]:
    """Yield ``(split, row_dict)`` pairs plus warnings; manifest-driven with glob fallback."""
    warnings: list[str] = []
    manifest = dataset_dir / "manifest.jsonl"
    shards: list[tuple[Path, str]] = []
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "shard":
                split = str(row.get("split") or Path(row["path"]).stem)
                shards.append((dataset_dir / str(row["path"]), split))
    else:
        warnings.append("manifest.jsonl missing; inferred shards from directory listing")
        shards = sorted(
            (p, p.stem)
            for p in dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jsonl", ".parquet"}
        )

    rows: list[tuple[str, dict]] = []
    parquet_skipped = 0
    for path, split in shards:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append((split, json.loads(line)))
        elif suffix == ".parquet" and _parquet_available():
            rows.extend((split, r) for r in _read_parquet_rows(path))
        elif suffix == ".parquet":
            parquet_skipped += 1
    if parquet_skipped:
        warnings.append(
            f"{parquet_skipped} Parquet shard(s) skipped from analysis (install 'adh[deliver]' "
            "for pyarrow)"
        )
    return rows, tuple(warnings)


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _read_parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    return [decode_parquet_row(row) for row in table.to_pylist()]


# --- metrics ------------------------------------------------------------------


def evaluate_dataset(
    dataset_dir: str | Path,
    *,
    schema: dict | None = None,
    references: Sequence[str | Path] | None = None,
    pii_redactor: PIIRedactor | None = None,
) -> DatasetQualityReport:
    """Compute the offline quality report for a written dataset directory.

    ``schema`` validates each row against a per-job JSON Schema when given, else
    against the base :class:`~adh.contract.Record` model. ``references`` adds an
    n-gram decontamination scan (via :class:`~adh.curation.decontam.Decontaminator`)
    on top of the per-record ``contamination_flag``. ``pii_redactor``, when injected,
    flags any record whose text it would change.
    """
    d = Path(dataset_dir)
    rows, warnings = _iter_shard_rows(d)
    n_assets, total_seconds = _media_stats(d)
    unknown_license_assets = count_unknown_license_assets(d)

    inputs: list[int] = []
    outputs: list[int] = []
    payloads: set[str] = set()
    duplicates = 0
    contaminated = 0
    unknown_license = 0
    schema_failures: list[str] = []
    schema_checks = 0
    pii_hits = 0

    validator = None
    if schema is not None:
        import jsonschema

        validator = jsonschema.Draft202012Validator(schema)

    decon = None
    if references:
        decon = Decontaminator.from_paths([str(r) for r in references])

    schema_failure_count = 0
    for _, row in rows:
        text_in = str(row.get("input") or "")
        text_out = str(row.get("output") or "")
        inputs.append(len(text_in))
        outputs.append(len(text_out))

        key = json.dumps({"i": text_in, "o": text_out}, ensure_ascii=False, sort_keys=True)
        if key in payloads:
            duplicates += 1
        else:
            payloads.add(key)

        is_flagged = bool(row.get("contamination_flag"))
        if not is_flagged and decon is not None and decon.is_contaminated(f"{text_in} {text_out}"):
            is_flagged = True
        contaminated += int(is_flagged)

        lic = str(row.get("source_license") or "").strip().lower()
        if lic in _UNKNOWN_LICENSES:
            unknown_license += 1

        schema_checks += 1
        failure = _schema_failure(row, validator)
        if failure is not None:
            schema_failure_count += 1
            if len(schema_failures) < _MAX_FAILURE_SAMPLES:
                schema_failures.append(failure)

        payload_text = f"{text_in} {text_out}"
        if pii_redactor is not None and pii_redactor.redact(payload_text) != payload_text:
            pii_hits += 1

    total = len(rows)
    counts_by_split: dict[str, int] = {}
    for split, _ in rows:
        counts_by_split[split] = counts_by_split.get(split, 0) + 1
    fractions = {s: (n / total if total else 0.0) for s, n in counts_by_split.items()}

    return DatasetQualityReport(
        dataset_dir=d,
        total_records=total,
        counts_by_split=counts_by_split,
        split_fractions=fractions,
        input_lengths=_length_stats(inputs),
        output_lengths=_length_stats(outputs),
        duplicate_records=duplicates,
        duplicate_rate=(duplicates / total) if total else 0.0,
        contamination_residual_count=contaminated,
        license_coverage=1.0 - (unknown_license / total) if total else 0.0,
        unknown_license_records=unknown_license,
        schema_validation_pass_rate=(
            1.0 - (schema_failure_count / schema_checks) if schema_checks else 1.0
        ),
        schema_failures=tuple(schema_failures),
        pii_residual_count=pii_hits,
        n_media_assets=n_assets,
        total_media_seconds=total_seconds,
        unknown_license_assets=unknown_license_assets,
        warnings=warnings,
    )


def _media_stats(dataset_dir: Path) -> tuple[int, float]:
    """Asset count + total seconds from manifest ``kind:"asset"`` rows (tolerant)."""
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        return 0, 0.0
    count = 0
    seconds = 0.0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") != "asset":
            continue
        count += 1
        try:
            seconds += float(row.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            pass
    return count, seconds


def count_unknown_license_assets(dataset_dir: Path) -> int:
    """Assets whose provenance declares no usable license (publish-gate input).

    Reads the same ``kind:"asset"`` manifest rows the delivery layouts write —
    the license/attribution columns travel with the asset, so distribution
    policy can be checked at publish time without re-opening the shards.
    """
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        return 0
    unknown = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") != "asset":
            continue
        lic = str(row.get("license") or "").strip().lower()
        if lic in _UNKNOWN_LICENSES:
            unknown += 1
    return unknown


def _schema_failure(row: dict, validator: Any | None) -> str | None:
    """Return the first validation error message for ``row``, or None when valid.

    Without an explicit schema the base :class:`Record` Pydantic model stands in.
    """
    if validator is None:
        try:
            Record.model_validate(row)
        except Exception as exc:  # pydantic.ValidationError — keep this module import-light
            return f"record {row.get('record_id', '?')}: {type(exc).__name__}"
        return None
    errors = sorted(validator.iter_errors(row), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    first = errors[0]
    where = "/".join(str(p) for p in first.absolute_path) or "<root>"
    return f"record {row.get('record_id', '?')} at {where}: {first.message}"


def _length_stats(values: list[int]) -> LengthStats:
    if not values:
        return LengthStats(n=0, minimum=0, mean=0.0, p50=0.0, p95=0.0, maximum=0)
    ordered = sorted(values)

    def pct(q: float) -> float:
        idx = (len(ordered) - 1) * q
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return LengthStats(
        n=len(ordered),
        minimum=ordered[0],
        mean=sum(ordered) / len(ordered),
        p50=pct(0.5),
        p95=pct(0.95),
        maximum=ordered[-1],
    )


# --- downstream-lift probe (geometric proxy; see module docstring) -------------


class LiftProbe(Protocol):
    """Port for downstream-lift probes: higher = aspects separate more."""

    def measure(self, groups: Mapping[str, Sequence[str]], *, seed: int = 0) -> float:
        """Return a lift score over texts grouped by aspect."""
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; zero vectors score 0 rather than raising."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_separation(
    embeddings_by_group: Mapping[str, Sequence[Sequence[float]]],
    *,
    pairs_per_group: int = 50,
    seed: int = 0,
) -> float:
    """Mean within-aspect cosine minus mean random cross-aspect cosine.

    The pure-math core of :class:`EmbeddingLiftProbe`: feed embeddings keyed by
    aspect cluster. Needs >=2 groups with >=2 vectors each; anything degenerate
    returns 0.0 (no contrast measurable). Sampling is seeded, so scores are
    reproducible.
    """
    usable = {g: list(vecs) for g, vecs in embeddings_by_group.items() if len(vecs) >= 2}
    if len(usable) < 2:
        return 0.0

    rng = random.Random(seed)
    same_group: list[float] = []
    for vecs in usable.values():
        pairs = [(i, j) for i, j in combinations(range(len(vecs)), 2)]
        if len(pairs) > pairs_per_group:
            pairs = rng.sample(pairs, pairs_per_group)
        same_group.extend(cosine(vecs[i], vecs[j]) for i, j in pairs)

    labels = [g for g, vecs in usable.items() for _ in vecs]
    pool = [vec for vecs in usable.values() for vec in vecs]
    cross: list[float] = []
    n_cross_target = pairs_per_group * len(usable)
    attempts = 0
    while len(cross) < n_cross_target and attempts < n_cross_target * 20:
        attempts += 1
        i, j = rng.randrange(len(pool)), rng.randrange(len(pool))
        if labels[i] != labels[j]:
            cross.append(cosine(pool[i], pool[j]))
    if not same_group or not cross:
        return 0.0

    return sum(same_group) / len(same_group) - sum(cross) / len(cross)


@dataclass
class EmbeddingLiftProbe:
    """Embedding-space lift probe over ``sentence-transformers`` (extra ``adh[evals]``).

    **Honest caveat:** this measures geometric separability of aspect clusters in
    embedding space — a cheap proxy for downstream lift. It does NOT train on a
    cluster and measure eval improvement; that true fine-tune-lift probe is future
    work. Importantly, separability can overstate lift (embeddings may separate
    facets a small fine-tune cannot exploit) and cannot understate it reliably.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    pairs_per_group: int = 50
    _model: Any = field(default=None, repr=False, compare=False)  # lazy ST model cache

    def measure(self, groups: Mapping[str, Sequence[str]], *, seed: int = 0) -> float:
        model = self._ensure_model()
        texts: list[str] = []
        grouped: dict[str, list[int]] = {}
        for group, group_texts in groups.items():
            start = len(texts)
            texts.extend(str(t) for t in group_texts)
            grouped[group] = list(range(start, len(texts)))
        if not any(len(idxs) >= 2 for idxs in grouped.values()):
            return 0.0  # need >=2 texts per aspect for within-aspect pairs
        vectors = model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
        by_group = {
            g: [vectors[i] for i in idxs] for g, idxs in grouped.items() if len(idxs) >= 2
        }
        return cosine_separation(by_group, pairs_per_group=self.pairs_per_group, seed=seed)

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "EmbeddingLiftProbe requires 'sentence-transformers'; "
                    "install with: pip install 'adh[evals]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model
