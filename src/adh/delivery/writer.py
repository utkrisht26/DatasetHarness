"""Dataset output (OUTLINE §4.1) — JSONL/Parquet shards + a manifest.

Writes verified records to one shard per split in the job's ``output_format`` and a
``manifest.jsonl`` listing every shard with its record count and SHA-256. Parquet uses
``pyarrow`` (the ``deliver`` extra); non-scalar record fields are JSON-encoded into
string columns so the Parquet schema stays flat and stable. Without pyarrow, Parquet
degrades to JSONL with a note rather than failing the run.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from adh.contract import Record
from adh.provenance import sha256_hex

logger = logging.getLogger(__name__)

_SCALAR = (str, int, float, bool, type(None))


@dataclass
class DeliveryResult:
    output_dir: Path
    format: str
    files: list[Path] = field(default_factory=list)
    counts_by_split: dict[str, int] = field(default_factory=dict)
    manifest_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return sum(self.counts_by_split.values())


def write_dataset(
    records: list[Record],
    *,
    spec,
    out_dir: str | Path,
    exit_reason: str = "completed",
) -> DeliveryResult:
    """Shard ``records`` by split into ``out_dir`` and write the dataset manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = (spec.output_format or "jsonl").lower()
    notes: list[str] = []

    by_split: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_split[r.split].append(r)

    if fmt == "parquet" and not _pyarrow_available():
        fmt = "jsonl"
        notes.append("pyarrow unavailable; wrote JSONL instead of Parquet (install adh[deliver])")

    files: list[Path] = []
    manifest_rows: list[dict] = []
    for split in sorted(by_split):
        rows = [r.model_dump(mode="json") for r in by_split[split]]
        if fmt == "parquet":
            path = out / f"{split}.parquet"
            _write_parquet(path, rows)
        else:
            path = out / f"{split}.jsonl"
            _write_jsonl(path, rows)
        files.append(path)
        manifest_rows.append(
            {
                "kind": "shard",
                "split": split,
                "path": path.name,
                "format": fmt,
                "n_records": len(rows),
                "sha256": sha256_hex(path.read_bytes()),
            }
        )

    counts = {split: len(v) for split, v in by_split.items()}
    manifest_path = out / "manifest.jsonl"
    summary = {
        "kind": "dataset",
        "job_id": spec.job_id or spec.name,
        "format": fmt,
        "exit_reason": exit_reason,
        "total_records": sum(counts.values()),
        "counts_by_split": counts,
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
        for row in manifest_rows:
            fh.write(json.dumps(row) + "\n")

    return DeliveryResult(
        output_dir=out,
        format=fmt,
        files=files,
        counts_by_split=counts,
        manifest_path=manifest_path,
        notes=notes,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flatten_for_parquet(row: dict) -> dict:
    """JSON-encode non-scalar values so every Parquet column is a flat scalar."""
    return {k: (v if isinstance(v, _SCALAR) else json.dumps(v, ensure_ascii=False)) for k, v in row.items()}


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([_flatten_for_parquet(r) for r in rows])
    pq.write_table(table, str(path))


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True
