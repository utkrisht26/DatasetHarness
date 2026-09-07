"""Dataset output (OUTLINE §4.1) — JSONL/Parquet shards + a manifest.

Writes verified records to one shard per split in the job's ``output_format`` and a
``manifest.jsonl`` listing every shard with its record count and SHA-256. Parquet uses
``pyarrow`` (the ``deliver`` extra); non-scalar record fields are JSON-encoded into
string columns so the Parquet schema stays flat and stable. Without pyarrow, Parquet
degrades to JSONL with a note rather than failing the run.
"""

from __future__ import annotations

import datetime
import decimal
import json
import logging
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, get_args, get_origin

from pydantic import AnyUrl

from adh.contract import Record
from adh.provenance import sha256_hex

logger = logging.getLogger(__name__)

_SCALAR = (str, int, float, bool, type(None))

# Types whose JSON-dumped form is always a scalar (model_dump(mode="json") turns
# datetimes/URLs into strings; enums dump to their value).
_SERIAL_SCALARS = (
    str,
    int,
    float,
    bool,
    bytes,
    type(None),
    datetime.datetime,
    datetime.date,
    datetime.time,
    decimal.Decimal,
    AnyUrl,
)


def _is_serial_scalar(annotation: object) -> bool:
    """True when a Record field of this annotation always dumps to a scalar."""
    origin = get_origin(annotation)
    if origin is None:
        return isinstance(annotation, type) and issubclass(annotation, _SERIAL_SCALARS)
    if origin is Union or origin is types.UnionType:
        return all(_is_serial_scalar(a) for a in get_args(annotation))
    return False


# Record fields _flatten_for_parquet JSON-encodes into string columns (everything
# non-scalar once dumped: lists, dicts, nested models, Optional wrappers thereof).
# Invariant: encode is value-based (`isinstance(v, _SCALAR)`) while decode is
# annotation-based — the two stay consistent only while every Record field whose
# annotation is a serial-scalar always *holds* a scalar at dump time. A future
# field typed e.g. `str | list[str]` would break that; extend _is_serial_scalar
# handling or switch encode to annotation-based instead.
_JSON_ENCODED_FIELDS: frozenset[str] = frozenset(
    name for name, f in Record.model_fields.items() if not _is_serial_scalar(f.annotation)
)


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
    staged_media_root: str | Path | None = None,
) -> DeliveryResult:
    """Shard ``records`` by split into ``out_dir`` and write the dataset manifest.

    Media-backed records (video jobs) are additionally packaged into the spec's
    delivery layout from ``staged_media_root``, and every shipped asset is
    registered in the manifest as a ``kind:"asset"`` row (sha256, bytes, probe
    metadata, referencing record ids).
    """
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

    # ---- media assets (video jobs): copy staged clips into the delivery layout
    asset_rows: list[dict] = []
    media_records = [r for r in records if r.media is not None]
    if media_records:
        if staged_media_root is None:
            notes.append(
                f"{len(media_records)} media-backed record(s) but no staged_media_root "
                "given; clips were NOT packaged into the dataset"
            )
        else:
            from adh.delivery.layouts import resolve_layout

            layout_out = resolve_layout(spec.delivery_layout)(
                media_records, out, Path(staged_media_root)
            )
            files.extend(layout_out.files)
            by_sha: dict[str, list[Record]] = {}
            for r in media_records:
                assert r.media is not None
                by_sha.setdefault(r.media.sha256, []).append(r)
            for sha in sorted(by_sha):
                group = by_sha[sha]
                media = group[0].media
                assert media is not None
                asset_rows.append(
                    {
                        "kind": "asset",
                        "split": group[0].split,
                        "path": layout_out.paths_by_sha[sha],
                        "format": media.format,
                        "sha256": sha,
                        "bytes": media.bytes,
                        "duration_s": media.duration_s,
                        "width": media.width,
                        "height": media.height,
                        # provenance travels with the asset: the publish gate
                        # counts unknown licenses from these rows
                        "license": group[0].source_license or "unknown",
                        "attribution": str(media.attribution or group[0].source_url),
                        "record_ids": [r.record_id for r in group],
                    }
                )

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
        for row in [*manifest_rows, *asset_rows]:
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


def decode_parquet_row(row: dict) -> dict:
    """Invert :func:`_flatten_for_parquet` for the nested Record fields.

    Only contract list/dict fields are decoded back from their JSON string
    columns; free text that merely *looks* like JSON passes through untouched,
    so schema validation sees the same semantic row a JSONL reader would.
    """
    out = {}
    for key, value in row.items():
        if key in _JSON_ENCODED_FIELDS and isinstance(value, str):
            try:
                out[key] = json.loads(value)
            except ValueError:
                out[key] = value  # foreign/legacy shard: keep the raw string
        else:
            out[key] = value
    return out


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
