"""Delivery layouts — trainer-facing bundle builders (the extensibility seam).

A *layout* turns accepted records + the staged media tree into the on-disk
bundle a trainer ingests, returning every file it wrote so the manifest,
croissant metadata, and publish plan can register them.

Built-ins:

* ``hf``  — HuggingFace-style default: clips copied content-addressed to
  ``media/<sha256>.<ext>`` (exactly each record's ``MediaRef.path``).
* ``ltx`` — LTX-Video-style finetuning layout: ``videos/<index>_<sha8>.<ext>``
  in stable ``(split, record_id)`` order plus an aligned ``metadata.jsonl``
  (``{"file_name", "caption"}`` rows) and ``prompts.txt``.

Custom layouts register via :func:`register_layout` and are selected per spec
through ``JobSpec.delivery_layout`` (validated by ``check_job_spec``).

Missing staged media fails closed with a loud error listing every gap — a
dataset that ships broken links is worse than one that refuses to build.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from adh.contract import MediaRef, Record

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayoutOutput:
    """What a layout wrote: every file plus each asset's dataset-relative path.

    ``paths_by_sha`` maps media sha256 -> POSIX path relative to ``out_dir``, so
    the manifest's ``kind:"asset"`` rows can point at the shipped file without
    reverse-engineering layout naming conventions.
    """

    files: tuple[Path, ...] = ()
    paths_by_sha: dict[str, str] = field(default_factory=dict)


LayoutFn = Callable[[Sequence[Record], Path, Path], LayoutOutput]

_BUILTIN_LAYOUTS: dict[str, LayoutFn] = {}


def register_layout(name: str, fn: LayoutFn) -> None:
    """Register (or replace) a delivery layout under ``name``."""
    _BUILTIN_LAYOUTS[name.strip().lower()] = fn


def known_layouts() -> tuple[str, ...]:
    """Names of every registered layout, sorted for stable messages."""
    return tuple(sorted(_BUILTIN_LAYOUTS))


def resolve_layout(name: str) -> LayoutFn:
    try:
        return _BUILTIN_LAYOUTS[name.strip().lower()]
    except KeyError:
        raise KeyError(
            f"unknown delivery layout {name!r} (available: {', '.join(known_layouts())})"
        ) from None


def _ext_of(media: MediaRef | None, default: str = "mp4") -> str:
    if media is None or not media.format:
        return default
    return media.format.lstrip(".") or default


def _locate_staged(records: Sequence[Record], staged_root: Path) -> dict[str, Path]:
    """Map sha256 -> staged clip path; fail closed listing every missing clip."""
    staged_root = Path(staged_root)
    found: dict[str, Path] = {}
    missing: list[str] = []
    for record in records:
        media = record.media
        if media is None or media.sha256 in found:
            continue
        matches = sorted(staged_root.rglob(f"{media.sha256}.{_ext_of(media)}"))
        if matches:
            found[media.sha256] = matches[0]
        else:
            missing.append(f"{media.path} (sha256={media.sha256})")
    if missing:
        raise RuntimeError(
            f"staged media missing for {len(missing)} clip(s): {'; '.join(missing)} "
            "(media_staging was likely pruned; re-run acquisition before delivering)"
        )
    return found


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".adh-tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def hf_layout(records: Sequence[Record], out_dir: Path, staged_root: Path) -> LayoutOutput:
    """Default layout: ``media/<sha256>.<ext>`` — exactly each ``MediaRef.path``."""
    staged = _locate_staged(records, staged_root)
    out_dir = Path(out_dir)
    written: list[Path] = []
    by_sha: dict[str, str] = {}
    for record in records:
        media = record.media
        if media is None or media.sha256 in by_sha:
            continue  # content-addressed: identical assets copy once
        src = staged[media.sha256]
        dst = out_dir / "media" / src.name
        _atomic_copy(src, dst)
        written.append(dst)
        by_sha[media.sha256] = dst.relative_to(out_dir).as_posix()
    return LayoutOutput(files=tuple(written), paths_by_sha=by_sha)


def ltx_layout(records: Sequence[Record], out_dir: Path, staged_root: Path) -> LayoutOutput:
    """LTX-Video trainer layout: ``videos/`` + ``metadata.jsonl`` + ``prompts.txt``.

    Clip names derive from the ``(split, record_id)`` order so rebuilds are
    byte-stable; ``metadata.jsonl`` rows are ``{"file_name", "caption"}`` and
    ``prompts.txt`` repeats the captions line-aligned for trainers that want a
    plain prompt file.
    """
    ordered = sorted(
        (r for r in records if r.media is not None),
        key=lambda r: (r.split, r.record_id),
    )
    staged = _locate_staged(ordered, staged_root)
    out_dir = Path(out_dir)

    video_paths: list[Path] = []
    by_sha: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    prompts: list[str] = []
    for index, record in enumerate(ordered):
        media = record.media
        assert media is not None  # filtered above
        name = f"{index:05d}_{media.sha256[:8]}.{_ext_of(media)}"
        dst = out_dir / "videos" / name
        _atomic_copy(staged[media.sha256], dst)
        video_paths.append(dst)
        rel = dst.relative_to(out_dir).as_posix()
        by_sha.setdefault(media.sha256, rel)
        rows.append({"file_name": rel, "caption": record.output})
        prompts.append(record.output)

    meta_path = out_dir / "metadata.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    prompts_path = out_dir / "prompts.txt"
    prompts_path.write_text("".join(f"{line}\n" for line in prompts), encoding="utf-8")

    logger.info("ltx layout: %d clip(s) + metadata.jsonl + prompts.txt", len(video_paths))
    return LayoutOutput(files=(*video_paths, meta_path, prompts_path), paths_by_sha=by_sha)


register_layout("hf", hf_layout)
register_layout("ltx", ltx_layout)
