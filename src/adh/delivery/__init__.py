"""Engine B delivery (OUTLINE §4.1) — dataset shards + documentation-as-byproduct."""

from __future__ import annotations

from .docs import croissant_json, dataset_card_md, datasheet_md, emit_docs
from .writer import DeliveryResult, write_dataset

__all__ = [
    "write_dataset",
    "DeliveryResult",
    "emit_docs",
    "datasheet_md",
    "croissant_json",
    "dataset_card_md",
]
