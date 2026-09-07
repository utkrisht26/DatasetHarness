"""Engine B delivery (OUTLINE §4.1) — dataset shards + documentation-as-byproduct.

Also re-exports the Phase 4 publishing/eval surface: ``publish_dataset`` (§4.3, HF
Hub upload with an injected HubClient port) and the §4.2 offline quality gate
(``evaluate_dataset`` / ``assert_publishable`` / lift probes).
"""

from __future__ import annotations

from .docs import croissant_json, dataset_card_md, datasheet_md, emit_docs
from .evals import (
    DatasetQualityReport,
    EmbeddingLiftProbe,
    LengthStats,
    LiftProbe,
    NotPublishableError,
    PIIRedactor,
    assert_publishable,
    cosine_separation,
    evaluate_dataset,
)
from .publish import (
    HubClient,
    HfApiHubClient,
    PublishFile,
    PublishRefusedError,
    PublishResult,
    default_hub_client,
    publish_dataset,
)
from .writer import DeliveryResult, write_dataset

__all__ = [
    # 4.1 output
    "write_dataset",
    "DeliveryResult",
    "emit_docs",
    "datasheet_md",
    "croissant_json",
    "dataset_card_md",
    # 4.2 dataset-quality evals
    "evaluate_dataset",
    "DatasetQualityReport",
    "LengthStats",
    "assert_publishable",
    "NotPublishableError",
    "PIIRedactor",
    "LiftProbe",
    "EmbeddingLiftProbe",
    "cosine_separation",
    # 4.3 publishing
    "publish_dataset",
    "PublishResult",
    "PublishFile",
    "PublishRefusedError",
    "HubClient",
    "HfApiHubClient",
    "default_hub_client",
]
