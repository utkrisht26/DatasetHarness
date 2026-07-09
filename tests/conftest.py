"""Shared fixtures: paths to repo fixtures and a minimal-but-valid Job Spec dict."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_adh_logger():
    """Keep the process-global ``adh`` logger from leaking across tests.

    ``configure_logging`` (called by ``adh`` CLI tests via ``main()``) installs a
    handler bound to the current ``sys.stderr`` — which pytest's capture closes at
    test end, so a stale handler would poison later tests with "I/O on closed
    file" errors and make idempotency hide freshly-configured streams. Strip the
    harness-managed handlers before each test and restore the exact state after.
    """
    logger = logging.getLogger("adh")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers[:] = [h for h in logger.handlers if not getattr(h, "_adh_managed", False)]
    try:
        yield
    finally:
        logger.handlers[:], logger.level, logger.propagate = saved


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def example_spec_path() -> Path:
    return REPO_ROOT / "examples" / "rust_async_macros.yaml"


def _minimal_spec() -> dict:
    """A Tier-2 spec that satisfies every hard invariant (used as an editable base)."""
    return {
        "name": "demo",
        "description": "demo dataset",
        "motivation": "test",
        "intended_uses": ["supervised-fine-tuning"],
        "verifiability_tier": 2,
        "task_categories": ["question-answering"],
        "languages": ["en"],
        "target_size": {"records": 100},
        "record_schema_ref": "./schemas/qa_record.schema.json",
        "allowed_licenses": ["MIT", "Apache-2.0"],
        "generator_models": ["anthropic:claude-haiku-4-5"],
        "judge_config": {
            "models": ["anthropic:claude-sonnet-4-6", "openai:gpt-judge"],
            "min_score": 3.5,
        },
        "decontaminate_against": ["./eval/heldout_rust_qa.jsonl"],
        "output_license": "MIT",
        "splits": {"train": 0.9, "validation": 0.05, "test": 0.05},
    }


@pytest.fixture
def minimal_spec_dict() -> dict:
    return copy.deepcopy(_minimal_spec())
