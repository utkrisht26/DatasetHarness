"""Tests for the HF Hub registry adapter (mapping logic + missing-extra path)."""

from __future__ import annotations

import pytest

from adh.search.hf_registry import HfDatasetRegistry, HfHubUnavailableError


class _StubDataset:
    def __init__(self, id_, tags, downloads, card):
        self.id = id_
        self.tags = tags
        self.downloads = downloads
        self.card_data = card
        self.last_modified = "2024-01-01"


class _StubApi:
    def __init__(self, datasets):
        self._datasets = datasets
        self.calls = []

    def list_datasets(self, search, limit, sort, direction=None):
        self.calls.append((search, limit, sort, direction))
        return self._datasets[:limit]


def test_hf_registry_maps_results():
    api = _StubApi(
        [
            _StubDataset("org/rust-qa", ["rust", "qa"], 4200, {"license": "mit"}),
            _StubDataset("org/other", ["misc"], 10, {}),
        ]
    )
    reg = HfDatasetRegistry(api=api)
    hits = reg.search("rust qa", limit=5)
    assert hits[0].dataset_id == "org/rust-qa"
    assert hits[0].locator == "https://huggingface.co/datasets/org/rust-qa"
    assert hits[0].license == "mit"
    assert hits[0].downloads == 4200
    assert api.calls and api.calls[0][0] == "rust qa"


def test_hf_registry_missing_extra_raises(monkeypatch):
    # Simulate missing extra by raising HfHubUnavailableError on _require_hf
    import adh.search.hf_registry

    def mock_require():
        raise HfHubUnavailableError("mocked unavailability")

    monkeypatch.setattr(adh.search.hf_registry, "_require_hf", mock_require)
    reg = HfDatasetRegistry()
    with pytest.raises(HfHubUnavailableError):
        reg.search("anything")
