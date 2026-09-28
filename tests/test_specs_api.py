"""The spec builder validates drafts without running discovery, crawlers, or models."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adh.contract import JobSpec
from adh.service.routers.specs import build_specs_router

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(build_specs_router(base_dir=ROOT), prefix="/api/specs")
    return TestClient(app)


def test_contract_is_generated_and_text_defaults_are_buildable(client):
    contract = client.get("/api/specs/schema").json()
    assert contract["json_schema"] == JobSpec.model_json_schema(mode="validation")
    spec = contract["defaults"]
    spec.update(
        name="Credit cards", description="Indian card fees and rewards", motivation="Compare cards"
    )
    result = client.post("/api/specs/validate", json={"spec": spec}).json()
    assert result["valid"], result
    assert result["spec"]["verifiability_tier"] == 2
    assert result["spec"]["record_schema_ref"] == "./schemas/qa_record.schema.json"
    assert yaml.safe_load(result["yaml_text"]) == result["spec"]


def test_every_profile_has_a_compatible_schema(client):
    contract = client.get("/api/specs/schema").json()
    for profile in contract["profiles"]:
        spec = copy.deepcopy(contract["defaults"])
        spec.update(name="Dataset", description="Goal", motivation="Use")
        spec.update({k: v for k, v in profile.items() if k != "label"})
        if spec["verifiability_tier"] == 1:
            spec["judge_config"] = None
        result = client.post("/api/specs/validate", json={"spec": spec}).json()
        assert result["valid"], (profile, result)


@pytest.mark.parametrize(
    ("update", "path"),
    [
        (
            {
                "record_schema_ref": "http://schema.org/Text",
                "verifiability_tier": 1,
                "judge_config": None,
            },
            ["record_schema_ref"],
        ),
        ({"verifiability_tier": 1, "judge_config": None}, ["record_schema_ref"]),
        ({"judge_config": {"models": ["openai:a", "openai:b"]}}, ["judge_config"]),
        ({"splits": {"train": 0.7}}, ["splits"]),
        (
            {"splits": {"train": 0.8, "test": 0.2}, "decontaminate_against": []},
            ["decontaminate_against"],
        ),
        ({"limits": {"max_llm_tokens": -1}}, ["limits", "max_llm_tokens"]),
        ({"target_size": {"tokens": 10, "records": 10}}, ["target_size"]),
        ({"unexpected_field": True}, ["unexpected_field"]),
        ({"name": "   "}, ["name"]),
    ],
)
def test_invalid_drafts_have_field_errors(client, minimal_spec_dict, update, path):
    spec = {**minimal_spec_dict, **update}
    response = client.post("/api/specs/validate", json={"spec": spec})
    assert response.status_code == 200
    result = response.json()
    assert not result["valid"]
    assert any(e["path"] == path for e in result["errors"]), result
    assert result["spec"] is None
    assert result["yaml_text"] is None


def test_yaml_roundtrip_preserves_all_settings(client, minimal_spec_dict):
    spec = copy.deepcopy(minimal_spec_dict)
    spec.update(
        generation_params={"temperature": 0.3, "stop": ["END"]},
        maintainer="Dataset team",
        out_of_scope_uses=["Lending decisions"],
        discovery={
            "mode": "seeded",
            "seeds": ["https://example.com"],
            "channels": ["web"],
            "allow_unknown_licenses": True,
            "constraints": {"include_domains": ["example.com"], "recency_after": "2026-01-01"},
        },
        limits={"max_pages_scraped": 125, "max_llm_tokens": 9000, "max_runtime_seconds": 600},
    )
    result = client.post("/api/specs/import", json={"yaml_text": yaml.safe_dump(spec)}).json()
    assert result["valid"], result
    again = client.post("/api/specs/import", json={"yaml_text": result["yaml_text"]}).json()
    assert again["spec"] == result["spec"]
    assert result["spec"]["generation_params"] == spec["generation_params"]
    assert result["spec"]["limits"]["max_runtime_seconds"] == 600
    assert result["spec"]["discovery"]["constraints"]["recency_after"] == "2026-01-01"


@pytest.mark.parametrize(
    "text",
    [
        "[one, two]",
        "name: [",
        '!!python/object/apply:os.system ["echo unsafe"]',
        "&cycle {self: *cycle}",
        "null",
    ],
)
def test_bad_yaml_is_rejected_without_applying_a_draft(client, text):
    response = client.post("/api/specs/import", json={"yaml_text": text})
    assert response.status_code == 200
    assert not response.json()["valid"]
    assert response.json()["draft"] is None


def test_incomplete_import_remains_editable(client):
    result = client.post("/api/specs/import", json={"yaml_text": "name: Saved goal"}).json()
    assert not result["valid"]
    assert result["draft"] == {"name": "Saved goal"}
    assert result["errors"]


def test_builder_does_not_expose_credentials(client, monkeypatch):
    monkeypatch.setenv("ADH_LLM_API_KEY", "sensitive-value-for-test")
    monkeypatch.setenv("ADH_GENERATOR_MODELS", "openai:test-generator")
    monkeypatch.setenv("ADH_JUDGE_MODELS", "openai:test-judge")
    contract = client.get("/api/specs/schema").json()
    assert contract["model_overrides"]["generator_models"] == ["openai:test-generator"]
    assert "sensitive-value-for-test" not in json.dumps(contract)
    draft = contract["defaults"]
    draft.update(name="Name", description="Goal", motivation="Reason")
    result = client.post("/api/specs/validate", json={"spec": draft}).json()
    assert any("fewer than two" in w for w in result["warnings"])
    assert "sensitive-value-for-test" not in json.dumps(result)
