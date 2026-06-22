"""Tests for the Phase 0 contract (PHASE0-CONTRACT.md §0.3–§0.5)."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from adh.contract import (
    Discovery,
    DiscoveryMode,
    JobSpec,
    Record,
    SourceChannel,
    VerifiabilityTier,
    check_job_spec,
    get_verifier_hook,
    load_job_spec,
    load_record_schema,
    validate_record,
)
from adh.contract.families import distinct_families, model_family
from adh.contract.licenses import incompatible_licenses, is_compatible
from adh.contract.validation import RecordValidationError


# ---------------------------------------------------------------- families
def test_model_family_prefix():
    assert model_family("anthropic:claude-sonnet-4-6") == "anthropic"
    assert model_family("openai:gpt-judge") == "openai"


def test_model_family_bare_id_is_own_family():
    assert model_family("gpt-judge") == "gpt-judge"


def test_distinct_families_counts_providers():
    fams = distinct_families(["anthropic:a", "anthropic:b", "openai:c"])
    assert fams == {"anthropic", "openai"}


# ---------------------------------------------------------------- licenses
def test_permissive_source_flows_into_anything():
    assert is_compatible("MIT", "CC-BY-4.0")
    assert is_compatible("Apache-2.0", "MIT")


def test_unknown_license_never_compatible():
    assert not is_compatible("unknown", "MIT")
    assert not is_compatible("MIT", "unknown")


def test_cc_by_requires_attribution_preserved():
    assert is_compatible("CC-BY-4.0", "CC-BY-4.0")
    assert not is_compatible("CC-BY-4.0", "CC0-1.0")  # CC0 drops attribution


def test_incompatible_licenses_lists_offenders():
    bad = incompatible_licenses(["MIT", "CC-BY-4.0"], "CC0-1.0")
    assert bad == ["CC-BY-4.0"]


# ---------------------------------------------------------------- job spec OK
def test_minimal_spec_constructs(minimal_spec_dict):
    spec = JobSpec.model_validate(minimal_spec_dict)
    assert spec.verifiability_tier == VerifiabilityTier.SOFT
    assert spec.target_size.records == 100


def test_extra_fields_forbidden(minimal_spec_dict):
    minimal_spec_dict["bogus_field"] = 1
    with pytest.raises(ValidationError):
        JobSpec.model_validate(minimal_spec_dict)


# ---------------------------------------------------------------- invariants
def test_tier2_requires_judge_config(minimal_spec_dict):
    minimal_spec_dict.pop("judge_config")
    with pytest.raises(ValidationError, match="requires judge_config"):
        JobSpec.model_validate(minimal_spec_dict)


def test_tier2_requires_two_distinct_judge_families(minimal_spec_dict):
    minimal_spec_dict["judge_config"]["models"] = ["anthropic:a", "anthropic:b"]
    with pytest.raises(ValidationError, match="distinct"):
        JobSpec.model_validate(minimal_spec_dict)


def test_tier1_needs_no_judge(minimal_spec_dict):
    minimal_spec_dict["verifiability_tier"] = 1
    minimal_spec_dict.pop("judge_config")
    minimal_spec_dict["record_schema_ref"] = "./schemas/code_record.schema.json"
    spec = JobSpec.model_validate(minimal_spec_dict)
    assert spec.judge_config is None


def test_output_license_incompatible_rejected(minimal_spec_dict):
    minimal_spec_dict["allowed_licenses"] = ["CC-BY-4.0"]
    minimal_spec_dict["output_license"] = "CC0-1.0"
    with pytest.raises(ValidationError, match="incompatible"):
        JobSpec.model_validate(minimal_spec_dict)


def test_eval_split_requires_decontamination(minimal_spec_dict):
    minimal_spec_dict.pop("decontaminate_against")
    with pytest.raises(ValidationError, match="decontaminate_against"):
        JobSpec.model_validate(minimal_spec_dict)


def test_train_only_split_allows_empty_decontam(minimal_spec_dict):
    minimal_spec_dict["splits"] = {"train": 1.0}
    minimal_spec_dict.pop("decontaminate_against")
    spec = JobSpec.model_validate(minimal_spec_dict)
    assert spec.decontaminate_against == []


def test_splits_must_sum_to_one(minimal_spec_dict):
    minimal_spec_dict["splits"] = {"train": 0.8, "validation": 0.05}
    with pytest.raises(ValidationError, match="sum to 1"):
        JobSpec.model_validate(minimal_spec_dict)


def test_global_dedup_scope_warns(minimal_spec_dict):
    minimal_spec_dict["dedup_policy"] = {"scope": "global"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        JobSpec.model_validate(minimal_spec_dict)
    assert any("global" in str(w.message) for w in caught)


def test_target_size_exactly_one(minimal_spec_dict):
    minimal_spec_dict["target_size"] = {"records": 10, "tokens": 10}
    with pytest.raises(ValidationError, match="exactly one"):
        JobSpec.model_validate(minimal_spec_dict)


# ---------------------------------------------------------------- discovery
def test_fixed_mode_requires_seeds():
    with pytest.raises(ValidationError, match="fixed"):
        Discovery(mode=DiscoveryMode.FIXED, seeds=[])


def test_auto_mode_allows_no_seeds():
    d = Discovery(mode=DiscoveryMode.AUTO)
    assert d.seeds == []
    assert d.approval_gate is True


def test_synthesis_only_detected():
    d = Discovery(channels=[SourceChannel.SYNTHESIS])
    assert d.is_synthesis_only is True
    d2 = Discovery(channels=[SourceChannel.WEB, SourceChannel.SYNTHESIS])
    assert d2.is_synthesis_only is False


# ---------------------------------------------------------------- loader + fs checks
def test_load_example_spec(example_spec_path):
    spec = load_job_spec(example_spec_path)
    assert spec.name == "rust-async-macro-patterns"
    assert spec.limits.max_cost_usd == 150


def test_check_job_spec_passes_for_example(example_spec_path, repo_root):
    spec = load_job_spec(example_spec_path)
    report = check_job_spec(spec, base_dir=repo_root)
    assert report.ok, report.errors


def test_tier1_requires_x_verifier(minimal_spec_dict, repo_root, tmp_path):
    # Tier 1 pointed at a schema with NO x-verifier => error in fs check.
    minimal_spec_dict["verifiability_tier"] = 1
    minimal_spec_dict.pop("judge_config")
    minimal_spec_dict["record_schema_ref"] = "./schemas/qa_record.schema.json"
    spec = JobSpec.model_validate(minimal_spec_dict)
    report = check_job_spec(spec, base_dir=repo_root)
    assert not report.ok
    assert any("x-verifier" in e for e in report.errors)


def test_tier1_x_verifier_present_passes(minimal_spec_dict, repo_root):
    minimal_spec_dict["verifiability_tier"] = 1
    minimal_spec_dict.pop("judge_config")
    minimal_spec_dict["record_schema_ref"] = "./schemas/code_record.schema.json"
    spec = JobSpec.model_validate(minimal_spec_dict)
    report = check_job_spec(spec, base_dir=repo_root)
    assert report.ok, report.errors


def test_get_verifier_hook(repo_root):
    schema = load_record_schema("./schemas/code_record.schema.json", base_dir=repo_root)
    hook = get_verifier_hook(schema)
    assert hook is not None
    assert hook["entry"] == "adh.verifiers.code:run_pytest"


def test_missing_schema_file_is_error(minimal_spec_dict, repo_root):
    minimal_spec_dict["record_schema_ref"] = "./schemas/nope.schema.json"
    spec = JobSpec.model_validate(minimal_spec_dict)
    report = check_job_spec(spec, base_dir=repo_root)
    assert not report.ok
    assert any("not found" in e for e in report.errors)


# ---------------------------------------------------------------- record validation
def _good_record() -> dict:
    return {
        "record_id": "r1",
        "job_id": "j1",
        "split": "train",
        "input": "How do I write a declarative macro?",
        "output": "Use macro_rules! with a $($t:tt)* repetition.",
        "source_channel": "web",
        "source_url": "https://docs.rs/example",
        "source_license": "CC-BY-4.0",
        "model_id": "anthropic:claude-haiku-4-5",
        "prompt_hash": "sha256:abc",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_step": "text-generation",
        "verifiability_tier": 2,
    }


def test_validate_record_against_schema(repo_root):
    schema = load_record_schema("./schemas/qa_record.schema.json", base_dir=repo_root)
    rec = validate_record(_good_record(), schema=schema)
    assert isinstance(rec, Record)
    assert rec.source_channel == SourceChannel.WEB


def test_validate_record_rejects_unknown_field(repo_root):
    schema = load_record_schema("./schemas/qa_record.schema.json", base_dir=repo_root)
    bad = _good_record()
    bad["surprise"] = "x"
    with pytest.raises(RecordValidationError):
        validate_record(bad, schema=schema)


def test_validate_record_missing_required_field():
    bad = _good_record()
    del bad["model_id"]
    with pytest.raises(RecordValidationError) as exc:
        validate_record(bad)
    assert exc.value.record is bad  # carries the payload for dead-letter routing


def test_synthesis_record_has_null_source_url():
    rec = _good_record()
    rec["source_channel"] = "synthesis"
    rec["source_url"] = None
    model = validate_record(rec)
    assert model.source_url is None
