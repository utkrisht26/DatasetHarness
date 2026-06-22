"""Extra contract coverage: loader edges, license branches, validation warnings."""

from __future__ import annotations

import copy
import json

import pytest

from adh.contract import JobSpec, check_job_spec, load_record_schema
from adh.contract.licenses import is_compatible, normalize_license
from adh.contract.loader import load_job_spec, load_job_spec_from_str


# ----------------------------------------------------------------- licenses
def test_share_alike_only_compatible_with_itself():
    assert is_compatible("CC-BY-SA-4.0", "CC-BY-SA-4.0")
    assert not is_compatible("CC-BY-SA-4.0", "MIT")


def test_non_commercial_branch():
    assert is_compatible("CC-BY-NC-4.0", "CC-BY-NC-4.0")
    assert not is_compatible("CC-BY-NC-4.0", "CC-BY-4.0")


def test_normalize_aliases_and_passthrough():
    assert normalize_license("cc0") == "CC0-1.0"
    assert normalize_license("weird-custom-license") == "weird-custom-license"
    assert normalize_license("") == "unknown"


def test_unrecognized_source_only_same_license():
    assert is_compatible("GPL-3.0", "GPL-3.0")
    assert not is_compatible("GPL-3.0", "MIT")


# ----------------------------------------------------------------- loader
def test_load_job_spec_from_str(minimal_spec_dict):
    import yaml

    spec = load_job_spec_from_str(yaml.safe_dump(minimal_spec_dict))
    assert isinstance(spec, JobSpec)


def test_load_job_spec_from_str_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        load_job_spec_from_str("- just\n- a\n- list")


def test_load_job_spec_file_non_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_job_spec(p)


# ----------------------------------------------------------------- validation
def _spec_with_schema(minimal_spec_dict: dict, schema_name: str) -> JobSpec:
    d = copy.deepcopy(minimal_spec_dict)
    d["record_schema_ref"] = f"./{schema_name}"
    return JobSpec.model_validate(d)


def test_check_warns_when_no_additional_properties_false(minimal_spec_dict, tmp_path):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}  # no additionalProperties
    (tmp_path / "s.json").write_text(json.dumps(schema), encoding="utf-8")
    spec = _spec_with_schema(minimal_spec_dict, "s.json")
    report = check_job_spec(spec, base_dir=tmp_path)
    assert report.ok  # warning, not error
    assert any("additionalProperties" in w for w in report.warnings)


def test_check_warns_when_decontam_path_missing(minimal_spec_dict, tmp_path):
    schema = {"type": "object", "additionalProperties": False}
    (tmp_path / "s.json").write_text(json.dumps(schema), encoding="utf-8")
    d = copy.deepcopy(minimal_spec_dict)
    d["record_schema_ref"] = "./s.json"
    d["decontaminate_against"] = ["./eval/missing.jsonl"]
    spec = JobSpec.model_validate(d)
    report = check_job_spec(spec, base_dir=tmp_path)
    assert any("does not exist yet" in w for w in report.warnings)


def test_check_errors_on_invalid_json(minimal_spec_dict, tmp_path):
    (tmp_path / "s.json").write_text("{not json", encoding="utf-8")
    spec = _spec_with_schema(minimal_spec_dict, "s.json")
    report = check_job_spec(spec, base_dir=tmp_path)
    assert not report.ok
    assert any("not readable JSON" in e for e in report.errors)


def test_check_errors_on_bad_schema(minimal_spec_dict, tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"type": 123}), encoding="utf-8")
    spec = _spec_with_schema(minimal_spec_dict, "s.json")
    report = check_job_spec(spec, base_dir=tmp_path)
    assert not report.ok
    assert any("not a valid JSON Schema" in e for e in report.errors)


def test_load_record_schema_absolute(repo_root):
    schema = load_record_schema(str(repo_root / "schemas" / "qa_record.schema.json"))
    assert schema["title"] == "QA Record"
