"""Job Spec builder: the contract, side-effect-free validation, and YAML exchange."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adh.contract import JobSpec, check_job_spec
from adh.contract.families import distinct_families
from adh.settings import get_settings


class SpecIssue(BaseModel):
    path: list[str | int] = Field(default_factory=list)
    message: str


class SpecValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: dict[str, Any]


class SpecImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    yaml_text: str = Field(max_length=200_000)


class SpecValidationResult(BaseModel):
    valid: bool
    errors: list[SpecIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Keep the parsed draft on import even when invalid so it can be repaired.
    draft: dict[str, Any] | None = None
    spec: JobSpec | None = None
    yaml_text: str | None = None


class RecordProfile(BaseModel):
    label: str
    modality: str
    record_schema_ref: str
    verifiability_tier: int


class SpecBuilderSchema(BaseModel):
    json_schema: dict[str, Any]
    defaults: dict[str, Any]
    profiles: list[RecordProfile]
    model_overrides: dict[str, list[str]]


def builder_schema() -> SpecBuilderSchema:
    # Required product choices live here; optional defaults come from Pydantic.
    draft = JobSpec.model_construct(
        name="",
        description="",
        motivation="",
        intended_uses=["supervised-fine-tuning"],
        verifiability_tier=2,
        task_categories=["question-answering"],
        languages=["en"],
        target_size={"records": 100, "tokens": None},
        record_schema_ref="./schemas/qa_record.schema.json",
        allowed_licenses=["MIT", "Apache-2.0"],
        generator_models=["openai:gpt-4o-mini"],
        judge_config={
            "models": ["openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"],
            "min_score": 3.5,
            "position_swap": True,
            "rubric_ref": None,
        },
        splits={"train": 1.0},
        output_license="Apache-2.0",
    ).model_dump(mode="json", warnings=False)
    settings = get_settings()
    return SpecBuilderSchema(
        json_schema=JobSpec.model_json_schema(mode="validation"),
        defaults=draft,
        profiles=[
            RecordProfile(
                label=label,
                modality=modality,
                record_schema_ref=f"./schemas/{file}",
                verifiability_tier=tier,
            )
            for label, modality, file, tier in [
                ("Text / question answering", "text", "qa_record.schema.json", 2),
                ("Code with tests", "code", "code_record.schema.json", 1),
                ("Image captions", "image", "image_caption_record.schema.json", 1),
                ("Video captions", "video", "video_caption_record.schema.json", 1),
            ]
        ],
        model_overrides={
            "generator_models": list(settings.generator_models),
            "judge_models": list(settings.judge_models),
        },
    )


def _issue_path(message: str) -> list[str]:
    # Cross-field Pydantic validators report the root; point the editor at the
    # setting that needs correction without duplicating the validation rules.
    for text, path in [
        ("x-verifier", ["record_schema_ref"]),
        ("record_schema", ["record_schema_ref"]),
        ("judge", ["judge_config"]),
        ("decontaminate", ["decontaminate_against"]),
        ("output_license", ["output_license"]),
        ("splits", ["splits"]),
        ("delivery_layout", ["delivery_layout"]),
    ]:
        if text in message:
            return path
    return []


def validate_spec_draft(
    data: dict[str, Any], *, base_dir: Path | None = None
) -> SpecValidationResult:
    result = SpecValidationResult(valid=False, draft=data)
    try:
        with warnings.catch_warnings(record=True) as caught:
            spec = JobSpec.model_validate(data)
        result.warnings.extend(str(w.message) for w in caught)
    except ValidationError as exc:
        result.errors = [
            SpecIssue(path=list(e["loc"]) or _issue_path(e["msg"]), message=e["msg"])
            for e in exc.errors(include_input=False, include_url=False, include_context=False)
        ]
        return result

    report = check_job_spec(spec, base_dir=base_dir)
    result.errors.extend(SpecIssue(path=_issue_path(e), message=e) for e in report.errors)
    result.warnings.extend(report.warnings)
    for name in ("name", "description", "motivation"):
        if not getattr(spec, name).strip():
            result.errors.append(SpecIssue(path=[name], message="Enter a non-empty value."))
    settings = get_settings()
    for key, models in [("generator", settings.generator_models), ("judge", settings.judge_models)]:
        if models:
            result.warnings.append(f"Deployment overrides {key} models: {', '.join(models)}")
    if (
        spec.verifiability_tier >= 2
        and settings.judge_models
        and len(distinct_families(list(settings.judge_models))) < 2
    ):
        result.warnings.append("The deployment judge override uses fewer than two model families.")
    result.valid = not result.errors
    if result.valid:
        result.spec = spec
        result.yaml_text = yaml.safe_dump(
            spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
    return result


def build_specs_router(*, base_dir: Path | None = None) -> APIRouter:
    router = APIRouter(tags=["specs"])

    @router.get("/schema", response_model=SpecBuilderSchema)
    def schema() -> SpecBuilderSchema:
        return builder_schema()

    @router.post("/validate", response_model=SpecValidationResult)
    def validate(body: SpecValidationRequest) -> SpecValidationResult:
        return validate_spec_draft(body.spec, base_dir=base_dir)

    @router.post("/import", response_model=SpecValidationResult)
    def import_yaml(body: SpecImportRequest) -> SpecValidationResult:
        try:
            data = yaml.safe_load(body.yaml_text)
            if not isinstance(data, dict):
                raise ValueError("Job Spec YAML must contain a mapping of field names to values.")
            # Reject aliases that form cycles and YAML values that are not JSON
            # data before returning the editable draft to the browser.
            import json

            data = json.loads(json.dumps(data, allow_nan=False))
        except (yaml.YAMLError, ValueError, TypeError, RecursionError) as exc:
            return SpecValidationResult(
                valid=False, errors=[SpecIssue(message=f"Invalid YAML: {exc}")]
            )
        return validate_spec_draft(data, base_dir=base_dir)

    return router
