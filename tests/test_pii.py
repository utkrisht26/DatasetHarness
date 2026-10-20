"""PII policy enforcement (OUTLINE §2.4, ADR-0004) — offline via injected fakes.

Covers both policy modes end-to-end through ``engine_b._curate``, the fail-closed
posture when the optional Presidio extra is missing, and (skipped unless the extra
is installed) a minimal run against the real ``PresidioRedactor`` adapter.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from adh.contract import JobSpec, SourceChannel, VerifiabilityTier
from adh.curation.pii import (
    ConfigurationError,
    PIIEntity,
    PIIRedactor,
    resolve_redactor,
)
from adh.generation import Candidate
from adh.orchestration.engine_b import _curate
from adh.verification import Verdict


# --------------------------------------------------------------------- fakes/helpers
class FakeRedactor:
    """Deterministic offline PIIRedactor: '@'-bearing tokens are EMAIL_ADDRESS."""

    name = "fake"

    def analyze(self, text: str) -> list[PIIEntity]:
        return [
            PIIEntity("EMAIL_ADDRESS", m.start(), m.end(), 0.99)
            for m in re.finditer(r"\S+@\S+", text)
        ]

    def redact(self, text: str, entities: list[PIIEntity]) -> str:
        for e in sorted(entities, key=lambda e: e.start, reverse=True):
            text = text[: e.start] + f"<{e.entity_type}>" + text[e.end :]
        return text


def _candidate(**kw) -> Candidate:
    base = dict(
        candidate_id="job:a1:abcdef123456",
        job_id="job",
        aspect_id="a1",
        verifiability_tier=VerifiabilityTier(2),
        input="Explain async macros.",
        output="Async macros expand at compile time to reduce boilerplate.",
        source_channel=SourceChannel.WEB,
        source_url="https://docs.rs/tokio",
        source_license="MIT",
        source_sha256="abc",
        model_id="anthropic:claude-haiku-4-5",
        prompt_hash="sha256:deadbeef",
        generated_at=datetime.now(timezone.utc),
        task_category="question-answering",
    )
    base.update(kw)
    return Candidate(**base)


def _verdict() -> Verdict:
    return Verdict(True, "jury", judge_scores={"overall": 4.0}, judge_models=("m1", "m2"))


def _spec(**over) -> JobSpec:
    base = dict(
        name="job",
        description="d",
        motivation="m",
        intended_uses=["supervised-fine-tuning"],
        verifiability_tier=2,
        task_categories=["question-answering"],
        languages=["en"],
        target_size={"records": 10},
        record_schema_ref="schemas/qa_record.schema.json",
        allowed_licenses=["MIT"],
        generator_models=[],
        judge_config={"models": ["anthropic:x", "openai:y"], "min_score": 3.5},
        decontaminate_against=["./eval/heldout_rust_qa.jsonl"],
        output_license="MIT",
        splits={"train": 0.9, "validation": 0.05, "test": 0.05},
    )
    base.update(over)
    return JobSpec(**base)


def _curate_one(cand: Candidate, *, spec: JobSpec, redactor: FakeRedactor | None = None):
    records, aspects, dead, dups, pii = _curate(
        [(cand, _verdict())], spec=spec, schema=None, schema_base_dir=None, redactor=redactor
    )
    return records, dead, pii


# --------------------------------------------------------------------- redact mode
def test_redact_mode_rewrites_text_and_appends_provenance():
    spec = _spec(pii_policy="redact")
    cand = _candidate(
        input="Email jane@example.org please.", output="Contact bob@example.org today."
    )

    records, dead, pii = _curate_one(cand, spec=spec, redactor=FakeRedactor())

    assert dead == []
    assert len(records) == 1
    rec = records[0]
    assert rec.input == "Email <EMAIL_ADDRESS> please."
    assert rec.output == "Contact <EMAIL_ADDRESS> today."
    note = rec.generation_params["pii_redaction"]
    assert note["tool"] == "fake" and note["policy"] == "redact"
    assert note["entity_types"] == ["EMAIL_ADDRESS"]
    assert pii == {"pii_dropped": 0, "pii_redacted": 1}


def test_redact_mode_leaves_clean_text_untouched():
    spec = _spec(pii_policy="redact")
    cand = _candidate()  # no '@' tokens -> no entities

    records, dead, pii = _curate_one(cand, spec=spec, redactor=FakeRedactor())

    assert dead == [] and len(records) == 1
    assert records[0].output == cand.output
    assert "pii_redaction" not in records[0].generation_params
    assert pii == {"pii_dropped": 0, "pii_redacted": 0}


# --------------------------------------------------------------------- drop mode
def test_drop_mode_dead_letters_pii_bearing_candidates():
    spec = _spec(pii_policy="drop")
    cand = _candidate(output="Reach jane@example.org for access.")

    records, dead, pii = _curate_one(cand, spec=spec, redactor=FakeRedactor())

    assert records == []
    assert len(dead) == 1
    assert "pii_policy='drop'" in dead[0]["error"]
    assert "EMAIL_ADDRESS" in dead[0]["error"]
    assert dead[0]["candidate_id"] == cand.candidate_id
    assert pii == {"pii_dropped": 1, "pii_redacted": 0}


def test_drop_mode_keeps_clean_candidates():
    spec = _spec(pii_policy="drop")
    records, dead, pii = _curate_one(_candidate(), spec=spec, redactor=FakeRedactor())
    assert len(records) == 1 and dead == []
    assert pii == {"pii_dropped": 0, "pii_redacted": 0}


# --------------------------------------------------------------------- allow mode
def test_allow_mode_ships_text_unscanned():
    spec = _spec(pii_policy="allow")
    cand = _candidate(output="Reach jane@example.org for access.")

    records, dead, pii = _curate_one(cand, spec=spec, redactor=FakeRedactor())

    assert len(records) == 1 and dead == []
    assert records[0].output == cand.output  # untouched, even though a redactor exists
    assert pii == {"pii_dropped": 0, "pii_redacted": 0}


# --------------------------------------------------------------------- fail-closed
def _no_presidio() -> None:
    raise ImportError("simulated: presidio extra not installed")


def test_fail_closed_raises_when_explicit_policy_without_presidio(monkeypatch):
    import adh.curation.pii as pii_mod

    monkeypatch.setattr(pii_mod, "_import_presidio", _no_presidio)
    spec = _spec(pii_policy="drop")  # explicitly configured -> must not be silently skipped

    with pytest.raises(ConfigurationError, match=r"adh\[pii\]"):
        _curate([], spec=spec, schema=None, schema_base_dir=None)


def test_fail_closed_raises_for_explicit_redact_without_presidio(monkeypatch):
    import adh.curation.pii as pii_mod

    monkeypatch.setattr(pii_mod, "_import_presidio", _no_presidio)
    spec = _spec(pii_policy="redact")

    with pytest.raises(ConfigurationError, match="presidio"):
        _curate([], spec=spec, schema=None, schema_base_dir=None)


def test_injected_redactor_bypasses_presidio_requirement(monkeypatch):
    import adh.curation.pii as pii_mod

    monkeypatch.setattr(pii_mod, "_import_presidio", _no_presidio)
    spec = _spec(pii_policy="redact")

    records, dead, pii = _curate_one(
        _candidate(output="Mail jane@example.org."), spec=spec, redactor=FakeRedactor()
    )
    assert len(records) == 1 and dead == []
    assert pii["pii_redacted"] == 1


def test_contract_default_degrades_loudly_without_presidio(monkeypatch, caplog):
    """Default (unset) pii_policy='redact': run proceeds but the skip is NOT silent."""
    import adh.curation.pii as pii_mod

    monkeypatch.setattr(pii_mod, "_import_presidio", _no_presidio)
    spec = _spec()  # no pii_policy key -> implicit contract default "redact"
    notes: list[str] = []

    with caplog.at_level("WARNING"):
        records, _aspects, dead, _dups, pii = _curate(
            [(_candidate(), _verdict())], spec=spec, schema=None, schema_base_dir=None, notes=notes
        )

    assert len(records) == 1 and dead == []
    assert pii == {"pii_dropped": 0, "pii_redacted": 0}
    assert any("pii" in r.message.lower() for r in caplog.records)
    assert any("pii" in n.lower() for n in notes)  # surfaced into run notes/summary


def test_round_tripped_spec_keeps_default_implicit(monkeypatch):
    """Regression: specs persist via a complete model_dump(), so a reload marks
    every defaulted field as set — explicitness must ride the VALUE (None vs a
    policy string), or every service-mediated run would fail-closed on build."""
    import adh.curation.pii as pii_mod

    monkeypatch.setattr(pii_mod, "_import_presidio", _no_presidio)
    stored = JobSpec(**_spec().model_dump(mode="json"))  # what load_spec() sees
    notes: list[str] = []

    # implicit default survives the round-trip -> loud degradation, not raise
    redactor = resolve_redactor(stored, notes=notes)
    assert redactor is None
    assert any("NOT enforced" in n for n in notes)

    # an explicitly configured policy also keeps failing closed across the trip
    explicit_stored = JobSpec(**_spec(pii_policy="drop").model_dump(mode="json"))
    with pytest.raises(ConfigurationError):
        resolve_redactor(explicit_stored)


# --------------------------------------------------------------------- port shape
def test_piiredactor_is_a_runtime_checkable_protocol():
    # runtime-checkable Protocols verify structure at the instance level, not
    # nominal inheritance (a method-presence ``issubclass`` would return True)
    redactor: PIIRedactor = FakeRedactor()
    assert isinstance(redactor, PIIRedactor)

    class NotARedactor:
        pass

    assert not isinstance(NotARedactor(), PIIRedactor)


# --------------------------------------------------------------------- integration
def test_presidio_redactor_integration_minimal():
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("presidio_anonymizer")
    from adh.curation.pii import PresidioRedactor

    try:
        redactor = PresidioRedactor(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], score_threshold=0.5)
    except Exception as exc:  # noqa: BLE001 - model binaries are a deploy concern
        pytest.skip(f"presidio NER model unavailable in this environment: {exc}")

    text = "Reach me at jane.doe@example.org or +1 555 010 9999 tomorrow."
    entities = redactor.analyze(text)
    assert any(e.entity_type == "EMAIL_ADDRESS" for e in entities)
    redacted = redactor.redact(text, entities)
    assert "jane.doe@example.org" not in redacted
    assert "<EMAIL_ADDRESS>" in redacted


def _presidio_redactor(**kw):
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("presidio_anonymizer")
    from adh.curation.pii import PresidioRedactor

    try:
        return PresidioRedactor(**kw)
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001 - model binaries are a deploy concern
        pytest.skip(f"presidio NER model unavailable in this environment: {exc}")


def test_default_presidio_entities_leave_technical_text_intact():
    # Live run rust-async-live-small: generic NER turned Rust -> <DATE_TIME>,
    # Unpin/Axum -> <PERSON>, Tokio -> <LOCATION>, TCP -> <NRP> and every URL ->
    # <URL> — 23 detections, all false positives. Defaults are direct identifiers.
    redactor = _presidio_redactor()
    technical = (
        "In Rust's Pin API, when T: Unpin, Pin<&mut T> behaves like &mut T. "
        "Tokio schedules tasks; Axum binds a TCP listener on port 3000. "
        "See https://docs.rs/tokio/latest/tokio/task/ for details."
    )
    assert redactor.analyze(technical) == []

    personal = "Contact jane.doe@example.org; my phone number is (212) 736-5000."
    found = {e.entity_type for e in redactor.analyze(personal)}
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER"} <= found


def test_default_entity_set_is_direct_identifiers_only():
    from adh.curation.pii import DEFAULT_PII_ENTITIES

    assert "EMAIL_ADDRESS" in DEFAULT_PII_ENTITIES
    assert "CREDIT_CARD" in DEFAULT_PII_ENTITIES
    for noisy in ("PERSON", "LOCATION", "DATE_TIME", "NRP", "URL"):
        assert noisy not in DEFAULT_PII_ENTITIES


def test_pii_entities_env_override_reaches_the_redactor(monkeypatch):
    pytest.importorskip("presidio_analyzer")
    monkeypatch.setenv("ADH_PII_ENTITIES", "person, EMAIL_ADDRESS")
    try:
        redactor = resolve_redactor(_spec(pii_policy="redact"))
    except OSError as exc:  # spaCy model missing
        pytest.skip(f"presidio NER model unavailable in this environment: {exc}")
    assert redactor is not None
    assert redactor.entities == ("PERSON", "EMAIL_ADDRESS")


def test_unknown_pii_entity_fails_fast():
    with pytest.raises(ConfigurationError, match="NOT_AN_ENTITY"):
        _presidio_redactor(entities=["EMAIL_ADDRESS", "NOT_AN_ENTITY"])
