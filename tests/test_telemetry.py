"""Tests for durable, redacted per-run telemetry."""

from __future__ import annotations

import json

import pytest

from adh.telemetry import TelemetryRecorder, read_telemetry


def test_full_response_telemetry_keeps_long_strings_and_all_items(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "full-response")
    arguments = '{ "query" : "' + "long query " * 400 + '", "api_key": "hidden-secret" }'
    response = {
        "arguments": arguments,
        "extra_fields": {f"field{i}": i for i in range(70)},
        "blocks": [str(i) for i in range(70)],
        "password": "hidden-password",
    }
    recorder.emit(
        "discovery",
        "model.response.raw",
        attributes={"response": response},
        truncate_attributes=False,
    )
    saved = read_telemetry(tmp_path)[0]["attributes"]["response"]
    assert saved["arguments"] == arguments.replace("hidden-secret", "[REDACTED]")
    assert saved["extra_fields"] == response["extra_fields"]
    assert saved["blocks"] == response["blocks"]
    assert saved["password"] == "[REDACTED]"
    assert "hidden-secret" not in json.dumps(saved)


def test_ordinary_telemetry_remains_size_bounded(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "bounded")
    recorder.emit("discovery", "tool.completed", attributes={"output": "x" * 3000})
    assert read_telemetry(tmp_path)[0]["attributes"]["output"] == "x" * 2000 + "…"


def test_recorder_orders_events_and_redacts_secrets(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "run-1", job_id="demo")
    recorder.emit(
        "discovery",
        "tool.started",
        attributes={
            "query": "rust async",
            "api_key": "super-secret",
            "headers": {"authorization": "Bearer live-secret"},
            "provider_message": "request used sk-or-v1-abcdefghijklmnop",
        },
    )
    recorder.emit("discovery", "tool.completed", status="succeeded")

    events = read_telemetry(tmp_path)
    assert [event["sequence"] for event in events] == [1, 2]
    assert events[0]["attributes"]["query"] == "rust async"
    assert events[0]["attributes"]["api_key"] == "[REDACTED]"
    assert events[0]["attributes"]["headers"]["authorization"] == "[REDACTED]"
    assert "sk-or-v1" not in events[0]["attributes"]["provider_message"]
    assert read_telemetry(tmp_path, after_sequence=1) == [events[1]]


def test_span_records_failure_and_duration(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "run-1")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        with recorder.span("discovery", "agent"):
            raise RuntimeError("provider unavailable")

    started, failed = read_telemetry(tmp_path)
    assert started["event"] == "agent.started"
    assert failed["event"] == "agent.failed"
    assert failed["status"] == "failed"
    assert failed["duration_ms"] is not None
    assert failed["attributes"]["error_type"] == "RuntimeError"
