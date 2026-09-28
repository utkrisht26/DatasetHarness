"""Exercise raw HTTP capture with the real model adapter, entirely offline."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from adh.discovery.deep_agent import _build_tracer
from adh.llm import UsageTracker, build_chat_model
from adh.llm.response_logging import capture_model_responses, install_response_logging
from adh.telemetry import TelemetryRecorder, bind_telemetry, read_telemetry


def _completion(arguments: str) -> dict:
    return {
        "id": "chatcmpl-diagnostic",
        "object": "chat.completion",
        "created": 0,
        "model": "diagnostic-model",
        "provider": "diagnostic-provider",
        "unknown_provider_fields": list(range(70)),
        "api_key": "body-secret",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-diagnostic",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": arguments},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


@pytest.mark.parametrize("malformed", [False, True])
def test_raw_and_parsed_responses_preserve_arguments_and_correlate(tmp_path, caplog, malformed):
    arguments = '{ "query" : "' + "India credit card rewards " * 150 + '" }'
    if malformed:
        arguments = arguments[:-1]
    body = _completion(arguments)
    recorder = TelemetryRecorder(tmp_path, "response-diagnostic")
    usage = UsageTracker()
    existing_hook_calls = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)),
        event_hooks={
            "response": [lambda response: existing_hook_calls.append(response.status_code)]
        },
    ) as client:
        model = build_chat_model(
            "openai:diagnostic-model",
            base_url="https://provider.test/v1",
            api_key="request-secret",
            http_client=client,
            max_retries=0,
        )
        assert install_response_logging(model)
        assert install_response_logging(model)  # reusing a model must not duplicate hooks
        with (
            bind_telemetry(recorder),
            capture_model_responses(),
            caplog.at_level(logging.DEBUG, logger="adh.llm.response_logging"),
        ):
            message = model.invoke(
                "Find credit cards",
                config={"callbacks": [_build_tracer(usage, telemetry=recorder)]},
            )

    assert existing_hook_calls == [200]
    assert usage.calls == 1 and usage.total.total_tokens == 120
    events = read_telemetry(tmp_path)
    raw = [e for e in events if e["event"] == "model.response.raw"]
    parsed = [e for e in events if e["event"] == "model.response.parsed"]
    assert len(raw) == len(parsed) == 1
    started = next(e for e in events if e["event"] == "model.started")
    assert raw[0]["operation_id"] == parsed[0]["operation_id"] == started["operation_id"]
    assert raw[0]["sequence"] < parsed[0]["sequence"]
    provider_response = raw[0]["attributes"]["response"]["body"]
    assert (
        provider_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        == arguments
    )
    assert provider_response["unknown_provider_fields"] == list(range(70))
    assert provider_response["provider"] == "diagnostic-provider"
    assert provider_response["api_key"] == "[REDACTED]"
    generation = parsed[0]["attributes"]["response"]["generations"][0][0]
    assert generation["generation_info"]["finish_reason"] == "tool_calls"
    if malformed:
        assert generation["message"]["invalid_tool_calls"][0]["args"] == arguments
        assert message.invalid_tool_calls
    else:
        assert generation["message"]["tool_calls"][0]["args"] == json.loads(arguments)
        assert message.tool_calls[0]["args"] == json.loads(arguments)
    assert "body-secret" not in caplog.text
    assert "request-secret" not in caplog.text
    assert json.dumps(arguments, ensure_ascii=False) in caplog.text


def test_malformed_http_json_is_saved_before_sdk_failure(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "bad-json")
    invalid_body = '{"choices": [not valid JSON'
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, text=invalid_body, headers={"content-type": "application/json"}
            )
        )
    ) as client:
        model = build_chat_model(
            "openai:diagnostic-model",
            base_url="https://provider.test/v1",
            api_key="dummy",
            http_client=client,
            max_retries=0,
        )
        install_response_logging(model)
        with (
            bind_telemetry(recorder),
            capture_model_responses(),
            pytest.raises(json.JSONDecodeError),
        ):
            model.invoke(
                "diagnose", config={"callbacks": [_build_tracer(None, telemetry=recorder)]}
            )
    events = read_telemetry(tmp_path)
    raw = next(e for e in events if e["event"] == "model.response.raw")
    assert raw["attributes"]["response"]["body_text"] == invalid_body
    failure = next(e for e in events if e["event"] == "model.failed")
    assert raw["sequence"] < failure["sequence"]
    assert raw["operation_id"] == failure["operation_id"]


def test_async_http_capture_preserves_body_and_run_context(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "async-response")
    body = _completion('{"query": ": "}')

    async def invoke():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body))
        ) as client:
            model = SimpleNamespace(root_async_client=SimpleNamespace(_client=client))
            assert install_response_logging(model)
            with bind_telemetry(recorder), capture_model_responses():
                response = await client.get("https://provider.test/v1")
                assert response.json() == body

    asyncio.run(invoke())
    raw = read_telemetry(tmp_path)[0]
    assert raw["event"] == "model.response.raw"
    assert raw["attributes"]["response"]["body"]["id"] == "chatcmpl-diagnostic"


def test_streaming_response_is_not_eagerly_consumed(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "stream-response")
    chunks_read = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            chunks_read.append(True)
            yield b"data: [DONE]\n\n"

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, stream=Stream(), headers={"content-type": "text/event-stream"}
            )
        )
    ) as client:
        install_response_logging(SimpleNamespace(root_client=SimpleNamespace(_client=client)))
        with bind_telemetry(recorder), capture_model_responses():
            with client.stream("GET", "https://provider.test/v1") as response:
                assert not chunks_read
                assert response.read() == b"data: [DONE]\n\n"
    assert read_telemetry(tmp_path)[0]["status"] == "skipped"
