"""Full discovery model responses before and after SDK/tool-call parsing.

HTTP hooks capture non-streaming OpenAI-compatible responses before the SDK reads
them. They are scoped to one model invocation, preserve existing HTTP clients and
hooks, and never collect request headers or credentials. Streaming bodies are not
eagerly consumed: doing so would change streaming latency and cancellation.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from adh.telemetry import TelemetryRecorder, current_telemetry, sanitize_attributes

logger = logging.getLogger(__name__)


@dataclass
class _ResponseContext:
    telemetry: TelemetryRecorder | None
    operation_id: str | None = None
    parent_operation_id: str | None = None


_response_context: ContextVar[_ResponseContext | None] = ContextVar(
    "adh_model_response_context", default=None
)


@contextmanager
def capture_model_responses() -> Iterator[None]:
    """Give each model invocation its own correlation context, including subagents."""
    token = _response_context.set(_ResponseContext(current_telemetry()))
    try:
        yield
    finally:
        _response_context.reset(token)


def bind_model_response_operation(
    operation_id: str,
    parent_operation_id: str | None,
    telemetry: TelemetryRecorder | None,
) -> None:
    """Associate HTTP hooks with the LangChain model-start callback's operation."""
    context = _response_context.get()
    if context is not None:
        context.operation_id = operation_id
        context.parent_operation_id = parent_operation_id
        context.telemetry = telemetry


def log_model_response(
    stage: str,
    response: Any,
    *,
    telemetry: TelemetryRecorder | None,
    operation_id: str | None,
    parent_operation_id: str | None,
    status: str = "succeeded",
) -> None:
    """Write a full redacted response to DEBUG and durable run telemetry."""
    try:
        if hasattr(response, "model_dump"):
            # LLMResult annotates messages as BaseMessage. Without duck-typed
            # serialization Pydantic omits AIMessage.tool_calls/invalid_tool_calls.
            response = response.model_dump(mode="python", serialize_as_any=True)
        payload = sanitize_attributes(response, truncate=False)
        logger.debug(
            "model response %s operation=%s: %s",
            stage,
            operation_id,
            json.dumps(payload, ensure_ascii=False),
        )
        if telemetry is not None:
            telemetry.emit(
                "discovery",
                f"model.response.{stage}",
                status=status,
                operation_id=operation_id,
                parent_operation_id=parent_operation_id,
                attributes={"response": payload},
                truncate_attributes=False,
            )
    except Exception as exc:  # diagnostics must not change model behavior
        logger.warning("could not log model response (%s)", type(exc).__name__)


def _record_http_response(response: Any, context: _ResponseContext) -> None:
    metadata = {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "request_id": response.headers.get("x-request-id") or response.headers.get("request-id"),
    }
    try:
        # Decode only the HTTP JSON envelope. function.arguments remains the
        # original string, including malformed JSON, escaping and whitespace.
        metadata["body"] = response.json()
    except ValueError:
        metadata["body_text"] = response.text
    log_model_response(
        "raw",
        metadata,
        telemetry=context.telemetry,
        operation_id=context.operation_id,
        parent_operation_id=context.parent_operation_id,
        status="failed" if response.is_error else "succeeded",
    )


def _capture_context(response: Any) -> _ResponseContext | None:
    context = _response_context.get()
    if context is not None and "text/event-stream" in response.headers.get("content-type", ""):
        log_model_response(
            "raw",
            {"capture_skipped": "streaming response; see the assembled parsed response"},
            telemetry=context.telemetry,
            operation_id=context.operation_id,
            parent_operation_id=context.parent_operation_id,
            status="skipped",
        )
        return None
    return context


def _capture_http_response(response: Any) -> None:
    if (context := _capture_context(response)) is None:
        return
    response.read()
    _record_http_response(response, context)


async def _capture_async_http_response(response: Any) -> None:
    if (context := _capture_context(response)) is None:
        return
    await response.aread()
    _record_http_response(response, context)


def install_response_logging(model: Any) -> bool:
    """Attach hooks to existing SDK HTTP clients without changing transport settings.

    OpenAI-compatible LangChain models expose their root SDK clients; the SDK's
    underlying httpx client owns the public event_hooks collection. Other model
    adapters still get the provider-independent parsed-response callback.
    """
    import httpx

    installed = False
    for name in ("root_client", "root_async_client", "client", "async_client"):
        sdk_client = getattr(model, name, None)
        http_client = getattr(sdk_client, "_client", None)
        if isinstance(http_client, httpx.AsyncClient):
            hook = _capture_async_http_response
        elif isinstance(http_client, httpx.Client):
            hook = _capture_http_response
        else:
            continue
        hooks = http_client.event_hooks.setdefault("response", [])
        if hook not in hooks:
            hooks.append(hook)
        installed = True
    return installed
