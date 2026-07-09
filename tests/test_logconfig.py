"""Tests for the opt-in logging setup (:mod:`adh.logconfig`).

The library must never configure logging on import; only the entry points call
:func:`configure_logging`. These tests pin the level resolution, idempotency
(one handler, not a stack), and that records actually reach a stream.
"""

from __future__ import annotations

import io
import logging

import pytest

from adh.logconfig import configure_logging, resolve_level

# The ``adh`` logger is isolated per-test by an autouse fixture in conftest.py.


@pytest.mark.parametrize(
    "value,expected",
    [
        ("debug", logging.DEBUG),
        ("INFO", logging.INFO),
        ("Warning", logging.WARNING),
        (10, 10),
        ("nonsense", logging.INFO),  # unknown name degrades, never raises
    ],
)
def test_resolve_level_coercions(value, expected):
    assert resolve_level(value) == expected


def test_resolve_level_uses_env_when_none(monkeypatch):
    monkeypatch.setenv("ADH_LOG_LEVEL", "DEBUG")
    assert resolve_level(None) == logging.DEBUG


def test_resolve_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv("ADH_LOG_LEVEL", raising=False)
    assert resolve_level(None) == logging.INFO


def test_configure_is_idempotent_single_handler():
    buf = io.StringIO()
    configure_logging("DEBUG", stream=buf)
    logger = configure_logging("WARNING", stream=buf)
    managed = [h for h in logger.handlers if getattr(h, "_adh_managed", False)]
    assert len(managed) == 1  # second call re-uses the handler, does not stack
    assert logger.level == logging.WARNING


def test_configure_emits_to_stream():
    buf = io.StringIO()
    configure_logging("INFO", stream=buf)
    logging.getLogger("adh.discovery.deep_agent").info("trace line %d", 7)
    out = buf.getvalue()
    assert "trace line 7" in out
    assert "adh.discovery.deep_agent" in out


def test_configure_keeps_propagation_for_capture():
    """Propagation stays on so pytest caplog / host root handlers still see us."""
    logger = configure_logging("INFO", stream=io.StringIO())
    assert logger.propagate is True
