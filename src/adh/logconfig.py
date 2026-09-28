"""Opt-in logging setup for the ``adh`` CLI and service.

The library never configures logging on import — that is the application's call.
Modules only ``logging.getLogger(__name__)`` and emit; the entry points (the
``adh`` CLI and ``adh serve``) call :func:`configure_logging` exactly once so
those emits become a readable stderr trace: the deep discovery agent's per-step
tool/model log, the crawler's progress, Engine B's drops.

Scoped deliberately to the ``adh`` logger namespace (never the root logger), so a
host process that imports ``adh`` as a library keeps full control of its own root
logger — we never hijack it. Propagation is left on: our entry points leave root
handler-less (so no double lines), while keeping records flowing to root means log
capture — pytest's ``caplog``, or a host's own root handler — still sees them.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from adh.settings import DEFAULT_LOG_LEVEL, LOG_LEVEL_ENV, get_settings

LOGGER_NAME = "adh"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_DEFAULT_LEVEL = DEFAULT_LOG_LEVEL
_ENV_VAR = LOG_LEVEL_ENV


def resolve_level(level: int | str | None) -> int:
    """Coerce a level (int, name like ``"debug"``, or ``None``) to a logging int.

    ``None`` falls back to ``$ADH_LOG_LEVEL`` then to :data:`_DEFAULT_LEVEL`. An
    unrecognised name degrades to ``INFO`` rather than raising — a bad flag should
    never crash the entry point.
    """
    if level is None:
        level = get_settings().log_level
    if isinstance(level, str):
        named = logging.getLevelName(level.strip().upper())
        return named if isinstance(named, int) else logging.INFO
    return int(level)


def configure_logging(
    level: int | str | None = None, *, stream: TextIO | None = None
) -> logging.Logger:
    """Install one stderr handler on the ``adh`` logger and return it.

    Idempotent: repeated calls only re-set the level (they never stack handlers),
    so the CLI dispatcher and ``adh serve`` can both call it without doubling
    every log line.
    """
    resolved = resolve_level(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved)

    for handler in logger.handlers:
        if getattr(handler, "_adh_managed", False):
            handler.setLevel(resolved)
            return logger

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    handler.setLevel(resolved)
    handler._adh_managed = True  # type: ignore[attr-defined]  # marks it ours for idempotency
    logger.addHandler(handler)
    return logger
