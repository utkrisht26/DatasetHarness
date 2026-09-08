"""Live-stack wiring shared by the CLI and the gateway service.

One place answers "what does this machine actually have?" — the optional extras
(Tavily meta-search, HF dataset registry, Crawl4AI web channel), provider keys
from the environment, and the deepagents discovery stack. ``adh run`` and
``POST /api/runs`` compose these same helpers, so both entrypoints behave
identically: deep discovery over live ports when configured, deterministic
heuristic fallbacks otherwise, and every LLM call metered into the run's
:class:`UsageTracker`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Callable

logger = logging.getLogger(__name__)


def live_ports() -> tuple:
    """Best live search/registry ports available; in-memory stubs otherwise.

    Web meta-search selects the Tavily adapter when ``TAVILY_API_KEY`` is set
    (and the ``search`` extra is installed); the in-memory stub remains the
    offline default so discovery never hard-fails for want of a key.
    """
    from adh.search import InMemoryDatasetRegistry, InMemorySearchProvider

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            from adh.search.tavily_provider import TavilySearchProvider

            search = TavilySearchProvider(api_key=tavily_key)
        except Exception as exc:  # pragma: no cover - needs the search extra
            logger.warning("Tavily search unavailable (%s); using in-memory stub", exc)
            search = InMemorySearchProvider()
    else:
        search = InMemorySearchProvider()
    try:
        from adh.search.hf_registry import HfDatasetRegistry

        registry = HfDatasetRegistry()
    except Exception as exc:  # pragma: no cover - needs the datasets extra
        logger.warning("HF dataset registry unavailable (%s); using in-memory stub", exc)
        registry = InMemoryDatasetRegistry()
    return search, registry


def _find_module(name: str) -> bool:
    """Importability probe without importing — optional extras stay optional."""
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _live_media_providers() -> tuple[list, list[str]]:
    """Every media backend this host can serve, in composite priority order.

    Openverse (keyless) → Pexels (``PEXELS_API_KEY``) → direct-web harvest over
    the live search port → yt-dlp (``webvideo`` extra). Each absence becomes a
    loud gap entry, never an exception: a partial stack beats none.
    """
    providers: list = []
    gaps: list[str] = []
    try:
        from adh.channels.media.openverse import OpenverseImageProvider

        providers.append(OpenverseImageProvider())
    except Exception as exc:
        gaps.append(f"media/openverse unavailable: {exc}")
    try:
        from adh.channels.media.pexels import PexelsMediaProvider

        providers.append(PexelsMediaProvider.from_env())
    except Exception as exc:
        gaps.append(f"media/pexels unavailable: {exc}")
    try:
        from adh.channels.media.direct_web import DirectWebMediaProvider
        from adh.net.httpfetch import PoliteFetcher

        search, _registry = live_ports()
        providers.append(DirectWebMediaProvider(search_provider=search, fetcher=PoliteFetcher()))
    except Exception as exc:
        gaps.append(f"media/direct-web unavailable: {exc}")
    try:
        if not _find_module("yt_dlp"):
            raise ImportError("install the webvideo extra (`uv sync --extra webvideo`)")
        from adh.channels.media.ytdlp import YtDlpMediaProvider

        providers.append(YtDlpMediaProvider())
    except Exception as exc:
        gaps.append(f"media/yt-dlp unavailable: {exc}")
    return providers, gaps


def live_channels() -> dict:
    """Wire real acquisition channels when their extras are importable."""
    from adh.contract import SourceChannel

    channels: dict = {}
    try:
        from adh.channels import WebChannel
        from adh.channels.web import Crawl4AIContentFetcher

        channels[SourceChannel.WEB] = WebChannel(fetcher=Crawl4AIContentFetcher())
    except Exception as exc:  # pragma: no cover - needs the web extra
        print(f"[adh] web channel unavailable: {exc}", file=sys.stderr)
    try:
        from adh.channels import DatasetsChannel, HfDatasetLoader

        channels[SourceChannel.DATASETS] = DatasetsChannel(loader=HfDatasetLoader())
    except Exception as exc:  # pragma: no cover - needs the datasets extra
        print(f"[adh] datasets channel unavailable: {exc}", file=sys.stderr)
    try:
        # Media-agnostic by construction: whatever backends this host has compose
        # into ONE provider behind the MEDIA channel (open-web included).
        from adh.channels.media import CompositeMediaProvider, MediaChannel

        providers, gaps = _live_media_providers()
        for note in gaps:
            print(f"[adh] {note}", file=sys.stderr)
        if providers:
            channels[SourceChannel.MEDIA] = MediaChannel(provider=CompositeMediaProvider(providers))
        else:
            print("[adh] media channel unavailable: no backends installed", file=sys.stderr)
    except Exception as exc:
        # Loud, not fatal: media jobs degrade to a coverage-gap note rather than
        # a text-about-media crawl.
        print(f"[adh] media channel unavailable: {exc}", file=sys.stderr)
    return channels


def live_agent_factory(discovery_model: str | None) -> Callable:
    """Deep discovery over live ports when configured; heuristic otherwise.

    The returned factory receives the run's :class:`UsageTracker` so every
    deepagents model call is metered into that run's metrics (decision 4).
    ``discovery_model=None`` selects the heuristic agent outright.
    """
    from adh.discovery import (
        DeepAgentDiscoveryAgent,
        FallbackDiscoveryAgent,
        SourceDiscoveryAgent,
    )

    search, registry = live_ports()

    def factory(usage):
        heuristic = SourceDiscoveryAgent(search=search, registry=registry)
        if discovery_model is None:
            return heuristic
        return FallbackDiscoveryAgent(
            primary=DeepAgentDiscoveryAgent(
                model=discovery_model, search=search, registry=registry, usage=usage
            ),
            fallback=heuristic,
        )

    return factory
