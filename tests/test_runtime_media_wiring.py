"""Runtime MEDIA wiring — one :class:`CompositeMediaProvider` from what the host has.

Whatever backends exist (API keys set, extras installed) compose into a single
provider in priority order [Openverse → Pexels → direct-web → yt-dlp]; every
absence is loud on stderr and none is fatal while another backend survives.
The keyless recipe (``examples/openverse_images.yaml``) validates too.
"""

from __future__ import annotations

import importlib.util

import pytest

from adh.channels.media import CompositeMediaProvider
from adh.contract import SourceChannel
from adh.runtime import live_channels


@pytest.fixture(autouse=True)
def _no_pexels_key(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)


def _names(channels) -> list[str]:
    media = channels.get(SourceChannel.MEDIA)
    if media is None:
        return []
    return [type(p).__name__ for p in media.provider.providers]


_YTDLP_INSTALLED = importlib.util.find_spec("yt_dlp") is not None


def test_media_channel_is_a_composite_even_without_keys_or_extras(capsys):
    channels = live_channels()

    media = channels.get(SourceChannel.MEDIA)
    assert isinstance(media.provider, CompositeMediaProvider)
    names = _names(channels)
    # the zero-config backends always wire, in composite priority order
    assert names[0] == "OpenverseImageProvider"
    assert "DirectWebMediaProvider" in names
    # no key → no Pexels; yt-dlp tracks webvideo-extra importability
    assert "PexelsMediaProvider" not in names
    assert ("YtDlpMediaProvider" in names) == _YTDLP_INSTALLED
    # the absence was announced, not swallowed
    assert "pexels" in capsys.readouterr().err.lower()


def test_pexels_joins_when_its_key_is_set(monkeypatch, capsys):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")

    names = _names(live_channels())

    assert names == [
        "OpenverseImageProvider",
        "PexelsMediaProvider",
        "DirectWebMediaProvider",
    ] + (["YtDlpMediaProvider"] if _YTDLP_INSTALLED else [])


def test_every_backend_absent_leaves_a_loud_gap_not_a_crash(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("host lacks this backend")

    monkeypatch.setattr("adh.channels.media.openverse.OpenverseImageProvider", boom)
    monkeypatch.setattr("adh.channels.media.direct_web.DirectWebMediaProvider", boom)
    monkeypatch.setattr("adh.runtime._find_module", lambda name: False)

    channels = live_channels()

    assert SourceChannel.MEDIA not in channels
    err = capsys.readouterr().err.lower()
    for backend in ("openverse", "pexels", "direct-web", "yt-dlp"):
        assert backend in err


# ------------------------------------------------------------- keyless recipe
def test_openverse_images_recipe_validates(repo_root):
    import yaml

    from adh.contract import JobSpec, check_job_spec

    raw = yaml.safe_load((repo_root / "examples" / "openverse_images.yaml").read_text())
    spec = JobSpec(**raw)

    report = check_job_spec(spec, base_dir=repo_root)

    assert report.errors == []
    assert spec.modality == "image"
    assert spec.discovery.allow_unknown_licenses is True
    assert spec.discovery.constraints.media_types == ["image", "gif"]
    assert spec.delivery_layout == "hf"
    assert spec.record_schema_ref == "./schemas/image_caption_record.schema.json"
