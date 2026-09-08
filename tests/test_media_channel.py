"""The MEDIA channel — Engine A acquisition of licensed media assets.

Offline: :class:`InMemoryMediaProvider` serves canned hits with in-memory bytes;
the channel stages them content-addressed exactly as the live providers will.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from adh.channels import FetchContext
from adh.channels.media import InMemoryMediaProvider, MediaChannel, MediaHit
from adh.contract import JobSpec, SourceChannel
from adh.corpus.models import CorpusChunk
from adh.discovery.agent import SourceDiscoveryAgent, fixed_plan
from adh.discovery.models import CandidateSource, RankedSource, SourceEvaluation

CLIP = b"\x00\x00\x00\x18ftypisom(fake-clip-payload)"


def _hit(**over) -> MediaHit:
    fields = dict(
        media_id="1",
        page_url="https://www.pexels.com/video/1/",
        download_url="https://cdn.example/1.mp4",
        title="Couple dancing on the beach at sunset",
        tags=("dance", "sunset"),
        duration_s=6.0,
        width=1920,
        height=1080,
        license="pexels",
        attribution="Pexels: A Videographer (https://www.pexels.com/@a)",
        content=CLIP,
    )
    fields.update(over)
    return MediaHit(**fields)


def _spec(**over) -> JobSpec:
    data = {
        "name": "dance-i2v",
        "description": "dance i2v triplets",
        "motivation": "train LTX image-to-video",
        "intended_uses": ["supervised-fine-tuning"],
        "verifiability_tier": 3,
        "task_categories": ["image-to-video"],
        "languages": ["en"],
        "modality": "video",
        "target_size": {"records": 10},
        "record_schema_ref": "./schemas/video_caption_record.schema.json",
        "allowed_licenses": ["Pexels-License"],
        "output_license": "CC-BY-4.0",
        "generator_models": [],
        "judge_config": {"models": ["anthropic:a", "openai:b"], "min_score": 3.5},
        "splits": {"train": 1.0},
        "discovery": {"channels": ["media"]},
        "delivery_layout": "ltx",
        "limits": {"max_storage_mb": 500},
    }
    data.update(over)
    return JobSpec(**data)


class _NullStore:
    """The MEDIA channel never writes text into the corpus store."""

    def put(self, job_id: str, content: str) -> str:  # pragma: no cover - guard
        raise AssertionError("media channel must not offload text to the corpus store")

    def size_mb(self, job_id: str) -> float:
        return 0.0


@dataclass
class RecordingGovernor:
    stop_after: int | None = None  # stop once this many units were observed
    observed: list[tuple[str, str]] = field(default_factory=list)

    def observe(self, aspect_id: str, text: str) -> None:
        self.observed.append((aspect_id, text))

    def should_stop(self, aspect_id: str) -> bool:
        return self.stop_after is not None and len(self.observed) >= self.stop_after


def _source(locator: str = "couple dancing", aspect: str = "a0") -> RankedSource:
    cand = CandidateSource(
        source_id=f"{aspect}:media:{locator}",
        channel=SourceChannel.MEDIA,
        locator=locator,
        title=locator,
        aspect_id=aspect,
        license="pexels",
        discovered_via="test",
    )
    return RankedSource(
        candidate=cand,
        evaluation=SourceEvaluation(
            source_id=cand.source_id,
            shape_match=1.0,
            quality=1.0,
            license_ok=True,
            freshness_ok=True,
            estimated_yield=1,
            score=1.0,
        ),
    )


def _ctx(tmp_path: Path, spec: JobSpec, governor=None) -> FetchContext:
    return FetchContext(
        job_id="job",
        store=_NullStore(),
        spec=spec,
        staging_root=tmp_path / "media_staging",
        governor=governor or RecordingGovernor(),
        clock=lambda: datetime.now(timezone.utc),
    )


def _fetch(tmp_path: Path, hits: list[MediaHit], spec: JobSpec, governor=None):
    # locator "" = match-all query for the in-memory provider
    channel = MediaChannel(provider=InMemoryMediaProvider(hits=hits))
    return channel.fetch(_source(""), _ctx(tmp_path, spec, governor))


# ------------------------------------------------------------------- acquisition
def test_fetch_stages_clips_as_content_addressed_chunks(tmp_path):
    hits = [
        _hit(media_id="1", title="Beach dance", tags=("sunset",)),
        _hit(
            media_id="2",
            title="Studio spin",
            tags=("studio",),
            duration_s=4.0,
            content=b"\x00\x00\x00\x18ftypisom(different-clip-bytes)",
        ),
    ]
    spec = _spec(job_id="job")

    chunks = _fetch(tmp_path, hits, spec)

    assert len(chunks) == 2
    for chunk in chunks:
        staged = Path(chunk.content_uri)
        assert staged.exists()
        digest = hashlib.sha256(staged.read_bytes()).hexdigest()
        assert chunk.source_sha256 == digest
        assert chunk.n_bytes == staged.stat().st_size
        assert chunk.media_format == "mp4"
        assert chunk.duration_s in (6.0, 4.0)
        assert chunk.width == 1920 and chunk.height == 1080
    [first] = [c for c in chunks if c.duration_s == 6.0]
    # caption source metadata travels beside every clip for the packager
    meta = json.loads(Path(first.content_uri + ".meta.json").read_text())
    assert meta["title"] == "Beach dance" and meta["tags"] == ["sunset"]
    assert first.source_license == "pexels"
    assert "Beach dance" in first.text_preview


def test_fetch_chunk_ids_are_content_addressed(tmp_path):
    spec = _spec(job_id="job")
    first = _fetch(tmp_path, [_hit()], spec)
    again = _fetch(tmp_path, [_hit()], spec)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]


# ------------------------------------------------------------------ spec gates
def test_incompatible_licenses_are_skipped(tmp_path):
    spec = _spec(job_id="job")

    chunks = _fetch(
        tmp_path,
        [_hit(license="research-only"), _hit(media_id="2")],
        spec,
    )

    assert len(chunks) == 1
    assert chunks[0].source_license == "pexels"


def test_media_constraints_filter_duration_and_orientation(tmp_path):
    spec = _spec(
        job_id="job",
        discovery={
            "channels": ["media"],
            "constraints": {
                "media_min_duration_s": 5,
                "media_max_duration_s": 10,
                "media_orientation": "portrait",
            },
        },
    )
    hits = [
        _hit(media_id="short", duration_s=2.0),
        _hit(media_id="long", duration_s=30.0),
        _hit(media_id="landscape-ok"),
        _hit(media_id="portrait-ok", width=1080, height=1920),
    ]

    chunks = _fetch(tmp_path, hits, spec)

    assert len(chunks) == 1
    meta = json.loads(Path(chunks[0].content_uri + ".meta.json").read_text())
    assert meta["media_id"] == "portrait-ok"


def test_storage_budget_halts_acquisition(tmp_path):
    spec = _spec(job_id="job", limits={"max_storage_mb": 1})
    big = b"x" * 700_000
    hits = [_hit(media_id=str(i), content=big) for i in range(3)]

    chunks = _fetch(tmp_path, hits, spec)

    assert len(chunks) == 1  # one clip fits; the next blows the 1 MB budget


def test_governor_saturation_flushes_the_frontier(tmp_path):
    spec = _spec(job_id="job")
    governor = RecordingGovernor(stop_after=1)

    chunks = _fetch(tmp_path, [_hit(media_id=str(i)) for i in range(4)], spec, governor=governor)

    assert len(chunks) == 1
    assert len(governor.observed) == 1


def test_poison_download_skips_hit_without_killing_the_batch(tmp_path, caplog):
    """Live backends hit hotlink-403s and dead links; one bad asset never
    aborts acquisition of the rest (loud skip, not fatal)."""

    class Poison(InMemoryMediaProvider):
        def download(self, hit):
            raise RuntimeError(f"hotlink 403 for {hit.media_id}")

    spec = _spec(job_id="job")
    dead = _hit(media_id="dead", content=None)  # content=None forces provider.download
    good = _hit(media_id="good")
    channel = MediaChannel(provider=Poison(hits=[dead, good]))

    with caplog.at_level(logging.WARNING):
        chunks = channel.fetch(_source(""), _ctx(tmp_path, spec))

    assert len(chunks) == 1  # only the healthy asset survived
    assert any("dead" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------- locator
def test_locator_prefixes_strip_to_the_query(tmp_path):
    seen_queries: list[str] = []

    class Spy(InMemoryMediaProvider):
        def search(self, query, *, limit=10, media_types=None):
            seen_queries.append(query)
            return super().search(query, limit=limit, media_types=media_types)

    spec = _spec(job_id="job")
    channel = MediaChannel(provider=Spy(hits=[_hit()]))

    channel.fetch(_source("media://studio dance"), _ctx(tmp_path, spec))

    assert seen_queries == ["studio dance"]


# ------------------------------------------------------------- corpus manifest
def test_corpus_chunk_media_fields_round_trip_through_manifest():
    chunk = CorpusChunk(
        chunk_id="job:a0:abc",
        job_id="job",
        aspect_id="a0",
        source_channel=SourceChannel.MEDIA,
        source_url="https://www.pexels.com/video/1/",
        source_license="pexels",
        source_sha256="a" * 64,
        content_uri="/tmp/x.mp4",
        text_preview="preview",
        n_tokens=3,
        fetched_at=datetime.now(timezone.utc),
        media_format="mp4",
        duration_s=6.0,
        width=1080,
        height=1920,
        n_bytes=2048,
    )

    row = chunk.to_manifest_row()
    restored = CorpusChunk.from_manifest_row(row, text_preview="preview")

    assert restored.media_format == "mp4"
    assert (restored.duration_s, restored.width, restored.height, restored.n_bytes) == (
        6.0,
        1080,
        1920,
        2048,
    )
    legacy = dict(row)
    del legacy["duration_s"]
    bare = CorpusChunk.from_manifest_row(legacy, text_preview="p")
    assert bare.duration_s is None
    assert bare.n_bytes == 2048  # untouched fields still parse


# ---------------------------------------------------------------- discovery routing
@dataclass
class _StubLLM:
    """Deterministic stand-in: one aspect, fixed queries, perfect assessments."""

    topic: str = "couple dancing"
    queries_out: tuple[str, ...] = ("couple dancing beach",)
    known: bool = False

    def decompose(self, goal: str, *, max_aspects: int = 6):
        from adh.llm import AspectDraft

        return [AspectDraft(topic=self.topic, rationale="stub", keywords=("couple", "dancing"))]

    def queries(self, aspect, *, n: int = 4):
        return list(self.queries_out)[:n]

    def assess_candidate(self, *, target_shape, keywords, sample_text):
        from adh.llm import CandidateAssessment

        return CandidateAssessment(shape_match=1.0, quality=1.0, est_yield=100)

    def domain_is_model_known(self, aspect) -> bool:
        return self.known


def test_video_spec_routes_to_media_when_provider_present(tmp_path):
    provider = InMemoryMediaProvider(hits=[_hit()])
    agent = SourceDiscoveryAgent(media_provider=provider, llm=_StubLLM())

    plan = agent.discover(_spec(discovery={"channels": ["media"]}), tmp_path)

    assert plan.sources, "expected media sources"
    assert all(s.channel == SourceChannel.MEDIA for s in plan.sources)
    assert plan.sources[0].candidate.locator == "couple dancing beach"


def test_video_spec_without_provider_reports_loud_gap(tmp_path):
    agent = SourceDiscoveryAgent(llm=_StubLLM())

    plan = agent.discover(_spec(discovery={"channels": ["media"]}), tmp_path)

    assert not plan.sources
    assert any("media" in n.lower() for n in plan.notes)


def test_fixed_plan_routes_seed_prefixes_by_channel():
    spec = _spec(
        name="mixed",
        discovery={
            "channels": ["media", "datasets", "web"],
            "mode": "fixed",
            "seeds": ["pexels://couple dancing", "hf://u/dset", "https://example.com/page"],
        },
    )

    plan = fixed_plan(spec, "job", notes=[])

    by_locator = {s.candidate.locator: s.channel for s in plan.sources}
    assert by_locator["pexels://couple dancing"] == SourceChannel.MEDIA
    assert by_locator["hf://u/dset"] == SourceChannel.DATASETS
    assert by_locator["https://example.com/page"] == SourceChannel.WEB


# ------------------------------------------------- media-agnostic staging + gates
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x40\x00\x00\x00\xf0"
    b"\x08\x02\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _image_hit(**over):
    fields = dict(
        media_id="img-1",
        page_url="https://openweb.example/photo/1",
        download_url="https://cdn.openweb.example/1.png",
        title="A mountain lake at dawn",
        tags=("landscape",),
        width=320,
        height=240,
        license="unknown",
        attribution="A mountain lake at dawn — open-web page",
        media_type="image",
        content=PNG_BYTES,
    )
    fields.update(over)
    return MediaHit(**fields)


def _image_spec(**over):
    data = {
        **{
            k: v
            for k, v in _spec(job_id="job").model_dump().items()
            if k
            in {
                "name",
                "description",
                "motivation",
                "intended_uses",
                "verifiability_tier",
                "task_categories",
                "languages",
                "target_size",
                "record_schema_ref",
                "allowed_licenses",
                "output_license",
                "generator_models",
                "judge_config",
                "splits",
                "delivery_layout",
                "limits",
            }
        },
        "name": "mountain-images",
        "description": "landscape image triplets",
        "motivation": "train image captioning",
        "task_categories": ["image-captioning"],
        "record_schema_ref": "./schemas/image_caption_record.schema.json",
        "allowed_licenses": ["CC-BY-4.0"],
        "modality": "image",
        "discovery": {"channels": ["media"]},
        "delivery_layout": "hf",
    }
    data.update(over)
    return JobSpec(**data)


def test_image_hits_stage_with_typed_extension_and_sidecar(tmp_path):
    spec = _image_spec(discovery={"channels": ["media"], "allow_unknown_licenses": True})

    chunks = _fetch(tmp_path, [_image_hit()], spec)

    assert len(chunks) == 1
    chunk = chunks[0]
    staged = Path(chunk.content_uri)
    assert staged.suffix == ".png"  # extension follows the hit's media type
    assert chunk.media_format == "png"
    assert chunk.duration_s is None  # images carry no duration, ever
    meta = json.loads(Path(chunk.content_uri + ".meta.json").read_text())
    assert meta["media_type"] == "image"
    assert meta["format"] == "png"
    assert meta["license"] == "unknown"


def test_out_of_type_hits_are_skipped(tmp_path):
    # modality image -> wanted types {image, gif}
    spec = _image_spec(discovery={"channels": ["media"], "allow_unknown_licenses": True})

    chunks = _fetch(
        tmp_path,
        [_image_hit(), _hit(media_id="vid-1")],  # the video hit is out-of-type here
        spec,
    )

    assert len(chunks) == 1
    assert Path(chunks[0].content_uri).suffix == ".png"


def test_unknown_license_is_quarantined_by_default(tmp_path):
    spec = _image_spec()  # output CC-BY-4.0; unknown is compatible with nothing

    assert _fetch(tmp_path, [_image_hit()], spec) == []


def test_unknown_license_gate_opens_with_the_spec_flag(tmp_path):
    spec = _image_spec(discovery={"channels": ["media"], "allow_unknown_licenses": True})

    chunks = _fetch(tmp_path, [_image_hit()], spec)

    assert len(chunks) == 1
    meta = json.loads(Path(chunks[0].content_uri + ".meta.json").read_text())
    assert meta["license"] == "unknown"  # recorded as unknown, never invented


def test_duration_bounds_ignore_static_images(tmp_path):
    spec = _image_spec(
        discovery={
            "channels": ["media"],
            "allow_unknown_licenses": True,
            "constraints": {"media_min_duration_s": 5},  # would reject any video at 0.0s
        }
    )

    chunks = _fetch(tmp_path, [_image_hit()], spec)

    assert len(chunks) == 1  # temporal bounds don't apply to stills


# ------------------------------------------------------- routing + discovery gates
def test_image_spec_routes_to_media_exclusively(tmp_path):
    provider = InMemoryMediaProvider(
        hits=[
            _image_hit(
                media_id="img-dance",
                title="A couple dancing on a mountain at dawn",
                tags=("dance",),
            )
        ],
        supported_media_types=frozenset({"image", "png"}),
    )
    agent = SourceDiscoveryAgent(media_provider=provider, llm=_StubLLM())

    plan = agent.discover(_image_spec(), tmp_path)

    assert plan.sources
    assert all(s.channel == SourceChannel.MEDIA for s in plan.sources)


def test_license_ok_normalizes_aliases_and_honors_unknown_flag():
    from adh.discovery.agent import license_ok

    plain = _image_spec()  # allowed CC-BY-4.0
    assert license_ok("by", plain)  # Openverse short form lands on the canonical name
    assert not license_ok("unknown", plain)  # quarantined by default

    flagged = _image_spec(discovery={"channels": ["media"], "allow_unknown_licenses": True})
    assert license_ok("unknown", flagged)  # explicit opt-in opens the gate
    assert not license_ok(None, flagged)  # missing info ≠ declared unknown
