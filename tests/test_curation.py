"""Curation — dedup (2.4), decontamination (2.4), format/validate (2.5), offline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from adh.contract import (
    RecordValidationError,
    SourceChannel,
    VerifiabilityTier,
    load_record_schema,
)
from adh.curation import (
    Decontaminator,
    Deduper,
    assign_split,
    build_record,
    dead_letter_row,
    dedupe,
)
from adh.curation.dedup import DedupAssignment
from adh.generation import Candidate
from adh.verification import Verdict


# --------------------------------------------------------------------- dedup
def test_exact_dedup_clusters_identical():
    out = Deduper(method="exact").cluster(["hello world", "hello world", "different text"])
    assert out[0].cluster_id == out[1].cluster_id
    assert out[0].is_duplicate is False and out[1].is_duplicate is True
    assert out[2].cluster_id != out[0].cluster_id and out[2].is_duplicate is False


def test_minhash_dedup_clusters_near_duplicates():
    a = "the quick brown fox jumps over the lazy dog in the meadow every morning"
    near = "the quick brown fox jumps over the lazy dog in the meadow each morning"
    far = "completely unrelated content about database indexing and query planners here"
    out = Deduper(method="minhash", ngram=5, jaccard=0.6).cluster([a, near, far])
    assert out[0].cluster_id == out[1].cluster_id  # near-dup joined
    assert out[1].is_duplicate is True
    assert out[2].cluster_id != out[0].cluster_id  # unrelated stays separate


def test_minhash_pure_fallback_matches_semantics(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_datasketch(name, *args, **kwargs):
        if name == "datasketch":
            raise ImportError("simulated: datasketch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_datasketch)
    a = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    dup = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    out = Deduper(method="minhash", ngram=3, jaccard=0.8).cluster([a, dup])
    assert out[1].is_duplicate is True


@pytest.mark.parametrize("method", ["exact", "minhash"])
def test_incremental_index_matches_batch_clustering(method):
    # Engine B counts distinct records as they are verified; the count must agree
    # with what curation's batch clustering will later ship.
    texts = [
        "the quick brown fox jumps over the lazy dog in the meadow every morning",
        "the quick brown fox jumps over the lazy dog in the meadow every morning",
        "the quick brown fox jumps over the lazy dog in the meadow each morning",
        "completely unrelated content about database indexing and query planners here",
        "",
    ]
    deduper = Deduper(method=method, ngram=5, jaccard=0.6)

    index = deduper.index()
    incremental = [index.add(t) for t in texts]

    assert incremental == deduper.cluster(texts)
    assert index.n_distinct == sum(not a.is_duplicate for a in incremental)


def test_dedupe_helper_warns_on_global_scope(caplog):
    from adh.contract import DedupPolicy

    with caplog.at_level("WARNING"):
        dedupe(["a b c d e", "x y z"], policy=DedupPolicy(scope="global"))
    assert any("global" in r.message for r in caplog.records)


# --------------------------------------------------------------------- split-aware dedup
def test_cluster_per_split_joins_near_dups_within_a_split_only():
    a = "the quick brown fox jumps over the lazy dog in the meadow every morning"
    near = "the quick brown fox jumps over the lazy dog in the meadow each morning"
    d = Deduper(method="minhash", ngram=5, jaccard=0.6)
    out = d.cluster_per_split([("train", a), ("train", near), ("test", a)])

    assert out[0].cluster_id == out[1].cluster_id  # near-dup pair clustered within train
    assert out[0].is_duplicate is False and out[1].is_duplicate is True
    # identical text in another split is that shard's own representative — never joined
    assert out[2].is_duplicate is False
    assert out[2].cluster_id != out[0].cluster_id
    assert out[0].cluster_id.startswith("train:") and out[2].cluster_id.startswith("test:")


def test_cluster_per_split_never_spans_splits_and_counts_line_up():
    texts = [
        "alpha one two three",
        "alpha one two three",  # dup pair -> different splits below on purpose
        "beta four five six",
        "beta four five six",
        "gamma seven eight nine",
        "delta ten eleven twelve",
    ]
    splits = ["train", "validation", "train", "validation", "test", "train"]
    items = list(zip(splits, texts))
    d = Deduper(method="exact")
    out = d.cluster_per_split(items)

    assert len(out) == len(items)
    owner: dict[str, str] = {}
    for (split, _text), assign in zip(items, out):
        assert owner.setdefault(assign.cluster_id, split) == split  # no cluster spans splits
        assert assign.cluster_id.startswith(f"{split}:")
    # counts consistent with clustering each split independently
    expected_dups = sum(
        sum(
            1
            for x in Deduper(method="exact").cluster([t for s, t in items if s == sp])
            if x.is_duplicate
        )
        for sp in set(splits)
    )
    assert sum(1 for a in out if a.is_duplicate) == expected_dups


def test_deduper_public_api_backward_compatible():
    out = Deduper(method="exact").cluster(["same text", "same text"])
    assert out[0].is_duplicate is False and out[1].is_duplicate is True
    assert out[0].cluster_id == out[1].cluster_id
    assert out[0].cluster_id.startswith("clu-")  # plain cluster() ids carry no split prefix


# --------------------------------------------------------------------- decontam
def test_decontaminator_flags_overlap(tmp_path):
    held = tmp_path / "heldout.jsonl"
    held.write_text(
        '{"input": "what is the capital of france", "output": "paris"}\n', encoding="utf-8"
    )
    d = Decontaminator.from_paths([str(held)], ngram=4)
    assert d.active
    assert d.is_contaminated("tell me what is the capital of france please") is True
    assert d.is_contaminated("an unrelated sentence about rust async macros") is False


def test_decontaminator_empty_when_no_paths():
    d = Decontaminator.from_paths([])
    assert not d.active
    assert d.is_contaminated("anything at all goes here") is False


# --------------------------------------------------------------------- format / split
def test_assign_split_is_deterministic_and_proportioned():
    splits = {"train": 0.8, "validation": 0.1, "test": 0.1}
    ids = [f"rec-{i}" for i in range(400)]
    counts = {"train": 0, "validation": 0, "test": 0}
    for rid in ids:
        counts[assign_split(rid, splits, contaminated=False)] += 1
    assert counts["train"] > counts["validation"]  # roughly proportional
    # deterministic
    assert assign_split("rec-1", splits, contaminated=False) == assign_split(
        "rec-1", splits, contaminated=False
    )


def test_contaminated_record_never_lands_in_eval_split():
    splits = {"train": 0.0, "validation": 0.5, "test": 0.5}  # would force eval without the guard
    for i in range(50):
        assert assign_split(f"r{i}", splits, contaminated=True) == "train"


def _candidate(tier=2, **kw):
    base = dict(
        candidate_id="job:a1:abcdef123456",
        job_id="job",
        aspect_id="a1",
        verifiability_tier=VerifiabilityTier(tier),
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


def test_build_record_produces_valid_record():
    schema = load_record_schema("schemas/qa_record.schema.json")
    verdict = Verdict(True, "jury", judge_scores={"overall": 4.0}, judge_models=("m1", "m2"))
    rec = build_record(
        _candidate(tier=2),
        spec=_spec(),
        schema=schema,
        verdict=verdict,
        dedup=DedupAssignment("clu-1", False),
        contaminated=False,
    )
    assert rec.split in ("train", "validation", "test")
    assert rec.judge_scores == {"overall": 4.0}
    assert rec.dedup_cluster_id == "clu-1"


def test_build_record_rejects_contract_violation_via_dead_letter():
    schema = load_record_schema("schemas/qa_record.schema.json")
    verdict = Verdict(True, "jury", judge_scores={"overall": 4.0}, judge_models=("m1",))
    bad = _candidate(tier=2, output="")  # violates minLength:1 on output
    with pytest.raises(RecordValidationError) as ei:
        build_record(
            bad,
            spec=_spec(),
            schema=schema,
            verdict=verdict,
            dedup=DedupAssignment("c", False),
            contaminated=False,
        )
    row = dead_letter_row(bad, ei.value)
    assert row["candidate_id"] == "job:a1:abcdef123456" and row["error"]


def _spec(tier=2, **over):
    from adh.contract import JobSpec

    base = dict(
        name="job",
        description="d",
        motivation="m",
        intended_uses=["supervised-fine-tuning"],
        verifiability_tier=tier,
        task_categories=["question-answering"],
        languages=["en"],
        target_size={"records": 10},
        record_schema_ref="schemas/qa_record.schema.json",
        allowed_licenses=["MIT"],
        generator_models=[],
        decontaminate_against=["./eval/heldout_rust_qa.jsonl"],
        output_license="MIT",
        splits={"train": 0.9, "validation": 0.05, "test": 0.05},
    )
    if tier in (2, 3):
        base["judge_config"] = {"models": ["anthropic:x", "openai:y"], "min_score": 3.5}
    base.update(over)
    return JobSpec(**base)


# --------------------------------------------------------------------- split-aware curation
def _id_in_split(prefix: str, split: str, splits: dict) -> str:
    """A record id whose deterministic split assignment lands in ``split``."""
    i = 0
    while assign_split(f"{prefix}-{i}", splits, contaminated=False) != split:
        i += 1
    return f"{prefix}-{i}"


def test_curate_clusters_within_splits_end_to_end():
    from adh.orchestration.engine_b import _curate

    a = "the quick brown fox jumps over the lazy dog in the meadow every morning"
    near = "the quick brown fox jumps over the lazy dog in the meadow each morning"
    splits = {"train": 0.9, "validation": 0.05, "test": 0.05}
    # same threshold the sibling Deduper tests use: this pair's jaccard sits
    # between 0.6 and the 0.8 contract default (a mid-text swap breaks 5 of 9
    # five-gram shingles)
    spec = _spec(
        pii_policy="allow",
        dedup_policy={"scope": "per-shard", "method": "minhash", "ngram": 5, "jaccard": 0.6},
    )
    verdict = Verdict(True, "jury", judge_scores={"overall": 4.0}, judge_models=("m1", "m2"))
    kept = [
        (_candidate(candidate_id=_id_in_split("t", "train", splits), output=a), verdict),
        (_candidate(candidate_id=_id_in_split("n", "train", splits), output=near), verdict),
        (_candidate(candidate_id=_id_in_split("x", "test", splits), output=a), verdict),
    ]

    records, _aspects, dead, dups, pii = _curate(kept, spec=spec, schema=None, schema_base_dir=None)

    assert dead == [] and pii == {"pii_dropped": 0, "pii_redacted": 0}
    assert dups == 1  # the train near-dup; the identical test copy is its shard's representative
    by_split = {r.split: r for r in records}
    assert set(by_split) == {"train", "test"}
    for rec in records:
        assert rec.dedup_cluster_id.startswith(f"{rec.split}:")  # cluster lives inside its shard
        assert rec.is_duplicate is False
    assert by_split["train"].dedup_cluster_id != by_split["test"].dedup_cluster_id
