"""Offline end-to-end demo of Engine A + Engine B (no network, no API keys).

Wires the deterministic in-memory backends (standing in for Crawl4AI / HF Hub / a
search provider) so you can watch the full path — Discovery -> Approval ->
Acquisition (Engine A) -> Seed -> Generate -> Tiered-Verify -> Dedup/Decontam ->
Format -> Deliver (Engine B) — turn a goal into a provenance-tagged corpus and then
into a verified, deduplicated dataset with a datasheet. This is also the template for
injecting *real* backends: swap the in-memory ports for ``Crawl4AIContentFetcher`` /
``HfDatasetLoader`` and the heuristic generator/judge for ``LlmGenerator`` /
``LlmJuryPanel`` (the ``agents`` extra).

Run:  uv run python examples/demo_offline.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from adh.channels import (
    DatasetsChannel,
    InMemoryContentFetcher,
    InMemoryDatasetLoader,
    WebChannel,
)
from adh.contract import SourceChannel, load_job_spec_from_str
from adh.discovery import AutoApprover, SourceDiscoveryAgent, render_plan
from adh.generation import HeuristicGenerator
from adh.orchestration import run_engine_a, run_engine_b
from adh.search import (
    DatasetHit,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
)
from adh.verification import HeuristicJudge

SPEC_YAML = """
name: rust-async-demo
description: "Rust async macro patterns. Tokio runtime internals scheduler."
motivation: "teach async"
intended_uses: ["supervised-fine-tuning"]
verifiability_tier: 2
task_categories: ["question-answering"]
languages: ["en"]
target_size: { records: 100 }
record_schema_ref: "./schemas/qa_record.schema.json"
discovery: { mode: "auto", approval_gate: true }
allowed_licenses: ["MIT", "Apache-2.0"]
generator_models: ["anthropic:claude-haiku-4-5"]
judge_config: { models: ["anthropic:claude-sonnet-4-6", "openai:gpt-judge"] }
decontaminate_against: ["./eval/heldout_rust_qa.jsonl"]
output_license: "MIT"
output_format: "jsonl"
splits: { train: 0.9, validation: 0.05, test: 0.05 }
"""


def build_agent() -> SourceDiscoveryAgent:
    registry = InMemoryDatasetRegistry(
        datasets=[
            DatasetHit(
                dataset_id="org/rust-async-macros-qa",
                locator="https://huggingface.co/datasets/org/rust-async-macros-qa",
                title="Rust async macro Q/A",
                description="rust async macro design declarative procedural "
                "question-answering input output",
                license="MIT",
                downloads=5000,
                last_modified="2024-09-01",
            )
        ],
        samples={
            "org/rust-async-macros-qa": "rust async macro declarative procedural input output"
        },
    )
    search = InMemorySearchProvider(
        pages=[
            SearchHit(
                url="https://tokio.rs/runtime-internals",
                title="Tokio runtime internals scheduler",
                snippet="tokio runtime internals scheduler tasks input output",
                published="2024-02-01",
                license_hint="MIT",
            )
        ]
    )
    return SourceDiscoveryAgent(registry=registry, search=search)


def build_channels() -> dict:
    loader = InMemoryDatasetLoader(
        records={
            "org/rust-async-macros-qa": [
                "Q: How do I write a declarative macro?\nA: Use macro_rules! with $($t:tt)*.",
                "Q: What pins an async state machine?\nA: pin_project for structural pinning.",
            ]
        }
    )
    fetcher = InMemoryContentFetcher(
        pages={
            "https://tokio.rs/runtime-internals": "# Tokio runtime internals\n\n"
            "The multi-threaded scheduler uses a work-stealing run queue ..."
        }
    )
    return {
        SourceChannel.DATASETS: DatasetsChannel(loader=loader),
        SourceChannel.WEB: WebChannel(fetcher=fetcher),
    }


def main() -> None:
    spec = load_job_spec_from_str(SPEC_YAML)
    agent = build_agent()

    print("=" * 70)
    print("DISCOVERY (user supplied NO sources):")
    print("=" * 70)
    plan = agent.discover(spec)
    print(render_plan(plan))

    work_dir = Path(tempfile.mkdtemp(prefix="adh_demo_"))
    print("\n" + "=" * 70)
    print(f"ACQUISITION -> {work_dir}")
    print("=" * 70)
    result = run_engine_a(
        spec,
        work_dir=work_dir,
        agent=agent,
        channels=build_channels(),
        approver=AutoApprover(),  # autonomous for the demo; real runs prompt a human
    )
    print(f"exit_reason       : {result.exit_reason}")
    print(f"corpus chunks     : {result.n_chunks}")
    print(f"  by channel      : "
          f"{json.loads(result.summary_path.read_text())['chunks_by_channel']}")
    print(f"synthesis handoffs: {result.n_handoffs}")
    print(f"corpus tokens     : {sum(c.n_tokens for c in result.chunks)}")

    print("\nmanifest.jsonl (first rows):")
    for line in result.manifest_path.read_text().splitlines()[:4]:
        row = json.loads(line)
        print(f"  - [{row['kind']}] {row.get('source_channel')} "
              f"{row.get('source_url') or row.get('topic')}")

    lineage = json.loads(result.lineage_path.read_text())
    print(f"\nlineage.json: {lineage['counts']} exit={lineage['exit_reason']}")

    # ---- Engine B: corpus -> verified, deduplicated dataset -------------------
    print("\n" + "=" * 70)
    print("SYNTHESIS + CURATION (Engine B)")
    print("=" * 70)
    # Offline: inject the deterministic generator + jury (production uses the live
    # LlmGenerator / LlmJuryPanel picked by build_generator/build_judge).
    b = run_engine_b(
        spec,
        work_dir=work_dir,
        generator=HeuristicGenerator(),
        judge=HeuristicJudge(),
    )
    c = b.counts
    print(f"exit_reason       : {b.exit_reason}")
    print(f"records shipped    : {b.n_records}  "
          f"(from {c['generated']} generated, accept rate {c['accept_rate']:.1%})")
    print(f"  by split        : {b.delivery.counts_by_split}")
    print(f"  dropped         : quality={c['dropped_quality']} verifier={c['dropped_verifier']} "
          f"jury={c['dropped_jury']} dup={c['duplicates']} dead={c['dead_lettered']}")
    print(f"dataset ({b.delivery.format}) -> {b.dataset_dir}")
    print(f"  docs            : {', '.join(p.name for p in b.docs)}")

    shards = [f for f in b.delivery.files if f.suffix == ".jsonl"]
    if shards:
        first = shards[0].read_text().splitlines()
        if first:
            rec = json.loads(first[0])
            print("\nsample record:")
            print(f"  input : {rec['input'][:70]}")
            print(f"  output: {rec['output'][:70]}")
            print(f"  split={rec['split']} tier={rec['verifiability_tier']} "
                  f"judge={rec.get('judge_scores')} dedup={rec['dedup_cluster_id']}")


if __name__ == "__main__":
    main()
