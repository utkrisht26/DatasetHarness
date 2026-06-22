# Agentic Dataset Generation Harness

A system that turns a one-line data goal + a structured **Job Spec** into a **structured,
deduplicated, verified, training-ready dataset** with full provenance — at scale.

> **Core idea:** two engines on one shared spine.
> **Engine A (Discovery + Acquisition)** — the user gives a *goal, not sources*; the system discovers
> where the data lives (existing datasets > web crawl > pure synthesis) and acquires it into a
> provenance-tagged corpus, pausing for source approval by default.
> **Engine B (Synthesis & Curation)** turns that corpus into verified records.
> The **shared spine** handles orchestration, limits, storage, provenance, cost-routing,
> observability, and governance across both.

## Documents

| File | What it is |
|------|-----------|
| [`GOAL.md`](./GOAL.md) | The north star — what the system is *for*, the input→output promise, and ready-to-paste briefs for Claude. **Start here.** |
| [`PHASE0-CONTRACT.md`](./PHASE0-CONTRACT.md) | **LOCKED.** Verifiability tiers + Job Spec + Record Schema + provenance. The interface everything else is built against. |
| [`OUTLINE.md`](./OUTLINE.md) | Full phased project plan (0–4), reuse-first stance, build order, risk table. |
| [`RESEARCH.md`](./RESEARCH.md) | Cited evidence base for every design decision (4 research passes). |
| [`diagrams/architecture.drawio`](./diagrams/architecture.drawio) | Editable architecture diagram (open in draw.io / VS Code Draw.io extension). |
| [`diagrams/gen_architecture.py`](./diagrams/gen_architecture.py) | Reproducible generator for the diagram. |

## Status

- **Phase 0 — Contract:** ✅ locked (§0.8 open decisions resolved with the recommended options; see [`PHASE1.md`](./PHASE1.md))
- **Phase 1 — Engine A (Discovery + Acquisition):** ✅ implemented in [`src/adh/`](./src/adh) — discovery loop, approval gate, three channels, saturation stop, circuit-breaker floor, durable resume, provenance/manifest. Reuse-libraries plug in as adapters. See [`PHASE1.md`](./PHASE1.md).
- **Phases 2–4:** outlined

### Quickstart

```bash
uv sync
uv run pytest -q                                   # 97 tests, ~94% coverage
uv run python examples/demo_offline.py             # end-to-end Engine A, offline
uv run adh discover examples/rust_async_macros.yaml
uv run adh run examples/rust_async_macros.yaml --autonomous --no-fetch
```

## Key decisions baked in

- **Discovery-first (the core value):** sources are **never required** from the user. Given only a goal
  + record shape, the system discovers sources across three channels — **existing datasets** (reuse) >
  **web crawl** > **pure synthesis** — and shows a ranked list for **approval by default**. This is the
  intelligent differentiator, not a config field.
- **Reuse over build:** Crawl4AI (+ official MCP server) · deepagents/LangGraph · Distilabel (generate+judge)
  · datatrove (CPU dedup) · Argilla (HITL) · HF Hub/Kaggle APIs (dataset discovery). Net build surface ≈
  the discovery loop + orchestration glue + the contract + the tiered verifier + provenance.
- **Verifiability tier is a first-class job parameter** (hard / soft / hybrid) — it selects the
  verification sub-graph.
- **Coverage-saturation is the primary stop signal**, hard limits are the circuit-breaker floor.
- **Documentation is a byproduct:** Datasheet + Croissant + dataset card auto-emitted from the Job Spec.

## Regenerate the diagram

```bash
cd diagrams && python3 gen_architecture.py   # writes architecture.drawio
```

## Provenance of this design

Built on a prior architecture exploration (`../[Gemini Conversation] Building Agentic Dataset
Generation Harness.txt`), then re-grounded with current research and corrected for several stale
library APIs (see `RESEARCH.md` §APIs). The main addition over that exploration is **Engine B** — the
synthesis/curation half that turns a scraped corpus into an actual dataset.
