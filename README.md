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
| [`PHASE1.md`](./PHASE1.md) | Engine A implementation notes: what was built, the port/adapter seams, and acceptance criteria → tests. |
| [`RESEARCH.md`](./RESEARCH.md) | Cited evidence base for every design decision (4 research passes). |
| [`DESIGN.md`](./DESIGN.md) | The ADH Console design system — aesthetic, tokens, components, and screen patterns for the operator frontend. |
| [`docs/design/`](./docs/design) | Platform + frontend plans and the Crawl4AI integration design doc. |
| [`diagrams/architecture.drawio`](./diagrams/architecture.drawio) | Editable architecture diagram (open in draw.io / VS Code Draw.io extension). |
| [`diagrams/gen_architecture.py`](./diagrams/gen_architecture.py) | Reproducible generator for the diagram. |

## Status

- **Phase 0 — Contract:** ✅ locked (§0.8 open decisions resolved with the recommended options; see [`PHASE1.md`](./PHASE1.md))
- **Phase 1 — Engine A (Discovery + Acquisition):** ✅ implemented in [`src/adh/`](./src/adh) — discovery loop, approval gate, three channels, saturation stop, circuit-breaker floor, durable resume, provenance/manifest. Reuse-libraries plug in as adapters. See [`PHASE1.md`](./PHASE1.md).
- **Servicified + live discovery:** ✅ Engine A now runs as an HTTP gateway (`adh serve`) behind the approval gate, backed by a durable [`RunStore`](./src/adh/orchestration/run_store.py). Live **deepagents/LangGraph discovery** ([`DeepAgentDiscoveryAgent`](./src/adh/discovery/deep_agent.py) with heuristic fallback), a real **Crawl4AI** web channel, and **live token/cost metering** ([`UsageTracker`](./src/adh/llm/usage.py)) are wired — all previously deferred from Phase 1.
- **Phases 2–4 (Engine B):** outlined

### Quickstart

```bash
uv sync
uv run pytest -q                                   # 270 tests
uv run python examples/demo_offline.py             # end-to-end Engine A + Engine B, offline
uv run adh discover examples/rust_async_macros.yaml
uv run adh run examples/rust_async_macros.yaml --autonomous --no-fetch
uv run adh serve -v                                # gateway (+ verbose step trace)
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

## The service surface

`adh serve` builds one thin gateway app ([`service/app.py`](./src/adh/service/app.py)) — the
frontend talks to a single origin; each domain lives under its own prefix so lifting one to
its own process later is a config flip, not an API change.

| Route | What it does |
|-------|--------------|
| `POST /api/runs` | Submit a Job Spec → discovery on a worker → parks at `AWAITING_APPROVAL` (autonomous specs auto-approve inline). |
| `POST /api/runs/{id}/approve` · `/cancel` | Drive the run state machine over the durable [`RunStore`](./src/adh/orchestration/run_store.py). |
| `GET /api/runs` · `/api/runs/{id}` | List / inspect runs, ranked plan, and per-run metrics (pages, tokens, cost). |
| `/api/crawl/*` | The standalone crawl microservice (submit/monitor/inspect extracted pages) over the Crawl4AI web channel. |
| `/api/healthz` | Liveness + crawl job counts. |

Discovery spend is metered per-run: the agent factory is fed a [`UsageTracker`](./src/adh/llm/usage.py)
so every deepagents model call folds into that run's metrics. Everything is injectable for
offline tests — the defaults wire the live stack when the extras are installed. The
operator console that consumes this surface is specced in [`DESIGN.md`](./DESIGN.md) and
[`docs/design/`](./docs/design).

```bash
uv pip install -e '.[serve]'   # + FastAPI/uvicorn over the web channel
uv pip install -e '.[all]'     # + Crawl4AI, deepagents, huggingface_hub adapters
```

## Observability

Every entry point installs a single stderr log handler on the `adh` namespace
([`logconfig.py`](./src/adh/logconfig.py)); the library itself never touches the root
logger, so importing `adh` as a dependency leaves a host's logging untouched.

```bash
uv run adh serve            # INFO: run lifecycle + deep-agent tool calls / model turns
uv run adh serve -v         # DEBUG: also tool results and per-page crawl decisions
uv run adh run … -q         # WARNING and above only
ADH_LOG_LEVEL=debug uv run adh …   # or set the level via the environment
```

The deepagents discovery loop is fully traced — no more silent 40k-token black box.
Each run emits, per model call, a running token total; each tool call and subagent
step; and, on failure, the **full traceback plus the tokens spent at the point of
failure** (then it degrades to the heuristic agent rather than killing the run):

```text
deep-agent starting: job=rust-async goal='Rust async macros…' channels=datasets,web,synthesis recursion_limit=50
deep-agent · model turn 1 (+8200 tok, 8200 total)
deep-agent → tool web_search({'query': 'rust async runtime internals'})
deep-agent · model turn 2 (+8200 tok, 16400 total)
deep-agent finished: 1 aspect(s), 1 source(s) — 3 turn(s), 2 tool call(s), 25800 token(s)
```

## Regenerate the diagram

```bash
cd diagrams && python3 gen_architecture.py   # writes architecture.drawio
```

## Provenance of this design

Built on a prior architecture exploration (`../[Gemini Conversation] Building Agentic Dataset
Generation Harness.txt`), then re-grounded with current research and corrected for several stale
library APIs (see `RESEARCH.md` §APIs). The main addition over that exploration is **Engine B** — the
synthesis/curation half that turns a scraped corpus into an actual dataset.
