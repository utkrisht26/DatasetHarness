# Phase 1 — Engine A (Discovery + Acquisition) · implementation notes

> Status: **implemented** against the LOCKED Phase 0 contract. Runs and is fully
> tested offline (no network / no API keys); reuse-libraries plug in as adapters.

## What was built

A Python package (`src/adh/`) implementing the Phase 0 contract **as code** plus
Engine A and the minimal shared-spine slice it needs to be auditable.

```
adh/
├── contract/      Phase 0 (LOCKED): JobSpec, Record, enums, validation, YAML loader
├── llm/           reasoning port for discovery (HeuristicLLMProvider offline)
├── search/        web-search + dataset-registry ports (in-memory + HF Hub adapter)
├── discovery/     1.0 Source Discovery Agent (decompose→route→query→evaluate→rank) + approval gate
├── channels/      1.1/1.6/1.7 web (Crawl4AI) · datasets (HF) · synthesis channels
├── corpus/        1.3 content-addressed store + context-offload (URI + preview)
├── provenance/    3.3 hashing + PROV-style lineage.json export
├── orchestration/ 1.4 saturation + circuit-breaker · 1.5 durable state · Engine A driver
└── cli.py         adh validate | discover | run
```

### Design stance: build the glue, reuse the heavy lifting

Per OUTLINE "the build surface is mostly orchestration glue", the **core deps are
light** (`pydantic`, `pyyaml`, `jsonschema`). The reuse-libraries are **optional
extras**, imported lazily behind ports so the harness runs and is testable without
them, and swaps to the real tools when installed:

| Port (interface) | Offline default | Production adapter (extra) |
|---|---|---|
| `llm.LLMProvider` | `HeuristicLLMProvider` | deepagents/LangGraph loop (`adh[agents]`) |
| `search.SearchProvider` | `InMemorySearchProvider` | SearXNG/Brave/Crawl4AI `ask` |
| `search.DatasetRegistry` | `InMemoryDatasetRegistry` | `HfDatasetRegistry` (`adh[datasets]`) |
| `channels.web.ContentFetcher` | `InMemoryContentFetcher` | `Crawl4AIContentFetcher` (`adh[web]`) |
| `channels.datasets.DatasetLoader` | `InMemoryDatasetLoader` | `HfDatasetLoader` (`adh[datasets]`) |
| `corpus.CorpusStore` | `LocalCorpusStore` | S3/MinIO impl |

```bash
uv sync                      # core (offline-capable)
uv pip install -e '.[all]'   # + Crawl4AI, deepagents, huggingface_hub adapters
```

## Acceptance criteria (OUTLINE §Phase 1) → where met

| Criterion | Implementation | Test |
|---|---|---|
| Given only a goal (no sources), discover + rank | `SourceDiscoveryAgent.discover` | `test_discovery.py::test_reuse_first_prefers_datasets` |
| Reuse > crawl > synthesize routing | `agent._route_aspect` | `test_web_fallback_when_no_dataset`, `test_synthesis_when_no_external_source` |
| Approval gate (default ON) | `discovery/approval.py` + CLI prompt | `test_auto_approver_approves_all`, `test_cli.py::test_interactive_decision_*` |
| Fetch via the right channel → provenance corpus | `channels/` + `corpus/offload` | `test_channels.py`, `test_orchestration.py::test_engine_a_end_to_end` |
| Synthesis hand-off (no corpus) | `channels/synthesis.py` | `test_engine_a_synthesis_only` |
| Stop on saturation (primary signal) | `orchestration/saturation.py` | `test_engine_a_saturation_skips_remaining_web` |
| Never exceed limits (circuit-breaker floor) | `orchestration/limits.py` | `test_engine_a_circuit_breaker_floor` |
| Resume cleanly after a kill | `orchestration/state.py` | `test_engine_a_resume_skips_completed` |
| Modes: auto / seeded / fixed | `agent.discover` / `_fixed_plan` | `test_fixed_mode_wraps_seeds`, `test_seeded_mode_enriches_goal` |
| Coverage-gap re-discovery loop | `agent.rediscover` | `test_rediscover_merges_new_sources` |

Run the offline end-to-end demo: `uv run python examples/demo_offline.py`.

## Decisions (documented, no contract deviation)

- **Phase 0 §0.8 open decisions** — adopted the recommended options: `input`/`output`
  payload (not `messages[]`); per-record provenance columns **+** a `lineage.json`
  PROV export (no MLMD yet); `unknown`-license sources are **quarantined**
  (`license_ok=False`, heavily down-ranked, never auto-approved by `ThresholdApprover`).
- **MCP transport mismatch** (RESEARCH §APIs flagged it): Crawl4AI's official server is
  **SSE/WS**, while `langchain-mcp-adapters` prefers `streamable_http`. **Decision for
  Phase 1:** the `web` channel talks to Crawl4AI **directly** via `AsyncWebCrawler`
  (`Crawl4AIContentFetcher`) — no MCP needed for the crawl itself. MCP is reserved for
  the deepagents *tool-loading* path; when that is wired, point
  `MultiServerMCPClient` at `transport: "sse"` / `/mcp/sse`, or front the service with
  a streamable-http shim. This avoids the mismatch on the hot path.
- **Circuit-breaker semantics:** the pre-fetch gate trips at `>=` the cap (a *floor*:
  the run never goes over, and a cap of `0` means "do not start"). Coverage-saturation
  remains the **primary** stop; limits are the safety floor.
- **Saturation** is a statistical approximation of Crawl4AI `AdaptiveCrawler`
  (`confidence = 1 − recent mean novelty`, warm-up `min_chunks`, threshold `0.7`);
  production can delegate to the real adaptive crawler.

## Deferred (out of Phase 1)

Engine B (generation + tiered verification + dedup/decontam), the deepagents/LangGraph
backend wiring, live token/cost metering, and S3/MinIO store — all have ports/seams
ready. Phase 0 §0.7 deferrals (multimodal, GitHub/API channels, streaming,
multi-tenant) remain deferred.
