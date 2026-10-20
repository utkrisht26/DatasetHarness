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
| [`docs/memory.md`](./docs/memory.md) | Current status handoff: what is done, what remains (prioritised), live-run results, and working conventions. **Read before picking up work.** |
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
- **Phases 2–4 (Engine B):** ✅ implemented — `adh build` (seeding → generation → tiered
  verification → PII/dedup/decontamination → JSONL/Parquet + datasheet + Croissant + quality
  report), `adh loop` (coverage-gap re-discovery), and `adh publish` (quality-gated Hub upload).
- **Live runs:** end-to-end text datasets have been produced on a live model; open quality items
  (self-judging, license coverage, scale) are tracked in [`docs/memory.md`](./docs/memory.md).

### Quickstart

```bash
# The test suite imports the optional extras CI installs (all except evals/webvideo):
uv sync --extra web --extra datasets --extra media --extra agents --extra search \
        --extra verify --extra dedup --extra deliver --extra serve --extra publish --extra pii
cp .env.example .env                              # optional deployment settings
uv run pytest -q                                   # 800+ tests, offline, ~1 min
uv run python examples/demo_offline.py             # end-to-end Engine A + Engine B, offline
uv run adh discover examples/rust_async_macros.yaml
uv run adh run examples/rust_async_macros.yaml --autonomous --no-fetch
uv run adh serve -v                                # gateway (+ verbose step trace)
```

### Configuration

Application settings are defined and validated in [`src/adh/settings.py`](./src/adh/settings.py).
Copy [`.env.example`](./.env.example) to `.env` for local configuration; `.env` is intentionally
gitignored and is loaded only by the CLI, so importing `adh` never injects secrets or changes the
process environment.

The main settings are:

| Variable | Purpose |
|----------|---------|
| `ADH_DISCOVERY_MODEL` | Default deep-discovery model; `--discovery-model` wins. |
| `ADH_LLM_BASE_URL`, `ADH_LLM_API_KEY` | Optional OpenAI-compatible model gateway. |
| `ADH_GENERATOR_MODELS`, `ADH_JUDGE_MODELS` | Comma-separated Engine B model overrides. |
| `TAVILY_API_KEY`, `PEXELS_API_KEY`, `OPENVERSE_API_KEY` | Discovery/media providers. |
| `ADH_API_KEY`, `ADH_CORS_ORIGINS` | Gateway authentication and allowed browser origins. |
| `ADH_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `ADH_PII_ENTITIES` | Comma-separated Presidio entity types to redact; default is direct identifiers only (email, phone, card, IBAN, IP, SSN…). Add `PERSON` for datasets about people. |

Provider-native credentials such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` continue to be read
by their SDKs. Job-specific settings—including discovery policy, model ladders, limits, schemas,
and delivery—remain in the Job Spec YAML.

Engine B validates the saved spec before starting generation. `record_schema_ref` must point
to a local JSON Schema file; a vocabulary URL such as `http://schema.org/Text` is not supported.
Tier 1 requires an `x-verifier` hook in that schema. Descriptive text input/output records can
use `./schemas/qa_record.schema.json`, Tier 2, and a `judge_config` with models from at least
two distinct families. These settings determine verification guarantees; builds never switch
tiers automatically. Invalid build configurations return HTTP 422 before queuing a worker.
Rejected candidates are logged with their reason, and a bounded sample of distinct reasons
is retained in `engine_b_summary.json` notes.

After correcting a spec for a build that already exhausted its seeds, use
`adh build examples/corrected-spec.yaml --work-dir runs/<run-id> --fresh` to regenerate from
the existing corpus. This resets Engine B's generation checkpoints; it does not repeat discovery
or crawling. Back up any previous dataset output before rebuilding it. A rebuild removes split
files the previous build wrote but the new one does not (e.g. a stale `validation.parquet`), so
the dataset folder always matches its manifest.

#### Live models

- **Reasoning models** spend part of `max_tokens` thinking before they answer. Set
  `generation_params.max_tokens` generously in the spec (16384 worked for MiniMax M3.1 Flash).
  A reply cut off at the limit is retried once with 8× the budget, capped at 32,768 tokens.
- **Every model request times out after 600 s**, so a stalled connection cannot hang a build.
  Pass `timeout=` to `build_chat_model` to override.
- **Records are standalone.** The generator writes records that never mention their source
  material, and returns a skip for grounding chunks with nothing to teach (navigation pages,
  job listings); the quality filter rejects records that refer to "the provided context".
- Run long builds with an absolute `--work-dir`. Relative spec paths (`record_schema_ref`,
  `rubric_ref`, `decontaminate_against`) resolve against the spec file's directory.

### Development checks

Install the Git hook once after cloning, then run the complete check suite whenever needed:

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
```

Pre-commit fixes trailing whitespace, missing final newlines, UTF-8 byte-order marks, and mixed
line endings. It also validates common configuration formats and runs Ruff lint fixes followed by
Ruff formatting for Python files.

The test suite is hermetic: [`tests/conftest.py`](./tests/conftest.py) disables `.env` loading,
removes `ADH_*` and provider-key variables, and replaces live search/registry ports with
in-memory stubs, so tests never make live or billed calls. Tests that need configuration set it
with `monkeypatch.setenv`.

CI ([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)) runs the pre-commit hooks and the
full test suite on every pull request and on pushes to `main`. It installs every extra except
`evals` and `webvideo`, plus spaCy's `en_core_web_lg` model for Presidio.

## Media datasets (video & images — open-internet acquisition)

Specs with `modality: "video"` or `modality: "image"` acquire **real media assets** instead of
text about the domain — Engine A routes them exclusively to the **MEDIA channel**, never falling
back to crawling pages *about* the topic. Assets are staged content-addressed
(`media_staging/<sha256>.<ext>` + metadata sidecar) under the spec's `limits.max_storage_mb`
budget, filtered by type/duration/orientation constraints and license, then packaged into records
carrying a `MediaRef` (sha256, bytes, duration, resolution, attribution). Captions v1 come from
**source metadata** (title/tags/alt); vision-model frame captioning is a designed-in upgrade seam.

### Providers — one port, many backends

Whatever this host has composes into one `CompositeMediaProvider` (priority order; every absence
is loud on stderr, none fatal while another backend serves):

| Backend | Media types | Needs | License posture |
|---------|-------------|-------|-----------------|
| [Openverse](https://api.openverse.org/) | image, gif | nothing (keyless; optional `OPENVERSE_API_KEY`) | CC / public-domain — known |
| Direct-web harvest | image, gif, video | live search port (`TAVILY_API_KEY` recommended) | **unknown**, recorded honestly |
| [Pexels](https://www.pexels.com/api/) | video | `PEXELS_API_KEY` | Pexels-License |
| yt-dlp | video | `webvideo` extra (`uv sync --extra webvideo`) | site terms → `"unknown"` unless recognized |

### License ladder (two-layer policy)

1. **Acquisition defaults to known-free** — hits whose license is incompatible with the spec's
   `output_license` are skipped; `"unknown"` is compatible with nothing.
2. **Opt-in to the unverified web** — `discovery.allow_unknown_licenses: true` admits
   unknown-license assets, always *recorded as* `"unknown"`, never invented.
3. **Distribution is a separate consent** — `adh publish` **refuses public uploads while
   unknown-license assets are present** (private/unlisted/dry-run unaffected); explicit override
   with `--allow-unknown-licenses`.

Direct-web fetching is polite and guarded by construction
([`adh.net.httpfetch`](./src/adh/net/httpfetch.py)): per-domain rate limiting, robots.txt honoring
(crawl-delay capped at 10s), and SSRF defenses (scheme allow-list, private-range blocks,
per-hop DNS repinning, post-connect peer re-check). DRM boundaries are respected by design:
yt-dlp's default refusal of protected streams **is** the boundary and its bypass flags are never
set — if it plays in an open player we can fetch it; Netflix-class DRM stays out.

Delivery picks a trainer layout via `delivery_layout`: `"hf"` (default; `media/<sha256>.<ext>` +
parquet/jsonl — sensible for images) or `"ltx"` (LTX-Video trainer: `videos/NNNNN_<sha8>.mp4` +
aligned `metadata.jsonl` + `prompts.txt` — sensible for video). Custom layouts register through
[`delivery/layouts.py`](./src/adh/delivery/layouts.py). Manifest rows of `kind:"asset"` flow into
publish uploads, Croissant distributions, and quality stats automatically.

Live recipes:

```bash
# Keyless open-web images — no API key required (examples/openverse_images.yaml):
uv sync --extra web --extra agents --extra deliver
uv run adh run examples/openverse_images.yaml --yes
uv run adh publish examples/openverse_images.yaml --repo-id <you>/images --dry-run

# Stock video clips (examples/dance_i2v_ltx.yaml):
export PEXELS_API_KEY=<free key from https://www.pexels.com/api/>
uv sync --extra web --extra media --extra webvideo --extra agents --extra deliver
uv run adh run examples/dance_i2v_ltx.yaml --yes
```

With no backend at all the MEDIA channel reports itself unavailable loudly on stderr — media jobs
surface an explicit coverage-gap note rather than silently degrading to a text crawl.

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
- **Junk sources are filtered before acquisition:** job boards, crypto price trackers,
  stock-ticker news wires, company-profile databases, and spam
  ([`junk.py`](./src/adh/junk.py)). A known junk domain is decisive; otherwise page text needs a
  strong signal ("apply now", "circulating supply"), and vocabulary shared with technical writing
  (a `/jobs/` path, "stock market") only corroborates. A category is exempt when the goal is
  about it. Every skip is recorded in the plan notes as `skip <url>: junk source (<reason>)`.
- **Coverage-saturation is the primary stop signal**, hard limits are the circuit-breaker floor.
- **Documentation is a byproduct:** Datasheet + Croissant + dataset card auto-emitted from the Job Spec.

## The service surface

`adh serve` builds one thin gateway app ([`service/app.py`](./src/adh/service/app.py)) — the
frontend talks to a single origin; each domain lives under its own prefix so lifting one to
its own process later is a config flip, not an API change.

| Route | What it does |
|-------|--------------|
| `GET /api/specs/schema` | Job Spec JSON Schema, editable defaults, bundled record profiles, and deployment model overrides. |
| `POST /api/specs/validate` | Validate `{ "spec": {...} }` against the contract and local schema/verification requirements; returns field errors, warnings, and normalized YAML. |
| `POST /api/specs/import` | Parse `{ "yaml_text": "..." }` safely; retain an invalid draft for repair and export only valid specs. |
| `POST /api/runs` | Submit a Job Spec → discovery on a worker → parks at `AWAITING_APPROVAL` (autonomous specs auto-approve inline). |
| `POST /api/runs/{id}/approve` · `/cancel` | Drive the run state machine over the durable [`RunStore`](./src/adh/orchestration/run_store.py). |
| `GET /api/runs` · `/api/runs/{id}` | List / inspect runs, ranked plan, and per-run metrics (pages, tokens, cost). |
| `GET /api/runs/{id}/events` | Read the ordered execution trace; poll with `?after=<sequence>&limit=200`. |
| `/api/crawl/*` | The standalone crawl microservice (submit/monitor/inspect extracted pages) over the Crawl4AI web channel. |
| `/api/healthz` | Liveness + crawl job counts. |

Discovery spend is metered per-run: the agent factory is fed a [`UsageTracker`](./src/adh/llm/usage.py)
so every deepagents model call folds into that run's metrics. Everything is injectable for
offline tests — the defaults wire the live stack when the extras are installed. The
operator console that consumes this surface is specced in [`DESIGN.md`](./DESIGN.md) and
[`docs/design/`](./docs/design).

### Job Spec Builder

Open `/spec` in the console from `../adh-frontend`. The same editor powers
`/discovery/new` and `/runs/{id}/edit`. It exposes the full Job Spec: goal and record
shape, verification, source constraints, licenses, generation, delivery, and budgets.
Client validation uses the server's JSON Schema; server validation checks cross-field
rules and referenced files before a run can be submitted from the builder.

Start with the text, code, image, or video record profile, or import YAML/JSON.
Text defaults use the bundled QA schema and Tier 2 verification. Tier 1 requires
a schema with an executable verifier. Schema and evaluation-reference paths resolve
on the server, not on the browser's filesystem. The form also shows active deployment
model overrides, which take precedence over the models saved in a spec.

Use **Validate & review**, review any warnings, then **Start discovery**. Validation
does not call models or start discovery. **Export YAML** downloads the validated,
normalized spec. **Save draft** / **Restore draft** use browser storage; editing and
restarting preserves the saved settings and creates a new run.

The browser acceptance tests use the actual gateway with offline discovery and crawl
doubles. After building `../adh-frontend`, run:

```bash
ADH_FRONTEND_DIST=../adh-frontend/dist .venv/bin/pytest tests/test_spec_builder_browser.py -o addopts='' -q
```

These tests require Playwright Chromium. Without `ADH_FRONTEND_DIST`, they are skipped;
the validation API tests in `tests/test_specs_api.py` always run.

API authentication is enabled when `ADH_API_KEY` is set. Send the value as a bearer token on
every `/api/*` request; `/api/healthz` remains public for liveness probes. The CLI refuses to bind
to a non-loopback address without a key unless the operator explicitly accepts the risk with
`--allow-unauthenticated`:

```bash
export ADH_API_KEY="$(openssl rand -hex 32)"
uv run adh serve --host 0.0.0.0
curl -H "Authorization: Bearer $ADH_API_KEY" http://127.0.0.1:8000/api/runs
```

This shared-key perimeter is suitable for a single trusted operator. Multi-tenant deployments
still require an identity provider plus per-run ownership and authorization checks.

```bash
uv pip install -e '.[serve]'   # + FastAPI/uvicorn over the web channel
uv pip install -e '.[all]'     # + Crawl4AI, deepagents, huggingface_hub adapters
.venv/bin/python -m playwright install chromium  # browser used by Crawl4AI
```

Install Chromium once before live web acquisition, and rerun that command after
upgrading Playwright. Installing the Python packages alone does not install the
matching browser executable. A `BrowserType.launch: Executable doesn't exist`
error means acquisition cannot open pages, even when discovery found sources.

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
failure** (then it degrades to the heuristic agent rather than killing the run).
Discovery follows a strict sequence: create and persist a search plan, execute only
the enabled channel tools, save the ranked fetch plan, then acquire/crawl. The search
plan is available at `<runs-dir>/<run-id>/research/search_plan.json`; if deep discovery
fails after planning, the heuristic recovery plan is saved separately as
`search_plan_fallback.json`. A disabled channel is omitted from the agent's tool set,
so disabling `datasets` prevents dataset registry calls instead of merely filtering
their results afterward.

The fetch plan pauses at `AWAITING_APPROVAL` before acquisition when
`discovery.approval_gate: true`. Set it to `false` only when the run should auto-approve
the completed fetch plan and start crawling immediately.

The same information is persisted as redacted JSON Lines in
`<runs-dir>/<run-id>/telemetry.jsonl`; query it while a run is active without reading
server logs:

```bash
curl "http://127.0.0.1:8000/api/runs/<run-id>/events?after=0&limit=200"
```

Discovery also records complete model responses for parsing diagnostics:
`model.response.raw` contains the OpenAI-compatible HTTP response before SDK/tool
parsing, and `model.response.parsed` contains the full LangChain result, including
`tool_calls`, `invalid_tool_calls`, content, finish reasons, and usage metadata.
Match their `operation_id` to `model.started`/`model.completed` to compare the
provider's `function.arguments` string with the parsed arguments. These response
events bypass ordinary telemetry size caps while retaining credential redaction,
and the same full payloads appear in DEBUG server logs (`adh serve -v`). Capture
starts on new model calls; previous runs cannot be reconstructed from token counts.
Raw HTTP capture currently covers non-streaming OpenAI-compatible SDK clients.
Streaming responses record an explicit raw-capture skip and retain the assembled
parsed response; other adapters retain the parsed response.

Every event has a sequence, timestamp, phase, operation/parent IDs, status, duration,
and bounded attributes. Common credential fields and bearer/API-key values are redacted
before they reach disk. The no-progress guard evaluates completed research batches,
so a parallel batch with any successful search counts as progress regardless of
completion order. Three consecutive unproductive batches trigger fallback; if
nonempty result files remain unread, a `guard.review_pending` event grants one
additional window of up to three unproductive batches to review them first.
Reading a new nonempty results file resets the counter; rereading an already
reviewed file does not. `research.batch.completed` records each batch's progress.
The agent receives a prioritized list of unread, nonempty files before each model
turn. Empty files need not be read, and unchanged zero-result queries are skipped
without another provider request. Search plans must contain at least one aspect.
Final reports must explicitly supply both `aspects` and `sources`; a missing
source list is a validation error, not an empty discovery result. Topic labels
are limited to 500 characters to reject reports accidentally embedded in a
string. `report.rejected` records validation failures. The model gets two
correction retries in the same research conversation before heuristic fallback.
An explicit `sources: []` is accepted only after at least one completed search in
each allowed external channel, review of nonempty results, and an explanation in
`notes`. Blocked/failed calls and plan submission do not count as searches. A
report routing every aspect to allowed synthesis may omit external research.
Premature empty reports receive concrete search/review feedback in the same
conversation, sharing the two-retry limit with schema errors. `report.rejected`
distinguishes these completion failures from schema errors with its `kind` field.
The stderr view remains concise:

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
