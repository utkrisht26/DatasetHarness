# Project memory: current status and handoff

> **Read this first** if you are picking the project up on a new machine, a new assistant session,
> or as a new contributor. It records what is done, what is open, and exactly where work stopped.
> Update the "Last updated" line and the relevant sections whenever you finish a chunk of work.

**Last updated:** 2026-10-20
**Active branch:** `fix/live-run-quality` (pushed to `origin`; **no PR opened yet, not merged** into `main`)
**Remote:** `github.com/utkrisht26/DatasetHarness`
**Frontend:** separate repo at `../adh-frontend` (React + Vite operator console, last commit "complete phase 2 operator console")

---

## 1. What this project is

ADH (Agentic Dataset Harness) turns a plain-English description of a dataset into a verified,
deduplicated, provenance-tracked, training-ready dataset, **without the requester naming sources**.
See [GOAL.md](../GOAL.md) for the north star, [OUTLINE.md](../OUTLINE.md) for the full design, and
[docs/design/adr/](design/adr/) for the architecture decisions.

It has two engines:

| Engine | Responsibility | Main code |
|---|---|---|
| **A: discovery + acquisition** | Decompose the goal into aspects, find sources (heuristic or deepagents LLM discovery, Tavily search, HF registry), approval gate, crawl (Crawl4AI), media providers, saturation governor, circuit breaker | `src/adh/discovery`, `src/adh/channels`, `src/adh/search`, `src/adh/media`, `src/adh/orchestration/engine_a.py` |
| **B: synthesis + curation + delivery** | Seed engine, then LLM generator, then verification router (quality filter, then deterministic verifier and/or LLM jury), then curation (PII, dedup, decontamination), then write shards + manifest + datasheet + Croissant + quality report | `src/adh/generation`, `src/adh/verification`, `src/adh/curation`, `src/adh/delivery`, `src/adh/orchestration/engine_b.py` |

Other surfaces: the CLI (`src/adh/cli.py`), an HTTP gateway (`adh serve`: `/api/crawl/*` and `/api/runs/*`,
`src/adh/service`), a coverage-gap loop (`adh loop`), and Hub publishing with a quality gate (`adh publish`).

### CLI

```bash
adh validate <spec.yaml>                         # validate a Job Spec
adh discover <spec.yaml> [--no-llm]              # print the source plan
adh run <spec.yaml> --work-dir <ABS path> --yes [--no-llm] [--fresh]   # Engine A
adh build <spec.yaml> --work-dir <ABS path> [--fresh]                  # Engine B -> <work-dir>/dataset/
adh loop <spec.yaml> ...                         # build, re-discover thin aspects, re-acquire, rebuild
adh publish <spec.yaml> --repo-id org/name [--public] [--dry-run]
adh serve --runs-dir ./runs                      # API for the frontend
```

---

## 2. Status: what is DONE

### Core pipeline (all implemented and tested)
- Job Spec contract + validation (`src/adh/contract`)
- Discovery: heuristic (offline, no model calls) and deepagents LLM discovery; HF registry search; Tavily search
- Acquisition channels: web crawl (Crawl4AI), synthesis, **media** (Openverse images (keyless), Pexels, direct-web harvest, yt-dlp); polite fetching (robots.txt, per-domain rate limits, SSRF re-validation per redirect hop)
- Saturation governor, circuit breaker, approval gate, resumable state for both engines
- Generator with routing ladder (cheap to frontier), per-tier retries, a corrective re-ask for unparseable replies, and a **truncation retry** for reasoning models
- Verification: quality filter (incl. "standalone" check), tiered verifiability (Tier 1 deterministic verifier / sandboxed code verifier, Tier 2 jury, Tier 3 hybrid), PoLL jury with position swap
- Curation: Presidio PII redaction (fail-closed if the extra is missing), MinHash dedup, decontamination against held-out eval sets
- Delivery: JSONL/Parquet shards, manifest with sha256, datasheet, Croissant, HF dataset card, quality report, media layouts (`hf`, `ltx`)
- Two-layer license policy: unknown-license assets may be **acquired**, but a **public** publish is refused while any remain
- Settings registry with env overrides (`src/adh/settings.py`); model endpoint routing by configuration (`ADH_LLM_BASE_URL`)
- HTTP gateway + file-based run store; API-key auth and CORS config
- Frontend operator console (separate repo): submit, approve, monitor, restart / edit-restart, dataset panel

### Tests / quality
- ~690 test functions under `tests/`; full suite green (`uv run pytest -q`, ~1 min), ruff check + ruff format clean, pre-commit hooks installed
- The suite is **hermetic**: an autouse fixture in `tests/conftest.py` disables `.env` loading, strips `ADH_*` and provider-key variables, and swaps live discovery ports (Tavily / HF Hub) for in-memory stubs. Tests that need configuration set it with `monkeypatch.setenv`.
- **CI:** `.github/workflows/ci.yml` runs pre-commit (ruff + file hygiene) and the full test suite on every PR and on pushes to `main`. It installs every extra except `evals` and `webvideo`, plus spaCy's `en_core_web_lg` for Presidio.

### Live end-to-end runs that worked
| Run | Result |
|---|---|
| `runs/live-gate-2` | Engine A live gate: 123 web chunks, 29/29 sources approved |
| `runs/Image-to-video-dance-dataset-b6b176e6` | First live Engine B dataset: 13 records (Aug 2026) |
| `runs/rust-async-live-small` | **Latest (2026-10-20):** 5 clean standalone records (train 4 / test 1), 0 PII markers, junk sources skipped. Funnel: 8 generated, 1 dropped by the quality filter, 2 rejected by the jury, 4 seeds skipped as junk grounding |

`runs/` is gitignored, so these exist only on the original machine.

### Latest branch: `fix/live-run-quality` (on top of `8cd69d1`)
All found by the live run, all fixed test-first:

| Commit | Fix |
|---|---|
| `c1a9238` | Engine A page budget is fair-shared across sources (`_fair_share` in `engine_a.py`); one deep crawl used to eat the whole budget and overshoot `max_pages_scraped` |
| `293f1a0` | PII default = **direct identifiers only** (`DEFAULT_PII_ENTITIES` in `curation/pii.py`); generic NER was redacting "Rust", "Tokio", URLs (23/23 false positives). `ADH_PII_ENTITIES=PERSON,...` re-enables extra entities; unknown entity names fail fast |
| `6d89eae` | Prompts demand **standalone** records and allow `{"skip": reason}` for junk grounding (job listings, index pages); `QualityFilter.check_standalone` rejects "the provided context..."-style records before the jury |
| `49adc6d` | Any `finish_reason` = `length`/`max_tokens` (empty **or** partial JSON) retries once with 8x `max_tokens`, capped at 32,768 |
| `a7be0f4` | Every model request has a default 600 s timeout (`DEFAULT_REQUEST_TIMEOUT_S` in `llm/models.py`); a build once hung 38 min on a stalled socket |
| `efe0c5c` | `write_dataset` removes shards that the previous manifest listed but the new build doesn't write (a stale `validation.parquet` was being picked up by the HF loader); never deletes outside the dataset dir |
| `2f39028` + follow-up | Goal-aware junk source filtering ([src/adh/junk.py](../src/adh/junk.py)): job boards, crypto trackers, stock-ticker news, company directories, spam. A known junk domain is decisive; otherwise path/text **strong** signals ("apply now", "circulating supply", "(NASDAQ: X)") are needed and **weak** ones ("/jobs/" path, "8 jobs", "stock market") only corroborate; job pages need two signals. A category is exempt when the goal is about it (spam never). Applied to web search hits in heuristic and deep discovery (each skip is a plan note) and to crawler outlinks (known domains, `jobs.`/`careers.` hosts, `/careers/`; pruned as "junk source"). Regression tests pin the live-run junk and six technical look-alikes (Kubernetes Jobs, GitLab CI jobs, ssh-keygen, ...). Ranking tie-break on assessment; license penalty stays 0.5 |

---

## 3. Environment / configuration

Copy `.env.example` to `.env` and fill it in. Key variables (never commit values):

| Variable | Purpose |
|---|---|
| `ADH_LLM_BASE_URL`, `ADH_LLM_API_KEY` | Route all model calls through an OpenAI-compatible gateway (currently OpenRouter: `https://openrouter.ai/api/v1`) |
| `ADH_DISCOVERY_MODEL`, `ADH_GENERATOR_MODELS`, `ADH_JUDGE_MODELS` | Override the models named in specs (comma-separated lists for the latter two) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Provider keys (an OpenRouter `sk-or-...` key works as `OPENAI_API_KEY`) |
| `TAVILY_API_KEY` | Web search for discovery |
| `PEXELS_API_KEY`, `OPENVERSE_API_KEY` | Media providers (Openverse works keyless) |
| `ADH_PII_ENTITIES` | Extra Presidio entities, e.g. `PERSON` |
| `ADH_API_KEY`, `ADH_CORS_ORIGINS` | Gateway auth / CORS |

**Current model setup (as of 2026-10-20):** discovery, generator and judge are all
`openai:minimax/minimax-m3.1-flash:free` on an **OpenRouter free-tier key** (nominally 50 requests/day; the cap did not
bite in practice). It is a **reasoning model** with highly variable thinking length (3k to 20k+ tokens per record), so
specs must set `generation_params.max_tokens: 16384`. It is slow: about 3 to 4 minutes per seed.

### Practical gotchas
- Spec file paths (`record_schema_ref`, `rubric_ref`, `decontaminate_against`) resolve **relative to the spec file's directory**; use absolute paths or paths relative to the spec.
- Always pass an **absolute** `--work-dir` when running builds in the background.
- Heuristic discovery (`--no-llm`) splits `description`/`motivation` into aspects **literally**, so write them as short topic phrases ("Rust Pin API for futures. Tokio task scheduling internals."), not prose.
- Full deepagents discovery costs ~80 model calls; use `--no-llm` discovery when quota is tight.
- To reproduce the latest live run, copy `examples/rust_async_macros.yaml` and change: `name: rust-async-live-small`, `target_size.records: 12`, `discovery.channels: [web, synthesis]`, `limits.max_pages_scraped: 40`, `generation_params.max_tokens: 16384`, `output_format: parquet`, plus the topic-phrase `description` above. Then:
  ```bash
  adh run  spec.yaml --work-dir "$PWD/runs/rust-async-live-small" --yes --no-llm
  adh build spec.yaml --work-dir "$PWD/runs/rust-async-live-small" --fresh
  ```

---

## 4. What REMAINS (prioritised)

### Immediate (where work stopped)
1. **Open a PR** for `fix/live-run-quality` into `main` (pushed, PR not yet created): https://github.com/utkrisht26/DatasetHarness/pull/new/fix/live-run-quality
2. **Confirm the first CI run is green** on the PR. The workflow was written and its lint step checked locally, but the clean-machine install is only exercised on GitHub.

### Dataset quality
3. **Self-judging:** the generator and the judge are the same model family. The jury once scored a non-answer 5/5/5, and accepted records have shown helpfulness 2.5 carried by clarity 5. Add a judge from a **different model family** to `ADH_JUDGE_MODELS`, and consider a per-dimension floor (not only the mean ≥ `min_score`).
4. **License coverage is 0%** for web records (all `unknown`), so a public publish is refused. Needs license detection for crawled pages (page metadata / footer / known-domain table) or an allowlist of known-licensed sources.
5. ~~**Junk sources** (job boards, crypto sites)~~ — **DONE** (`2f39028` + follow-up, see §2). Not yet validated on a live run; watch the plan notes (`skip <url>: junk source (...)`) for wrongly skipped pages. Known gap: `adh.channels.base` imports `adh.discovery.models`, so the crawler still depends on the discovery package (pre-existing).
6. **Heuristic aspect decomposition** takes goal text literally (see the gotchas above); it should extract topics, not sentences.
7. **Judge 429s look like rejections:** a rate-limited judge abstains with score 0, which is counted as a jury rejection. It should surface as an outage (retry / pause), not a quality drop.

### Scale / ops
8. **`cost_usd` is always null** for OpenRouter responses, so `limits.max_cost_usd` can never trip. Read cost from the provider's usage payload or a price table.
9. **Bigger run:** 5 records proves the pipeline; a real training set needs a larger `target_size` (and a paid or faster model). Partial-success builds cannot be topped up by re-running `build` (resume skips completed seeds); use `adh loop`.
10. **No frontend tests**, and the frontend repo has no CI.
11. **Media:** VLM captioning for video/image records (captions currently come from source metadata); a live Pexels run needs `PEXELS_API_KEY`.
12. Publishing has **no HTTP route** by design (uploads need an auth layer); CLI only.

---

## 5. Working conventions
- **Quality bar (owner directive):** production-grade, industry-standard solutions, no MVP shortcuts; get a live run working first, then fill the gaps.
- **TDD:** write a failing test first, then fix; every live-run defect gets a regression test that names the run it came from.
- **Git:** conventional commits (`fix:`, `feat:`, ...), work on a branch, never commit directly to `main`. Run lint + tests before committing (pre-commit runs ruff).
- **Secrets:** only in `.env` (gitignored). Never print or commit key values.
- Long live builds: run in the background and watch the log for HTTP 429s; rate limits otherwise masquerade as jury rejections.
