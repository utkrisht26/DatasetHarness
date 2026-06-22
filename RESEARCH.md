# Research Record

> Evidence base for the Phase 0 contract and project outline. Compiled 2026-06-13 from four parallel
> research passes (GitHub + primary docs + web). Every claim carries an inline source. Organized by
> the design decision it supports.

---

## §DISCOVERY — Source discovery & channel routing (the intelligent core)

The user states a goal, not sources. The system routes each aspect of the goal to the cheapest viable
channel: **existing datasets > web crawl > pure synthesis.** Evidence for this ordering:

- **Reuse before scrape.** Re-deriving data that already exists as a clean, licensed dataset wastes
  compute and risks lower quality than the curated original; large open corpora are published with
  rich machine-readable metadata for exactly this reuse (HF dataset cards, Croissant — see §SPEC).
  Discovery should therefore search dataset registries (HF Hub, Kaggle, data.gov/CKAN) first.
- **Synthesis is viable with no source at all.** Persona Hub generates 1B-persona-conditioned records
  (e.g. math problems) purely from model knowledge, with diversity controlled by MinHash + embedding
  dedup — i.e. for domains the model already knows, the "source" is the model itself
  ([arXiv 2406.20094](https://arxiv.org/abs/2406.20094)). This justifies a `synthesis` channel that
  bypasses acquisition entirely.
- **Web crawl is the fallback** when no ready dataset exists and the model's own knowledge is
  insufficient — handled by Crawl4AI with `AdaptiveCrawler` saturation stopping (see §APIs).

> Note: the *channel-routing policy* is a design synthesis, not a single cited result. The individual
> channels (dataset reuse, persona-style synthesis, adaptive crawl) are each grounded above; the
> orchestration that picks among them per aspect — plus the **coverage-gap-driven re-discovery loop**
> and the **human approval gate** — is the harness's own contribution.

---

## §SPEC — Dataset specification, documentation & provenance standards

### Datasheets for Datasets (Gebru et al.)
Seven prescribed sections: **Motivation, Composition, Collection Process, Preprocessing/Cleaning/Labeling,
Uses, Distribution, Maintenance** ([arxiv.org/abs/1803.09010](https://arxiv.org/abs/1803.09010),
[pdf](https://arxiv.org/pdf/1803.09010)). These map 1:1 onto our split: Motivation/Uses/Distribution →
**Job Spec**; Composition/Collection/Preprocessing/Maintenance → the **auto-generated datasheet**.

### Hugging Face Dataset Cards
Metadata lives in YAML frontmatter in the dataset `README.md`. Confirmed fields: `language`, `license`,
`license_name`/`license_link`, `tags`, `annotations_creators`, `pretty_name`, `size_categories`,
`source_datasets`, `task_categories`, `task_ids`, `configs` (`config_name`, `data_files`→`split`/`path`),
`dataset_info` (`features`→`name`/`dtype`, `splits`→`num_examples`)
([docs](https://huggingface.co/docs/hub/datasets-cards),
[template](https://raw.githubusercontent.com/huggingface/hub-docs/main/datasetcard.md)).
Caveat: `multilinguality` is legacy — not in the current template.

### Croissant (MLCommons)
JSON-LD on schema.org (`cr:` + `sc:` + `dct:` namespaces); conformance via
`"dct:conformsTo": "http://mlcommons.org/croissant/1.0"`. Raw sources = **`FileObject`**
(`contentUrl`, `encodingFormat`, `sha256`, `contentSize`) / **`FileSet`** (`includes`/`excludes`).
Structured view = **`RecordSet`** → **`Field`** (`source`+`extract`+`transform`, `dataType`).
Loads directly into TF/JAX/PyTorch; implemented across HF/Kaggle/OpenML; discoverable via Google
Dataset Search ([spec](https://docs.mlcommons.org/croissant/docs/croissant-spec.html),
[mlcommons.org/croissant](https://mlcommons.org/croissant/)).

### Provenance models — W3C PROV & MLMD
- **PROV:** Entity / Activity / Agent; relations `wasGeneratedBy`, `used`, `wasDerivedFrom`,
  `wasAttributedTo`, `wasAssociatedWith`, `wasInformedBy`; times `generatedAtTime`/`startedAtTime`
  ([PROV-DM](https://www.w3.org/TR/prov-dm/), [PROV-O](https://www.w3.org/TR/prov-o/)).
- **MLMD:** Artifact / Execution / Context, linked by Event (INPUT/OUTPUT), Attribution, Association;
  nodes carry typed `properties` + free `custom_properties`
  ([TFX MLMD](https://www.tensorflow.org/tfx/guide/mlmd),
  [proto](https://github.com/google/ml-metadata/blob/master/ml_metadata/proto/metadata_store.proto)).
- **Per-record mapping:** record = Entity/Artifact; generation step = Activity/Execution; model =
  Agent (PROV) or input Artifact (MLMD). At volume, batch to one node per partition and drop to
  per-record only where traceability demands.

### Per-record lineage conventions
- **Data Provenance Initiative** tracks `Dataset URL`, `Languages`, `Task Categories`, `Text Sources`,
  `Model Generated`, `Derived from Datasets`, `Licenses` + `License Conditions`, `Time of Collection`
  ([repo](https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection),
  [arXiv 2310.16787](https://arxiv.org/pdf/2310.16787)).
- **Distilabel** attaches `model_name` + `distilabel_metadata` (`raw_output_{STEP}`, `raw_input_{STEP}`,
  `statistics_{STEP}` with `input_tokens`/`output_tokens`); each step namespaces its keys
  ([task guide](http://distilabel.argilla.io/dev/sections/how_to_guides/basic/task/)).

### Validation & storage
- **JSON Schema:** `type`, `properties`, `required`, `additionalProperties:false`
  ([json-schema.org](https://json-schema.org/understanding-json-schema/reference/object)).
- **Pydantic:** `BaseModel` + `Field(...)` constraints; `model_validate_json()`; one `ValidationError`
  with field locations ([pydantic.dev](https://pydantic.dev/docs/validation/latest/concepts/models/)).
- **JSONL** for streaming/append ([jsonlines.org](https://jsonlines.org/)); **Parquet** (columnar,
  embedded schema, ~100MB row groups, predicate pushdown) is the HF `datasets` default
  ([HF Parquet](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format),
  [Arrow](https://arrow.apache.org/docs/python/parquet.html)).

---

## §REUSE — Synthetic-data & curation frameworks (buy vs build)

| Framework | Core abstractions | Maturity / license | Decision |
|-----------|-------------------|--------------------|----------|
| **Distilabel** (Argilla/HF) | `Pipeline`→`Step`/`Task`→`LLM`; `Distiset` output; `UltraFeedback` judge (`ratings`/`rationales`, aspects: helpfulness/honesty/instruction-following/truthfulness) | ~3.25k★, Apache-2.0, active | **Reuse** as generate+judge layer |
| **datatrove** (HF) | `Document`(`text`/`id`/`metadata`), `PipelineStep`; MinHash 4-stage (`Signature`→`Buckets`→`Cluster`→`Filter`); Local/Slurm/Ray executors | ~3.08k★, Apache-2.0, v0.9.0 | **Reuse** for CPU dedup/clean |
| **NeMo Curator** (NVIDIA) | `DocumentDataset` (Dask); `FuzzyDuplicates`, `SemDedup`, `PIIRedactor`, `QualityClassifier`; RAPIDS+Ray | ~1.6k★, Apache-2.0, active | **Optional** GPU bolt-on |
| **Argilla** | `TextGenerationToArgilla`/`PreferenceToArgilla`; UltraFeedback scores auto-injected as suggestions | ~5k★, Apache-2.0, active | **Reuse** as HITL surface |
| **Lilac** | Signals (PII/stats/lang/near-dup), Concepts, clustering | ~1.1k★, Apache-2.0, **ARCHIVED 2025-07-25** | **Concepts only** — do not depend |
| **Bespoke Curator** | structured-output-centric synthetic pipelines | ~1.7k★, Apache-2.0, active | lighter Distilabel alt |
| **NeMo Data Designer** | seed/from-scratch synthetic w/ statistical diversity + field correlations | active | relevant for tabular/structured |

Sources: [distilabel](https://distilabel.argilla.io/latest/sections/getting_started/quickstart/) ·
[UltraFeedback](https://distilabel.argilla.io/latest/components-gallery/tasks/ultrafeedback/) ·
[datatrove](https://github.com/huggingface/datatrove) ·
[datatrove minhash example](https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication.py) ·
[NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) ·
[NeMo fuzzy dedup](https://docs.nvidia.com/nemo/curator/0.25.7/curate-text/process-data/deduplication/index.html) ·
[Argilla integration](https://distilabel.argilla.io/latest/sections/how_to_guides/advanced/argilla/) ·
[Lilac archive](https://github.com/databricks/lilac) /
[Databricks blog](https://www.databricks.com/blog/lilac-joins-databricks-simplify-unstructured-data-evaluation-generative-ai) ·
[Bespoke Curator](https://github.com/bespokelabsai/curator) ·
[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner).

**Net build surface:** orchestration glue tying datatrove (clean) → Distilabel (generate+judge) →
Argilla (review), + NeMo Curator as optional GPU dedup/PII. Little core logic is net-new.

---

## §APIs — Acquisition stack (verified current, June 2026)

### Crawl4AI (`unclecode/crawl4ai`, v0.8.x)
- Core async: `AsyncWebCrawler(config=BrowserConfig(...))`; `arun(url, config=CrawlerRunConfig(...))`,
  `arun_many(urls, ...)` ([README](https://raw.githubusercontent.com/unclecode/crawl4ai/main/README.md),
  [deep-crawling](https://docs.crawl4ai.com/core/deep-crawling/)).
- Deep crawl: `BFSDeepCrawlStrategy` / `DFSDeepCrawlStrategy` / **`BestFirstCrawlingStrategy`**
  (`max_depth`, `max_pages`, `score_threshold`, `url_scorer`).
- Filtering: `PruningContentFilter(threshold=0.48)`, **`BM25ContentFilter`** → `result.markdown.fit_markdown`.
- **`AdaptiveCrawler`** — stops on **coverage / consistency / saturation**;
  `AdaptiveConfig(confidence_threshold=0.7, ...)`; `strategy="statistical"` (no model) or `"embedding"`
  ([adaptive-crawling](https://docs.crawl4ai.com/core/adaptive-crawling/)).
- **Official MCP server** ships with the Docker deployment at `http://localhost:11235`, schemas at
  `/mcp/schema`, transports **SSE `/mcp/sse`** + WS `/mcp/ws`; tools: `md`, `html`, `screenshot`, `pdf`,
  `execute_js`, `crawl`, `ask` ([self-hosting](https://docs.crawl4ai.com/core/self-hosting/)).
- `ProxyConfig` rotation; `browser_type="undetected"`; v0.8.9 adds SSRF validation.

### deepagents (`langchain-ai/deepagents`, v0.6.10, 2026-06-13)
- `create_deep_agent(model=None, tools=..., *, system_prompt=..., subagents=[...], middleware=...,
  checkpointer=..., store=..., ...)` — built on LangGraph, returns a `CompiledStateGraph`
  ([graph.py](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py),
  [docs](https://docs.langchain.com/oss/python/deepagents/overview)).
- Built-in tools auto-injected: `write_todos`; virtual FS `ls`/`read_file`/`write_file`/`edit_file`;
  `task` (spawn context-isolated subagents).
- `SubAgent` = `name` + `description` + `system_prompt`/`prompt` (+ optional `tools`, `model`).

### langchain-mcp-adapters
- `MultiServerMCPClient({...}).get_tools()` is the URL-based entry point;
  `load_mcp_tools(session)` takes a **`ClientSession`**, not a URL.
- Transports: `stdio`, **`streamable_http`/`http`** (preferred), `sse` (legacy)
  ([README](https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/main/README.md)).

### ⚠ Stale assumptions in the original thread → corrected
| Thread (2025) | Current (Jun 2026) |
|---|---|
| `create_deep_agent(instructions=...)` | **`system_prompt`** |
| model id `claude-3-5-sonnet` | `model` default `None`; docs use `anthropic:claude-sonnet-4-6` |
| `load_mcp_tools("http://...sse")` | `MultiServerMCPClient({...}).get_tools()` |
| MCP default = SSE | adapters default **`streamable_http`**; but Crawl4AI's official server is **SSE/WS** → mismatch to handle |
| `BestFirstStrategy` | **`BestFirstCrawlingStrategy`** |
| "BM25" filter | **`BM25ContentFilter`** |

---

## §QC — Quality control: verifiability, diversity, dedup, contamination, judging

### RLVR / verifiability
- RLVR (Tülu 3) "replaces the reward model with a verification function" — reward only when
  "verifiably correct, otherwise 0" ([allenai](https://allenai.org/blog/tulu-3-technical),
  [arXiv 2411.15124](https://arxiv.org/abs/2411.15124)). DeepSeek-R1 uses rule-based Accuracy+Format
  rewards, avoiding neural PRMs because verifiable signals are "harder to game"
  ([arXiv 2501.12948](https://arxiv.org/abs/2501.12948)).
- Three verifiable classes: math (normalize+exact-match), code (sandboxed unit tests — Docker/gVisor,
  no network, timeouts), structured/constraint (IFEval, execution-checked SQL, format rewards)
  ([promptfoo](https://www.promptfoo.dev/blog/rlvr-explained/)).
- **Verifier is only as good as its coverage:** extensional-only checks admit false positives ("hacking
  gap") ([arXiv 2604.15149](https://arxiv.org/abs/2604.15149)); **spurious/random rewards** can still
  lift Qwen on MATH (+21pts) via GRPO clipping bias, model-family-dependent
  ([arXiv 2506.10947](https://arxiv.org/abs/2506.10947)). → **treat verifiers as adversarially gameable.**

### Diversity / seeding
- **Self-Instruct** — 175 seeds, add only if ROUGE-L < 0.7 vs existing ([arXiv 2212.10560](https://arxiv.org/abs/2212.10560)).
- **Evol-Instruct/WizardLM** — in-depth + in-breadth evolving; elimination of no-gain evolutions
  ([arXiv 2304.12244](https://arxiv.org/abs/2304.12244)).
- **Persona Hub** — 1B personas; MinHash(0.9)+embedding(0.9) dedup; output diversity exceeds persona
  diversity ([arXiv 2406.20094](https://arxiv.org/abs/2406.20094)).
- **Model collapse** — "tails of the original distribution disappear" with indiscriminate synthetic
  training ([Nature 2024](https://www.nature.com/articles/s41586-024-07566-y)); **accumulate** real+synthetic
  ([arXiv 2404.01413](https://arxiv.org/html/2404.01413v2)); diversity > volume
  ([arXiv 2511.01490](https://arxiv.org/pdf/2511.01490)).

### Deduplication
- Exact/substring: dedup makes LMs emit memorized text ~10× less; `ExactSubstr` ≥50 BPE tokens
  ([arXiv 2107.06499](https://arxiv.org/abs/2107.06499)).
- MinHash+LSH near-dup, `P=1−(1−s^r)^b`; de-facto **0.8 Jaccard** (SlimPajama 1.21T→627B; FineWeb 5-gram/0.75)
  ([HF dedup](https://huggingface.co/blog/dedup),
  [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)).
- Semantic dedup (SemDeDup/D4): cluster-then-prune; ~50% LAION removable, ~15–20% efficiency gains
  ([arXiv 2303.09540](https://arxiv.org/abs/2303.09540), [arXiv 2308.12284](https://arxiv.org/abs/2308.12284)).
- **Over-dedup warning:** FineWeb global MinHash deleted higher-quality recurring content (490B→31B,
  worse than no dedup) → **dedup per-shard, not global**
  ([kili](https://kili-technology.com/blog/what-can-we-learn-from-hugging-face-s-fineweb-dataset)).

### Benchmark contamination
- Leakage 1–45% across QA benchmarks, growing over time; rephrased contamination evades n-gram
  detection (13B "MMLU 85.9") ([arXiv 2406.04244](https://arxiv.org/html/2406.04244v1),
  [arXiv 2311.04850](https://arxiv.org/abs/2311.04850),
  [lmsys decontaminator](https://www.lmsys.org/blog/2023-11-14-llm-decontaminator/)).
- Methods cheapest→strongest: n-gram (13-gram / 50-char / 70%-of-8-gram), embedding cosine>0.8,
  LLM-rephrase-judge (highest F1), Min-K%++ membership inference, canary strings
  ([arXiv 2310.16789](https://arxiv.org/abs/2310.16789)).
  **Mandatory before any eval split.**

### LLM-as-a-judge (Zheng et al., [arXiv 2306.05685](https://arxiv.org/abs/2306.05685))
- Biases: **position** (GPT-4 ~65% order-consistency, Claude-v1 ~24%), **verbosity** (list-padding
  failure: Claude/GPT-3.5 91%, GPT-4 9%), **self-preference** (GPT-4 +10% / Claude-v1 +25% on own outputs).
- Mitigations: swap+both-order agreement; reference-guided (math fail 70%→15%);
  **panel of disjoint-family judges (PoLL)** — beats single GPT-4 at >7× lower cost
  ([arXiv 2404.18796](https://arxiv.org/abs/2404.18796)); length-controlled
  ([arXiv 2404.04475](https://arxiv.org/abs/2404.04475)). Judges fail at reasoning verification;
  correlation alone ≠ validation ([arXiv 2506.13639](https://arxiv.org/html/2506.13639v1)).

### Quality filtering
- Heuristic (Gopher word-count/symbol-ratio/repetition/stop-words; C4 punctuation/langdetect;
  CCNet/KenLM perplexity) ([Gopher 2112.11446](https://arxiv.org/pdf/2112.11446),
  [C4 2104.08758](https://arxiv.org/pdf/2104.08758)).
- Classifiers: DCLM fastText (+3.5 Core), **FineWeb-Edu** (Llama-3-70B labels → linear head, ≥3 threshold;
  MMLU 33→37, ARC 46→57 at ~10× fewer tokens) ([arXiv 2406.17557](https://arxiv.org/html/2406.17557v1)).
- **Caveat:** C4 blocklist removed AAE ~42% / Hispanic-English ~32% vs White-aligned ~6.2%; KenLM
  perplexity structurally higher on AAE ([arXiv 2104.08758](https://arxiv.org/pdf/2104.08758),
  [arXiv 2104.06390](https://arxiv.org/pdf/2104.06390)) → **audit dialect skew.**
- Pattern: heuristics first (strip garbage) → classifier second (quality cut).

### Cheapest-first verification stack (the operative recipe)
`hash → heuristics → n-gram decontam → MinHash → deterministic verifier → quality classifier →
semantic dedup → LLM-jury → LLM paraphrase decontam`. Steps 1–5 are CPU/cheap and remove the bulk of
bad records before any GPU embedding (6–7) or LLM call (8–9). Tier 1 can terminate at step 5; Tier 3
gates at step 5 so the jury sees only valid candidates; eval splits additionally require steps 3 + 9.
