# Project Goal

## North star (one sentence)

> **Turn a plain-English description of a dataset someone wishes existed into that dataset —
> verified, deduplicated, provenance-tracked, and training-ready — without the requester needing to
> know where the data lives or how to get it.**

## The problem this exists to solve

The hard part of building a dataset is not generating or scraping — it's **knowing where the data is
in the first place.** A person knows *what* they need ("Q/A pairs on advanced Rust async macros, as
JSONL") and *what shape* they need it in, but not *which sources, datasets, or methods* will produce
it. Existing tools assume you already hand them sources. This system removes that assumption.

So the system must be **intelligent, not just automated**: given a goal, it decides *where the data
lives* and *how to obtain it* — reusing an existing dataset when one exists, crawling the web when it
doesn't, or generating it directly when the model already knows the domain.

## The promise (input → output)

| You give it | It gives you |
|-------------|--------------|
| A goal in plain English + the record shape you want + a verifiability tier + budget limits. **No sources.** | A verified, deduplicated dataset (JSONL/Parquet) + a manifest + an auto-generated datasheet + Croissant metadata + full per-record provenance. |

## What "intelligent" means here (the differentiator)

1. **Discovery-first.** It finds sources from the goal — never requires them. It searches existing
   dataset registries first (reuse > crawl > synthesize), evaluates candidates against the actual
   target shape, ranks them, and (by default) shows you the list for approval before fetching.
2. **Coverage-driven, not budget-driven.** It stops collecting a topic when new data stops adding
   information (saturation), not when an arbitrary page count is hit. Hard limits are a safety floor.
3. **Verify everything.** Every record passes a tiered, cheapest-first quality gate matched to its
   verifiability (deterministic check, LLM-jury, or a hybrid of both). Nothing ships unverified.
4. **Provenance always.** Every record knows where it came from, which model made it, under what
   prompt and license, and how it scored — auditable and reproducible.
5. **Controllable.** A human approval gate on discovered sources (default on); a flag for autonomous
   runs. Cost-bounded throughout.

## Success criteria

The system succeeds when a non-expert can:

- describe a dataset they want in a sentence or two and the target format,
- **not** supply any sources,
- approve a sensible ranked list of discovered sources,
- and receive a dataset where every record validates against the schema, carries provenance, is
  deduplicated and decontaminated, and **measurably improves a model when trained on** (downstream
  lift — the only evaluation that ultimately matters).

…all within stated cost/time limits, resumable after failure.

## Scope (Phase 0–4)

**In scope now:** text/code datasets · three source channels (existing datasets · web crawl · pure
synthesis) · the three verifiability tiers · single-tenant batch jobs.

**Explicitly deferred:** multimodal payloads · GitHub-code & structured-API channels · streaming/online
updates · multi-tenant marketplace · an LLM that auto-derives limits from requirements.

## How a specific request enters the system

Your one-line goal becomes the `goal`/`motivation` of a **Job Spec** (see
[`PHASE0-CONTRACT.md`](./PHASE0-CONTRACT.md)). The Source Discovery Agent decomposes it into aspects
and routes each to a channel. Everything downstream is built against the locked Phase 0 contract.

---

## Two briefs you can hand to Claude

### A. To **build** the harness

```
You are building the Agentic Dataset Generation Harness in this repo.
Read, in order: GOAL.md, README.md, PHASE0-CONTRACT.md, OUTLINE.md, RESEARCH.md.

Goal: <e.g. "implement Phase 1.0 Source Discovery Agent + the web/datasets channels">

Rules:
- Honor the LOCKED Phase 0 contract exactly (Job Spec + Record Schema + verifiability tiers).
- Reuse-first per the OUTLINE table (Crawl4AI, deepagents, Distilabel, datatrove, Argilla, HF Hub API)
  — verify current APIs against RESEARCH §APIs (several 2025-era signatures are stale).
- Discovery-first: never require sources from the user; default the approval gate ON.
- Stop and confirm before any architectural deviation from the contract.
```

### B. To **request a dataset** (once the harness exists — the real product input)

```
Build me a dataset.

Goal:        <plain English — what the data is about and what it's for>
Record shape: <e.g. {input, output} Q/A pairs  |  point to a JSON Schema>
Format:      <jsonl | parquet>
Verifiability: <1 hard | 2 soft | 3 hybrid>
Limits:      <max cost, max time, target #records>
Sources:     none — discover them. (Optional: hints / exclusions / license requirements.)
```

Example B:
```
Build me a dataset.
Goal: Teach a model idiomatic advanced Rust async macro patterns (declarative + procedural).
Record shape: {input, output} Q/A pairs.
Format: parquet.  Verifiability: 2 (soft).  Limits: $150, 8h, 25,000 records.
Sources: none — discover them. Prefer permissive licenses; exclude reddit.com.
```
