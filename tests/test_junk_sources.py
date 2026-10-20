"""Tests for heuristic junk source detection, filtering, and ranking (docs/memory.md §4)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from adh.contract import JobSpec, SourceChannel
from adh.discovery import SourceDiscoveryAgent
from adh.discovery.agent import DiscoveryConfig, dedupe_and_rank, score_candidate
from adh.discovery.models import CandidateSource, RankedSource, SourceEvaluation
from adh.junk import is_junk_source, is_junk_url
from adh.search import DatasetHit, InMemoryDatasetRegistry, InMemorySearchProvider, SearchHit

DUMMY_PATH = Path(".")


def _rust_spec(minimal_spec_dict: dict) -> JobSpec:
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Q/A pairs on advanced Rust async macro design (declarative + procedural)"
    d["motivation"] = "SFT data to teach a model idiomatic async macro patterns"
    d.setdefault("discovery", {})
    return JobSpec.model_validate(d)


# =====================================================================
# Regression tests for live run: rust-async-live-small
# =====================================================================
# Verbatim junk hits scraped in the rust-async-live-small run that were
# subsequently skipped by the generator as off-topic junk grounding.
JUNK_HITS_RUST_ASYNC_LIVE_SMALL = [
    (
        "https://www.indeed.com/q-async-programming-jobs.html",
        "Now Hiring: 800 Async Programming Jobs | Indeed",
        (
            "+ View all AuraOne Human Data jobs in Remote - Remote jobs - AI Writing Evaluator jobs in Remote "
            "+ Salary Search: Rust Programming AI Evaluator salaries in Remote | | | "
            "Python Backend Developer Altius Technologes, inc Georgia | + Expertise in async/await patterns "
            "and concurrent programming. + Deep experience with REST APIs, GraphQL, and various integration patterns."
        ),
        "job",
    ),
    (
        "https://cryptorank.io/news/pin-ai",
        "PIN AI: Latest News, Social Media Updates and Insights | CryptoRank.io",
        (
            "Latest cryptocurrency newsRWA Tokenized stocks, ETFs, bonds and commoditiesHeatmaps Cryptocurrency "
            "market state visualizationPrediction Markets All-in-one analytics for prediction marketsAlerts "
            "Cryptocurrency signals and price alertsDrop Hunting Potential airdrops and activitiesWidgets Useful widgets"
        ),
        "crypto",
    ),
    (
        "https://cryptorank.io/price/pin-ai/team",
        "PIN AI Team | CryptoRank.io",
        (
            "Latest cryptocurrency newsRWA Tokenized stocks, ETFs, bonds and commoditiesHeatmaps Cryptocurrency "
            "market state visualizationPrediction Markets All-in-one analytics for prediction marketsAlerts "
            "Cryptocurrency signals and price alertsDrop Hunting Potential airdrops and activitiesWidgets Useful widgets"
        ),
        "crypto",
    ),
    (
        "https://www.stocktitan.net/news/WIMI/wi-mi-optimized-cloud-task-scheduling-in-cloud-computing-using-group-ptaiu1y8fuli.html",
        "WiMi Optimized Cloud Task Scheduling in Cloud Computing Using Group Intelligence Algorithms | WIMI Stock News",
        (
            "Cloud task scheduling using group intelligence algorithms can meet users' needs, improve the response "
            "speed of the system, reduce the cost, and improve resource utilization. The group intelligence algorithm "
            "can be applied to different aspects of cloud task scheduling, such as task allocation, task scheduling..."
        ),
        "stock",
    ),
    (
        "https://www.zoominfo.com/c/async-labs-llc/463194660",
        "Async Labs",
        (
            "# Async Labs. Async Labs is a software development and digital agency that specializes in delivering "
            "innovative IT solutions and digital products to clients worldwide. Their services include software and app "
            "development, digital marketing, digital design, and blockchain development. ## Company Details."
        ),
        "directory",
    ),
    (
        "https://www.indeed.com/viewjob?jk=2dc5f2656409bd09",
        "Sr. Rust Software Engineer - Remote - Indeed.com",
        (
            "Sr. Rust Software Engineer - Remote - Indeed.com. This job has expired on Indeed. Reasons could include: "
            "the employer is not accepting applications, is not actively hiring, or is reviewing applications. "
            "We seek a Senior Rust engineer with a startup mentality to construct the next generation of real-time cloud..."
        ),
        "job",
    ),
    (
        "https://www.indeed.com/q-full-time-rust-programming-$140,000-jobs.html",
        "Now Hiring: 1,000 Full Time Rust Programming $140,000 Jobs | Indeed",
        (
            "+ No relevant asynchronous programming experience. + Describe their use of asynchronous programming and "
            "concurrency. + Full-time, direct-hire position. + View all HireNow Staffing jobs in New York, NY - New York jobs "
            "+ Salary Search: Backend Software Engineer — Rust & Real-Time Crypto Trading salaries in New York, NY"
        ),
        "job",
    ),
]

LEGITIMATE_HITS_RUST_ASYNC_LIVE_SMALL = [
    (
        "https://rust-lang.github.io/async-book/07_workarounds/05_async_in_traits.html",
        "async in Traits - Asynchronous Programming in Rust",
        "Asynchronous Programming in Rust async in Traits workarounds and static dispatch patterns",
    ),
    (
        "https://docs.rs/async-trait",
        "async_trait - Rust",
        "Type erasure for async trait methods in Rust",
    ),
    (
        "https://doc.rust-lang.org/std/pin/index.html",
        "std::pin - Rust",
        "Types which borrow and must not be moved; Pin<&mut T> guarantees",
    ),
    (
        "https://blog.cloudflare.com/pin-and-unpin-in-rust",
        "Pin, Unpin, and why Rust needs them | Cloudflare Blog",
        "Deep dive into why self-referential futures require pinning in Rust and how Unpin is implemented",
    ),
    (
        "https://cafbit.com/post/tokio_internals",
        "Tokio internals: Understanding Rust's asynchronous I/O framework from the bottom up : Caffeinated Bitstream",
        "Tokio task scheduling, reactor, driver, and worker thread execution models",
    ),
    (
        "https://users.rust-lang.org/t/what-is-the-best-way-to-run-scheduled-concurrent-tasks-in-rust/43931",
        "What is the best way to run scheduled concurrent tasks in Rust? - help - The Rust Programming Language Forum",
        "Discussion on running scheduled async background tasks with Tokio and async-std",
    ),
    (
        "https://blog.masteringbackend.com/introducing-async-closures-rust",
        "Are async blocks dead? Introducing async closures (rust 1.85) - Mastering Backend",
        "An in-depth explanation of async closures in Rust 1.85, AsyncFn traits and borrowing rules",
    ),
]


@pytest.mark.parametrize(
    "url,title,snippet,expected_category",
    JUNK_HITS_RUST_ASYNC_LIVE_SMALL,
)
def test_junk_sources_from_rust_async_live_run_are_rejected(
    minimal_spec_dict, url, title, snippet, expected_category
):
    spec = _rust_spec(minimal_spec_dict)
    verdict = is_junk_source(url, title=title, snippet=snippet, spec=spec)
    assert verdict.is_junk, f"Expected {url} to be flagged as junk"
    assert verdict.reason
    assert expected_category in verdict.reason.lower()


@pytest.mark.parametrize(
    "url,title,snippet",
    LEGITIMATE_HITS_RUST_ASYNC_LIVE_SMALL,
)
def test_legitimate_sources_from_rust_async_live_run_are_accepted(
    minimal_spec_dict, url, title, snippet
):
    spec = _rust_spec(minimal_spec_dict)
    verdict = is_junk_source(url, title=title, snippet=snippet, spec=spec)
    assert not verdict.is_junk, (
        f"Expected legitimate source {url} NOT to be flagged as junk, got: {verdict.reason}"
    )


# =====================================================================
# Goal-awareness (topic exception) tests
# =====================================================================
def test_job_board_accepted_when_spec_is_about_recruitment(minimal_spec_dict):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Dataset of software engineering job postings, requirements, and salaries"
    d["motivation"] = "Training a model on tech recruitment market trends"
    spec = JobSpec.model_validate(d)

    verdict = is_junk_source(
        "https://www.indeed.com/viewjob?jk=2dc5f2656409bd09",
        title="Sr. Rust Software Engineer - Remote - Indeed.com",
        snippet="Full-time direct-hire position for senior software engineer",
        spec=spec,
    )
    assert not verdict.is_junk, (
        "Job board should NOT be flagged as junk when the spec is about recruitment"
    )


def test_crypto_site_accepted_when_spec_is_about_cryptocurrency(minimal_spec_dict):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = (
        "Cryptocurrency tokenomics, liquidity pools, and market capitalization trends"
    )
    d["motivation"] = "DeFi and crypto analytics"
    spec = JobSpec.model_validate(d)

    verdict = is_junk_source(
        "https://cryptorank.io/news/pin-ai",
        title="PIN AI: Latest News, Social Media Updates and Insights | CryptoRank.io",
        snippet="Cryptocurrency market state visualization and token price predictions",
        spec=spec,
    )
    assert not verdict.is_junk, (
        "Crypto site should NOT be flagged as junk when the spec is about cryptocurrency"
    )


def test_stock_news_accepted_when_spec_is_about_financial_markets(minimal_spec_dict):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Financial equity markets, earnings reports, and stock ticker analysis"
    d["motivation"] = "Financial NLP dataset"
    spec = JobSpec.model_validate(d)

    verdict = is_junk_source(
        "https://www.stocktitan.net/news/AAPL/apple-quarterly-earnings.html",
        title="Apple Quarterly Earnings | StockTitan",
        snippet="Apple reports quarterly results and stock price target revisions",
        spec=spec,
    )
    assert not verdict.is_junk, (
        "Financial site should NOT be flagged as junk when the spec is about stock markets"
    )


def test_company_directory_accepted_when_spec_is_about_b2b_companies(minimal_spec_dict):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "B2B SaaS companies, headquarters, revenue, and employee counts directory"
    d["motivation"] = "Company research"
    spec = JobSpec.model_validate(d)

    verdict = is_junk_source(
        "https://www.zoominfo.com/c/async-labs-llc/463194660",
        title="Async Labs",
        snippet="Company Details: headquarters, employees, industry, revenue",
        spec=spec,
    )
    assert not verdict.is_junk, (
        "Company directory should NOT be flagged as junk when spec is about companies"
    )


# =====================================================================
# Integration with SourceDiscoveryAgent
# =====================================================================
def test_heuristic_discovery_filters_junk_sources_from_plan(minimal_spec_dict):
    """Heuristic discovery must discard junk search hits before returning the fetch plan."""
    pages = [
        # 1 valid technical source
        SearchHit(
            url="https://rust-lang.github.io/async-book/07_workarounds/05_async_in_traits.html",
            title="async in Traits - Asynchronous Programming in Rust",
            snippet="rust async macro declarative procedural async traits workarounds",
            published="2024-01-01",
            license_hint="MIT",
        ),
        # 2 junk hits from live run rust-async-live-small
        SearchHit(
            url="https://www.indeed.com/q-async-programming-jobs.html",
            title="Now Hiring: 800 Async Programming Jobs | Indeed",
            snippet="rust async macro declarative procedural now hiring 800 jobs salary search",
            published="2024-01-01",
        ),
        SearchHit(
            url="https://cryptorank.io/news/pin-ai",
            title="PIN AI: Latest News, Social Media Updates and Insights | CryptoRank.io",
            snippet="rust async macro pin ai cryptocurrency market state token price",
            published="2024-01-01",
        ),
    ]
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(),
        search=InMemorySearchProvider(pages=pages),
    )
    plan = agent.discover(_rust_spec(minimal_spec_dict), DUMMY_PATH)

    urls = [s.candidate.locator for s in plan.sources]
    assert "https://rust-lang.github.io/async-book/07_workarounds/05_async_in_traits.html" in urls
    assert "https://www.indeed.com/q-async-programming-jobs.html" not in urls
    assert "https://cryptorank.io/news/pin-ai" not in urls
    # Filtering is visible in the plan, as it is for the deep agent.
    junk_notes = [n for n in plan.notes if "junk source" in n]
    assert any("indeed.com" in n for n in junk_notes)
    assert any("cryptorank.io" in n for n in junk_notes)
    assert len(junk_notes) == len(set(junk_notes))  # one note per URL, not per query


# =====================================================================
# Technical pages that share vocabulary with junk categories
# =====================================================================
# "Job" is a core term in CI, schedulers and orchestrators; ssh-keygen is not
# warez; ML tutorials use stock data. A single weak signal never convicts.
TECHNICAL_LOOKALIKES = [
    (
        "Tokio task scheduling internals",
        "https://blog.example.dev/tokio-cron",
        "Scheduling jobs in Tokio with tokio-cron-scheduler",
        "Run periodic async tasks",
    ),
    (
        "Kubernetes workload controllers",
        "https://kubernetes.io/docs/concepts/workloads/controllers/job/",
        "Jobs | Kubernetes",
        "A Job creates one or more Pods and retries them until they succeed",
    ),
    (
        "GitLab CI pipelines",
        "https://docs.gitlab.com/ee/ci/jobs/",
        "CI/CD jobs | GitLab",
        "Pipeline configuration begins with jobs",
    ),
    (
        "Distributed build systems",
        "https://bazel.build/docs/parallel",
        "Parallel builds",
        "Bazel can run 8 jobs in parallel on one machine",
    ),
    (
        "Linux system administration with SSH",
        "https://www.ssh.com/academy/ssh/keygen",
        "ssh-keygen - Generate a New SSH Key",
        "ssh-keygen is a tool for creating new authentication key pairs",
    ),
    (
        "Time-series forecasting with LSTMs",
        "https://machinelearningmastery.com/lstm-forecast",
        "Time series prediction with LSTM",
        "Using stock market data as an example dataset",
    ),
]


@pytest.mark.parametrize(("goal", "url", "title", "snippet"), TECHNICAL_LOOKALIKES)
def test_technical_pages_sharing_junk_vocabulary_are_kept(
    minimal_spec_dict, goal, url, title, snippet
):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = goal
    verdict = is_junk_source(url, title=title, snippet=snippet, spec=JobSpec.model_validate(d))
    assert not verdict, f"{url} wrongly flagged: {verdict.reason}"


@pytest.mark.parametrize(
    ("url", "title", "snippet", "category"),
    [
        # Unknown hosts need corroborating signals, but real listings have them.
        (
            "https://acme.dev/careers/senior-rust-engineer",
            "Senior Rust Engineer",
            "Full-time position on our platform team. Apply now.",
            "job",
        ),
        (
            "https://example-jobs-aggregator.net/rust",
            "Now hiring: 120 Rust jobs in Berlin",
            "Salary search for Rust developers",
            "job",
        ),
        (
            "https://coins.example.net/pin-ai",
            "PIN AI price",
            "Circulating supply, 24h trading volume and market cap rank",
            "crypto",
        ),
        (
            "https://wire.example.com/wimi",
            "WiMi announces cloud scheduling (NASDAQ: WIMI)",
            "Press release",
            "stock",
        ),
        (
            "https://downloads.example.ru/photoshop",
            "Photoshop 2024 keygen",
            "free download crack",
            "spam",
        ),
    ],
)
def test_corroborated_junk_on_unknown_hosts_is_rejected(
    minimal_spec_dict, url, title, snippet, category
):
    verdict = is_junk_source(url, title=title, snippet=snippet, spec=_rust_spec(minimal_spec_dict))
    assert verdict
    assert category in verdict.reason.lower()


@pytest.mark.parametrize(
    ("goal", "url"),
    [
        ("Wall Street analyst reports", "https://seekingalpha.com/article/123"),
        ("Smart contract auditing in Solidity", "https://etherscan.io/address/0xabc"),
        ("Business directory entries for SaaS vendors", "https://www.zoominfo.com/c/acme/1"),
    ],
)
def test_multi_word_intent_terms_exempt_their_category(minimal_spec_dict, goal, url):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = goal
    assert not is_junk_source(url, spec=JobSpec.model_validate(d))


def test_llm_token_goal_does_not_switch_off_crypto_filtering(minimal_spec_dict):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "LLM tokenizer training: BPE tokens and vocabularies"
    verdict = is_junk_source(
        "https://coinmarketcap.com/currencies/pin-ai/", spec=JobSpec.model_validate(d)
    )
    assert verdict and "crypto" in verdict.reason


@pytest.mark.parametrize(
    ("url", "keywords", "expected"),
    [
        ("https://www.indeed.com/viewjob?jk=1", (), True),
        ("https://careers.example.com/rust", (), True),
        ("https://www.linkedin.com/jobs/view/123", (), True),
        ("https://www.rust-lang.org/careers/", (), True),
        ("https://kubernetes.io/docs/concepts/workloads/controllers/job/", ("kubernetes",), False),
        ("https://docs.gitlab.com/ee/ci/jobs/", ("gitlab", "ci"), False),
        ("https://www.indeed.com/viewjob?jk=1", ("rust", "jobs"), False),
    ],
)
def test_crawler_outlink_filter_blocks_job_sites_not_job_concepts(url, keywords, expected):
    assert is_junk_url(url, keywords=keywords) is expected


def test_dataset_candidates_are_not_junk_scored(minimal_spec_dict):
    """Junk categories describe web pages; a matching dataset keeps its real evaluation."""
    ds = DatasetHit(
        dataset_id="org/rust-async-macros-qa",
        locator="https://huggingface.co/datasets/org/rust-async-macros-qa",
        title="Rust async macro Q/A",
        description=(
            "rust async macro design declarative procedural question-answering input output "
            "(sources include a NASDAQ: RUST vendor press release)"
        ),
        license="MIT",
        downloads=5000,
        last_modified="2024-09-01",
    )
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(datasets=[ds], samples={ds.dataset_id: ds.description}),
        search=InMemorySearchProvider(),
    )
    plan = agent.discover(_rust_spec(minimal_spec_dict), DUMMY_PATH)

    chosen = [s for s in plan.sources if s.candidate.locator == ds.locator]
    assert chosen, f"dataset dropped; plan notes: {plan.notes}"
    assert chosen[0].evaluation.license_ok
    assert chosen[0].evaluation.score > 0


def test_allowed_license_outranks_better_shaped_unknown_license_source():
    """The license ladder: an allowed-license source beats a stronger-looking unknown one."""
    config = DiscoveryConfig()
    common = {"est_yield": 10, "channel": SourceChannel.WEB, "freshness_ok": True}
    licensed = score_candidate(
        shape_match=0.5, quality=0.5, license_ok=True, config=config, **common
    )
    unknown = score_candidate(
        shape_match=0.9, quality=0.9, license_ok=False, config=config, **common
    )
    assert licensed > unknown


# =====================================================================
# Ranking and tie-breaking
# =====================================================================
def test_dedupe_and_rank_breaks_ties_on_shape_and_quality():
    """When candidates have identical overall scores (e.g. 0.0 due to penalties),
    higher shape_match and quality must rank first rather than arbitrary FIFO."""
    cand1 = CandidateSource(
        source_id="1",
        channel=SourceChannel.WEB,
        locator="https://example.com/low-quality",
        title="Low Quality",
        aspect_id="a0",
    )
    eval1 = SourceEvaluation(
        source_id="1",
        shape_match=0.15,
        quality=0.20,
        license_ok=False,
        freshness_ok=True,
        estimated_yield=0,
        score=0.0,
    )

    cand2 = CandidateSource(
        source_id="2",
        channel=SourceChannel.WEB,
        locator="https://example.com/high-quality",
        title="High Quality",
        aspect_id="a0",
    )
    eval2 = SourceEvaluation(
        source_id="2",
        shape_match=0.60,
        quality=0.45,
        license_ok=False,
        freshness_ok=True,
        estimated_yield=1,
        score=0.0,
    )

    ranked = dedupe_and_rank(
        [
            RankedSource(candidate=cand1, evaluation=eval1),
            RankedSource(candidate=cand2, evaluation=eval2),
        ]
    )

    assert len(ranked) == 2
    assert ranked[0].candidate.locator == "https://example.com/high-quality"
    assert ranked[1].candidate.locator == "https://example.com/low-quality"
