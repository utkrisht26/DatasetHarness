"""Heuristic junk source classification and filtering (Phase 1.0).

Filters out irrelevant, low-information, and spam search hits before acquisition:
- Job boards & recruitment listings (Indeed, Glassdoor, LinkedIn jobs, Greenhouse, Lever, etc.)
- Cryptocurrency price/token speculation trackers (CryptoRank, CoinMarketCap, CoinGecko, etc.)
- Stock ticker & financial PR news wire spam (StockTitan, Seeking Alpha, Benzinga, etc.)
- Business directories & company profile databases (ZoomInfo, Crunchbase, etc.)
- Parked domains, coupon/promo spam, crack/warez downloads

All category rules are goal-aware: if the job spec or aspect is explicitly about
jobs, cryptocurrency, finance, or corporate intelligence, hits in that category
are exempt and preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from adh.contract import JobSpec
    from adh.discovery.models import Aspect

_WORD_RE = re.compile(r"[a-z0-9]+")

# ------------------------------------------------------------- Intent vocabularies
# When the goal or aspect contains keywords from these vocabularies, candidate hits
# matching the corresponding category are NOT treated as junk.
_JOB_INTENT_TERMS = frozenset(
    {
        "job",
        "jobs",
        "hiring",
        "recruitment",
        "recruiter",
        "recruiters",
        "career",
        "careers",
        "employment",
        "resume",
        "resumes",
        "salary",
        "salaries",
        "interview",
        "interviews",
        "workplace",
        "staffing",
        "compensation",
        "headhunting",
        "jobseeker",
        "jobseekers",
    }
)

_CRYPTO_INTENT_TERMS = frozenset(
    {
        "crypto",
        "cryptocurrency",
        "cryptocurrencies",
        "bitcoin",
        "ethereum",
        "blockchain",
        "blockchains",
        "web3",
        "token",
        "tokens",
        "tokenomics",
        "defi",
        "nft",
        "nfts",
        "solana",
        "smart contract",
        "smart contracts",
        "altcoin",
        "altcoins",
        "memecoin",
        "memecoins",
        "dao",
        "daos",
    }
)

_FINANCE_INTENT_TERMS = frozenset(
    {
        "stock",
        "stocks",
        "equity",
        "equities",
        "finance",
        "financial",
        "investing",
        "investor",
        "investors",
        "trading",
        "sec",
        "ticker",
        "tickers",
        "earnings",
        "wall street",
        "nasdaq",
        "nyse",
        "bonds",
        "forex",
        "shareholder",
        "shareholders",
        "dividend",
        "dividends",
    }
)

_COMPANY_INTENT_TERMS = frozenset(
    {
        "company",
        "companies",
        "startup",
        "startups",
        "directory",
        "directories",
        "firm",
        "firms",
        "vendor",
        "vendors",
        "b2b",
        "headquarters",
        "lead generation",
        "corporate directory",
        "business directory",
    }
)

# ------------------------------------------------------------- Category Domains
_JOB_DOMAINS = frozenset(
    {
        "indeed.com",
        "glassdoor.com",
        "ziprecruiter.com",
        "monster.com",
        "dice.com",
        "simplyhired.com",
        "lever.co",
        "greenhouse.io",
        "workable.com",
        "wellfound.com",
        "angel.co",
        "remoteok.com",
        "weworkremotely.com",
        "remotive.com",
        "builtin.com",
        "jobs.net",
        "careers24.com",
        "jobserve.com",
        "theladders.com",
        "hired.com",
    }
)

_CRYPTO_DOMAINS = frozenset(
    {
        "cryptorank.io",
        "coinmarketcap.com",
        "coingecko.com",
        "coincodex.com",
        "dexscreener.com",
        "dextools.io",
        "etherscan.io",
        "bscscan.com",
        "solscan.io",
        "tokenterminal.com",
        "defillama.com",
        "coincheckup.com",
        "cryptocompare.com",
    }
)

_FINANCE_DOMAINS = frozenset(
    {
        "stocktitan.net",
        "seekingalpha.com",
        "benzinga.com",
        "fool.com",
        "investorplace.com",
        "barchart.com",
        "finviz.com",
        "zacks.com",
        "tipranks.com",
    }
)

_COMPANY_DOMAINS = frozenset(
    {
        "zoominfo.com",
        "crunchbase.com",
        "pitchbook.com",
        "owler.com",
        "dnb.com",
        "apollo.io",
        "rocketreach.co",
        "lusha.com",
        "cognism.com",
    }
)

# ------------------------------------------------------------- Category Regexes
_JOB_PATH_RE = re.compile(
    r"/(?:jobs?|careers?|viewjob|q-[^/]+-jobs|salaries|hiring|openings?)(?:[/?#]|$)",
    re.IGNORECASE,
)

_JOB_TEXT_RE = re.compile(
    r"\b(?:now\s+hiring|hiring\s+now)\b"
    r"|\b\d+[\d,]*\s+(?:remote\s+)?jobs?\b"
    r"|\bjobs?\s+in\s+[a-z]+"
    r"|\b(?:salary\s+search|salaries\s+in\b)"
    r"|\b(?:full-time|part-time|direct-hire)\s+(?:position|role|job|contract)\b"
    r"|\bjob\s+(?:has\s+expired|posting|opening|description)\b"
    r"|\b(?:employer\s+is\s+not\s+accepting\s+applications|not\s+actively\s+hiring)\b"
    r"|\bapply\s+(?:now|for\s+this\s+job|online)\b"
    r"|\b(?:responsibilities|qualifications|benefits|compensation|salary\s+range)\s*:"
    r"|\bview\s+all\s+[\w\s'-]+jobs\b",
    re.IGNORECASE,
)

_CRYPTO_TEXT_RE = re.compile(
    r"\b(?:cryptocurrency\s+market|latest\s+cryptocurrency\s+news)\b"
    r"|\b(?:tokenized\s+stocks|prediction\s+markets|crypto\s+signals)\b"
    r"|\b(?:potential\s+airdrops|cryptocurrency\s+converter)\b"
    r"|\b(?:circulating\s+supply|24h\s+trading\s+volume|all-time\s+high|market\s+cap\s+rank)\b"
    r"|\b(?:token\s+price|coin\s+price|price\s+prediction|tokenomics)\b",
    re.IGNORECASE,
)

_FINANCE_TEXT_RE = re.compile(
    r"\b(?:stock\s+news|stock\s+market|press\s+release\s+wire|earnings\s+call\s+transcript"
    r"|quarterly\s+results|target\s+price|zacks\s+rank)\b"
    r"|\b(?:shares\s+(?:surged|fell|rose|plunged|jumped|tumbled)|market\s+watch)\b"
    r"|\b(?:NYSE|NASDAQ|OTCMKTS|LON)\s*:\s*[A-Z]+\b"
    r"|\([A-Z]{1,5}\s+stock\s+news\)",
    re.IGNORECASE,
)

_COMPANY_TEXT_RE = re.compile(
    r"\bcompany\s+details\b"
    r"|\b(?:headquarters|company\s+type|number\s+of\s+employees|annual\s+revenue)\s*:"
    r"|\bis\s+a\s+(?:software\s+development|digital|marketing|consulting)\s+agency\s+that\s+specializes\s+in\b"
    r"|\b(?:view\s+company\s+profile|business\s+profile|lead\s+intelligence)\b",
    re.IGNORECASE,
)

_SPAM_TEXT_RE = re.compile(
    r"\b(?:coupon\s+codes?|promo\s+codes?|discount\s+codes?|coupons?\s+and\s+promo)\b"
    r"|\b(?:free\s+download\s+crack|keygen|warez|full\s+torrent)\b"
    r"|\b(?:buy\s+this\s+domain|domain\s+is\s+for\s+sale|parked\s+free\s+courtesy)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JunkVerdict:
    """Outcome of junk source classification."""

    is_junk: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.is_junk


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _extract_intent_tokens(spec: JobSpec | None, aspect: Aspect | None) -> set[str]:
    tokens: set[str] = set()
    if spec is not None:
        text = f"{spec.description} {spec.motivation} {' '.join(spec.task_categories)}"
        tokens.update(_tokenize(text))
    if aspect is not None:
        text = f"{aspect.topic} {aspect.rationale} {' '.join(aspect.keywords)}"
        tokens.update(_tokenize(text))
    return tokens


def _matches_domain(host: str, domains: frozenset[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_junk_url(url: str, *, keywords: tuple[str, ...] = ()) -> bool:
    """Lightweight check for web crawling outlinks.

    Used by crawler policy to avoid following links into careers/jobs portals.
    """
    kw_tokens = set(_WORD_RE.findall(" ".join(keywords).lower()))
    if kw_tokens & _JOB_INTENT_TERMS:
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if _matches_domain(host, _JOB_DOMAINS):
        return True
    if host.startswith(("jobs.", "careers.", "hiring.", "boards.greenhouse.")):
        return True
    if _JOB_PATH_RE.search(parsed.path):
        return True
    return False


def is_junk_source(
    url: str,
    *,
    title: str = "",
    snippet: str = "",
    spec: JobSpec | None = None,
    aspect: Aspect | None = None,
) -> JunkVerdict:
    """Evaluate whether a candidate source URL and its snippet represent junk grounding.

    Returns a :class:`JunkVerdict`. When the goal is legitimately about the category
    (e.g. tech recruitment, cryptocurrency, stock markets), that category is exempt.
    """
    intent_tokens = _extract_intent_tokens(spec, aspect)

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    combined_text = f"{title} {snippet}"

    # 1. Spam / coupon / parked domain check (always active)
    if _SPAM_TEXT_RE.search(combined_text):
        return JunkVerdict(True, "spam / promotional / parked domain")

    # 2. Job boards & recruitment listings
    if not (intent_tokens & _JOB_INTENT_TERMS):
        is_job_host = _matches_domain(host, _JOB_DOMAINS) or host.startswith(
            ("jobs.", "careers.", "hiring.", "boards.greenhouse.")
        )
        is_linkedin_job = "linkedin.com" in host and (
            path.startswith("/jobs") or path.startswith("/job")
        )
        is_job_path = bool(_JOB_PATH_RE.search(path))
        is_job_content = bool(_JOB_TEXT_RE.search(combined_text))

        if is_job_host or is_linkedin_job:
            return JunkVerdict(True, "job board / recruitment domain")
        if is_job_path and (is_job_content or "job" in path or "career" in path):
            return JunkVerdict(True, "job recruitment listing path")
        if is_job_content and (
            "job" in combined_text.lower()
            or "hiring" in combined_text.lower()
            or "salary" in combined_text.lower()
        ):
            return JunkVerdict(True, "job recruitment listing text")

    # 3. Cryptocurrency speculation & coin price trackers
    if not (intent_tokens & _CRYPTO_INTENT_TERMS):
        is_crypto_host = _matches_domain(host, _CRYPTO_DOMAINS)
        is_crypto_content = bool(_CRYPTO_TEXT_RE.search(combined_text))

        if is_crypto_host:
            return JunkVerdict(True, "cryptocurrency tracker / market domain")
        if is_crypto_content:
            return JunkVerdict(True, "cryptocurrency tracker / market speculation")

    # 4. Stock tickers & financial PR wires
    if not (intent_tokens & _FINANCE_INTENT_TERMS):
        is_finance_host = _matches_domain(host, _FINANCE_DOMAINS)
        is_finance_content = bool(_FINANCE_TEXT_RE.search(combined_text))

        if is_finance_host:
            return JunkVerdict(True, "stock ticker / financial news domain")
        if is_finance_content:
            return JunkVerdict(True, "stock ticker / financial news wire")

    # 5. Business directories & company profile databases
    if not (intent_tokens & _COMPANY_INTENT_TERMS):
        is_company_host = _matches_domain(host, _COMPANY_DOMAINS)
        is_company_content = bool(_COMPANY_TEXT_RE.search(combined_text))

        if is_company_host:
            return JunkVerdict(True, "business directory / company database domain")
        if is_company_content:
            return JunkVerdict(True, "business directory / company profile listing")

    return JunkVerdict(False, "")
