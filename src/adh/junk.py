"""Goal-aware junk-source classification for discovery and the crawl frontier.

Junk = pages that rank on a topic's keywords but carry no teachable content about
it: job boards, crypto price trackers, stock-ticker news wires, company-profile
databases, and spam. Live run rust-async-live-small grounded generation on such
pages (Indeed listings, CryptoRank, StockTitan, ZoomInfo).

Evidence model:

* A known category domain (indeed.com, cryptorank.io, ...) is decisive on its own.
* Path and text patterns are evidence. *Strong* signals are phrasings that only
  appear on such pages ("apply now", "circulating supply", "(NASDAQ: WIMI)");
  *weak* signals are vocabulary shared with technical writing (a ``/jobs/`` path,
  "8 jobs", "stock market"). Weak signals corroborate but never convict: a page is
  junk only with a strong signal, and the job category, whose vocabulary overlaps
  most with CI, schedulers and orchestrators, needs a second signal on top.
* A category stands down when the goal (spec or aspect) is about it, so a
  recruitment dataset may still crawl job boards. Spam never stands down.

Lives outside :mod:`adh.discovery` because the web crawler's frontier policy uses
it too, and a channel must not depend on the discovery package.
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
_I = re.IGNORECASE


@dataclass(frozen=True)
class JunkVerdict:
    """Outcome of junk source classification; truthy when the source is junk."""

    is_junk: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.is_junk


@dataclass(frozen=True)
class _Category:
    domain_reason: str
    text_reason: str
    # Words or phrases meaning the goal is about this category (category exempt).
    intent: frozenset[str] = frozenset()
    domains: frozenset[str] = frozenset()
    host_prefixes: tuple[str, ...] = ()
    host_paths: tuple[tuple[str, str], ...] = ()  # (host suffix, path prefix)
    strong: tuple[re.Pattern[str], ...] = ()
    weak: tuple[re.Pattern[str], ...] = ()
    weak_path: re.Pattern[str] | None = None
    min_signals: int = 1  # strong + weak signals required (at least one strong)
    # Outlink paths the crawler never follows (only where no technical reading exists).
    crawl_block_path: re.Pattern[str] | None = None

    def owns_host(self, host: str, path: str) -> bool:
        if _matches_domain(host, self.domains) or host.startswith(self.host_prefixes):
            return True
        return any(_matches_domain(host, {h}) and path.startswith(p) for h, p in self.host_paths)

    def text_signals(self, path: str, text: str) -> tuple[int, int]:
        strong = sum(bool(p.search(text)) for p in self.strong)
        weak = sum(bool(p.search(text)) for p in self.weak)
        if self.weak_path is not None and self.weak_path.search(path):
            weak += 1
        return strong, weak


_SPAM = _Category(
    domain_reason="spam / promotional / parked domain",
    text_reason="spam / promotional / parked domain",
    strong=(
        re.compile(r"\b(?:coupon|promo|discount)\s+codes?\b|\bcoupons?\s+and\s+promo\b", _I),
        # "ssh-keygen" is a tool, not warez: keygen must stand alone.
        re.compile(
            r"\bfree\s+download\s+crack\b|(?<![\w-])keygen\b|\bwarez\b|\bfull\s+torrent\b", _I
        ),
        re.compile(r"\bbuy\s+this\s+domain\b|\bdomain\s+is\s+for\s+sale\b|\bparked\s+free\b", _I),
    ),
)

_JOB = _Category(
    domain_reason="job board / recruitment domain",
    text_reason="job recruitment listing",
    intent=frozenset(
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
            "job market",
            "labor market",
        }
    ),
    domains=frozenset(
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
    ),
    host_prefixes=("jobs.", "careers.", "hiring.", "boards.greenhouse."),
    host_paths=(("linkedin.com", "/jobs"),),
    strong=(
        re.compile(r"\bnow\s+hiring\b|\bhiring\s+now\b", _I),
        re.compile(r"\bsalary\s+search\b|\bsalaries\s+in\b|\bsalary\s+range\b", _I),
        re.compile(r"\b(?:full|part)-time\s+(?:position|role|job|contract)\b", _I),
        re.compile(r"\bdirect-hire\b", _I),
        re.compile(r"\bjob\s+(?:has\s+expired|posting|opening)s?\b", _I),
        re.compile(r"\bnot\s+accepting\s+applications\b|\bnot\s+actively\s+hiring\b", _I),
        re.compile(r"\bapply\s+(?:now|for\s+this\s+job|online)\b", _I),
        re.compile(r"\bview\s+all\s+[\w\s'-]+\bjobs\b", _I),
    ),
    weak=(
        re.compile(r"\b\d[\d,]*\s+(?:remote\s+)?jobs?\b", _I),
        re.compile(r"\bjobs?\s+in\s+[a-z]+", _I),
        re.compile(r"\b(?:responsibilities|qualifications|benefits|compensation)\s*:", _I),
    ),
    weak_path=re.compile(
        r"/(?:jobs?|careers?|viewjob|q-[^/]+-jobs[^/]*|salaries|hiring|openings?)(?:[/?#.]|$)",
        _I,
    ),
    min_signals=2,
    crawl_block_path=re.compile(r"/careers?(?:[/?#]|$)", _I),
)

_CRYPTO = _Category(
    domain_reason="cryptocurrency tracker / market domain",
    text_reason="cryptocurrency tracker / market speculation",
    # Not "token(s)": LLM tokenizer goals say it too, and must keep crypto filtering.
    intent=frozenset(
        {
            "crypto",
            "cryptocurrency",
            "cryptocurrencies",
            "bitcoin",
            "ethereum",
            "blockchain",
            "blockchains",
            "web3",
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
            "stablecoin",
            "stablecoins",
            "dao",
            "daos",
        }
    ),
    domains=frozenset(
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
    ),
    strong=(
        re.compile(r"\bcryptocurrency\s+(?:market|news|converter)\b|\bcrypto\s+signals\b", _I),
        re.compile(r"\btokenized\s+stocks\b|\bpotential\s+airdrops\b|\btokenomics\b", _I),
        re.compile(r"\bcirculating\s+supply\b|\b24h\s+trading\s+volume\b", _I),
        re.compile(r"\bmarket\s+cap\s+rank\b|\bcoin\s+price\b", _I),
    ),
    weak=(
        re.compile(r"\ball-time\s+high\b|\bprediction\s+markets?\b", _I),
        re.compile(r"\bprice\s+prediction\b|\btoken\s+price\b", _I),  # also ML / LLM-pricing talk
    ),
)

_FINANCE = _Category(
    domain_reason="stock ticker / financial news domain",
    text_reason="stock ticker / financial news wire",
    intent=frozenset(
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
            "stock market",
        }
    ),
    domains=frozenset(
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
    ),
    strong=(
        # Exchange-qualified tickers are upper case by convention: case-sensitive.
        re.compile(r"\b(?:NYSE|NASDAQ|OTCMKTS|LSE)\s*:\s*[A-Z]{1,5}\b"),
        re.compile(r"\([A-Z]{1,5}\s+[Ss]tock\s+[Nn]ews\)"),
        re.compile(r"\bpress\s+release\s+wire\b|\bearnings\s+call\s+transcript\b", _I),
        re.compile(r"\bzacks\s+rank\b", _I),
        re.compile(r"\bshares\s+(?:surged|fell|rose|plunged|jumped|tumbled)\b", _I),
    ),
    weak=(
        re.compile(r"\bstock\s+(?:news|market)\b", _I),  # ML tutorials use stock data
        re.compile(r"\bquarterly\s+results\b|\btarget\s+price\b|\bmarket\s+watch\b", _I),
    ),
)

_COMPANY = _Category(
    domain_reason="business directory / company database domain",
    text_reason="business directory / company profile listing",
    intent=frozenset(
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
    ),
    domains=frozenset(
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
    ),
    strong=(
        re.compile(r"\bview\s+company\s+profile\b|\blead\s+intelligence\b", _I),
        re.compile(
            r"\b(?:headquarters|company\s+type|number\s+of\s+employees|annual\s+revenue)\s*:", _I
        ),
    ),
    weak=(
        re.compile(r"\bcompany\s+details\b|\bbusiness\s+profile\b", _I),
        re.compile(r"\bis\s+a\s+[\w\s-]{0,40}\bagency\s+that\s+specializes\s+in\b", _I),
    ),
)

# Spam first: it is never exempt, so its verdict wins over a category exemption.
_CATEGORIES: tuple[_Category, ...] = (_SPAM, _JOB, _CRYPTO, _FINANCE, _COMPANY)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _goal_mentions(goal: str, terms: frozenset[str]) -> bool:
    """Whole-word / whole-phrase match of ``terms`` against normalized ``goal``."""
    padded = f" {goal} "
    return any(f" {term} " in padded for term in terms)


def _goal_text(spec: JobSpec | None, aspect: Aspect | None) -> str:
    parts: list[str] = []
    if spec is not None:
        parts += [spec.description, spec.motivation, *spec.task_categories]
    if aspect is not None:
        parts += [aspect.topic, aspect.rationale, *aspect.keywords]
    return _normalize(" ".join(parts))


def _matches_domain(host: str, domains: frozenset[str] | set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_junk_url(url: str, *, keywords: tuple[str, ...] = ()) -> bool:
    """URL-only check for crawl-frontier outlinks (no title or snippet to weigh).

    Blocks known junk domains and career pages only. A ``/jobs/`` path alone is
    not enough: CI, Kubernetes and scheduler docs live under such paths.
    """
    goal = _normalize(" ".join(keywords))
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    for category in _CATEGORIES:
        if _goal_mentions(goal, category.intent):
            continue
        if category.owns_host(host, path):
            return True
        if category.crawl_block_path is not None and category.crawl_block_path.search(path):
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
    """Classify a candidate web source from its URL, title and snippet.

    Returns a truthy :class:`JunkVerdict` with a reason when the source is junk for
    this goal; categories the goal is about are exempt (spam never is).
    """
    goal = _goal_text(spec, aspect)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    text = f"{title} {snippet}"
    for category in _CATEGORIES:
        if _goal_mentions(goal, category.intent):
            continue
        if category.owns_host(host, path):
            return JunkVerdict(True, category.domain_reason)
        strong, weak = category.text_signals(path, text)
        if strong and strong + weak >= category.min_signals:
            return JunkVerdict(True, category.text_reason)
    return JunkVerdict(False)
