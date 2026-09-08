"""Polite + SSRF-guarded HTTP fetching for open-internet media acquisition.

One module answers "how does ADH touch a URL it did not choose?" — SERP results
and harvested asset links are attacker-influenceable, so every request passes:

* **SSRF policy** (:class:`SSRFPolicy`) — scheme allowlist, IDNA encoding, DNS
  resolution, and rejection of private/loopback/link-local (incl. cloud IMDS),
  reserved, multicast, and otherwise non-global addresses. Redirect hops are
  re-validated from scratch, and the connected peer IP is re-checked against
  the validated set before any body byte is read (closes the DNS-rebinding
  TOCTOU window as well as the transport allows without monkeypatching).
* **Politeness** (:class:`PoliteFetcher`) — cached ``robots.txt`` per host with
  crawl-delay honored (capped), unreachable robots degrading loudly to
  allow-all, and per-netloc spacing between requests.
* **Bounded reads** — HTML bodies capped at a char budget; asset streams
  content-type-checked, refused upfront when ``Content-Length`` exceeds the
  budget, and hard-aborted when the body lies about its size.

The HTTP client is injected (``client_factory``) and MUST be built with
redirects disabled — this module walks redirects itself so every hop is
gated. Offline tests inject fakes for the client, resolver, clock, sleeper,
and robots loader; production wires httpx.
"""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import socket
import time
import urllib.request
import urllib.robotparser
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_REDIRECTS = 5
_CHUNK_BYTES = 64 * 1024

#: Content-type prefixes an asset download may serve. Anything else (HTML error
#: pages, JSON APIs) is rejected before bytes move.
DEFAULT_ASSET_CONTENT_TYPES: tuple[str, ...] = (
    "image/",
    "video/",
    "audio/",
    "application/octet-stream",
    "binary/octet-stream",
)


class UrlBlocked(ValueError):
    """A URL was refused by the SSRF guard or robots.txt — never retried."""


class OversizedAssetError(RuntimeError):
    """The asset exceeds its download budget (declared or actual)."""


class UnsupportedContentType(ValueError):
    """The response is not an asset stream (HTML page, API payload, …)."""


def is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for non-globally-routable addresses.

    Rejects private (RFC1918/ULA), loopback, link-local (including the cloud
    IMDS at 169.254.169.254), reserved, multicast, unspecified, and everything
    ``ipaddress`` does not consider global. IPv4-mapped IPv6 and 6to4 wrappers
    are unwrapped first so tunneled private space is still caught.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def resolve_hostname(hostname: str) -> list[str]:
    """Resolve a hostname to its address strings (the injectable DNS seam)."""
    infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    return [str(info[4][0]) for info in infos]


@dataclass(frozen=True)
class SSRFPolicy:
    """Validate URLs against scheme/DNS/IP policy before any connection."""

    #: hostname → IP strings seam; defaults to real :func:`resolve_hostname`.
    resolver: Callable[[str], list[str]] | None = None

    def validate(self, url: str) -> tuple[str, list[str]]:
        """Return ``(idna_hostname, validated_ips)`` or raise :class:`UrlBlocked`."""
        parts = urlsplit(url.strip())
        if parts.scheme not in _ALLOWED_SCHEMES:
            raise UrlBlocked(f"URL scheme not allowed: {parts.scheme!r} (must be http or https)")
        hostname = parts.hostname
        if not hostname:
            raise UrlBlocked("URL is missing a hostname")
        try:
            encoded = hostname.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise UrlBlocked(f"could not encode hostname {hostname!r} as IDNA") from exc
        try:
            raw_ips = (self.resolver or resolve_hostname)(encoded)
        except socket.gaierror as exc:
            raise UrlBlocked(f"could not resolve hostname {encoded!r}: {exc}") from exc
        validated: list[str] = []
        for raw in raw_ips:
            try:
                ip = ipaddress.ip_address(raw.split("%", 1)[0])
            except ValueError as exc:
                raise UrlBlocked(f"hostname {hostname!r} resolved to malformed {raw!r}") from exc
            if is_blocked_ip(ip):
                logger.warning("SSRF guard blocked %r: %s resolves to %s", url, hostname, ip)
                raise UrlBlocked(f"URL hostname {hostname!r} resolves to blocked address {ip}")
            validated.append(str(ip))
        if not validated:
            raise UrlBlocked(f"hostname {hostname!r} resolved to no usable addresses")
        return encoded, validated


def _fetch_robots_text(netloc: str) -> str | None:
    """Best-effort robots.txt fetch for the production loader seam."""
    for scheme in ("https", "http"):
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme pinned above
                f"{scheme}://{netloc}/robots.txt", timeout=5
            ) as resp:
                return resp.read(256 * 1024).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - unreachable robots must degrade, not raise
            continue
    return None


def _default_httpx_client():
    """Live httpx transport; redirects DISABLED (per-hop revalidation owns
    them). Lazy import keeps the core stdlib-only — without the media extra,
    this fails loudly exactly where a live fetch would have happened."""
    import httpx

    return httpx.Client(follow_redirects=False, timeout=httpx.Timeout(30.0))


@dataclass
class PoliteFetcher:
    """The one way acquisition code touches foreign URLs.

    ``fetch_html`` returns ``None`` for any politely-skipped page (robots,
    non-HTML, HTTP errors) — page-level failures are soft. ``stream`` raises
    typed errors instead, because a failed asset download is a concrete,
    countable event for the caller.
    """

    #: httpx.Client-like transport factory; redirects DISABLED. Defaults to a
    #: lazy httpx client so wiring needs no import gymnastics (see above).
    client_factory: Callable[[], Any] = field(default_factory=lambda: _default_httpx_client())
    policy: SSRFPolicy = field(default_factory=SSRFPolicy)
    min_interval_s: float = 1.0
    max_crawl_delay_s: float = 10.0
    max_chars: int = 2_000_000
    user_agent: str = "Mozilla/5.0 (compatible; ADH/0.1; open-dataset-acquisition)"
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    #: netloc → robots.txt text (None = unreachable); defaults to live fetch.
    robots_loader: Callable[[str], str | None] | None = None
    _last_request_at: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------ html
    def fetch_html(self, url: str, *, max_chars: int | None = None) -> str | None:
        """GET ``url`` through every gate; the page text or None when skipped."""
        client = self.client_factory()
        current = url
        budget = self.max_chars if max_chars is None else max_chars
        for _hop in range(_MAX_REDIRECTS + 1):
            host, ips = self.policy.validate(current)  # UrlBlocked propagates
            if not self._robots_allow(current):
                logger.warning("skipping %s: robots.txt disallows this path", current)
                return None
            self._wait_turn(urlsplit(current).netloc.lower())
            response = client.get(current, headers={"User-Agent": self.user_agent})
            self._check_peer(response, host, ips)
            status = getattr(response, "status_code", 0)
            location = response.headers.get("location") if hasattr(response, "headers") else None
            if 300 <= status < 400:  # noqa: PLR2004 - HTTP redirect class
                if not location:
                    logger.warning("skipping %s: redirect %s lacks Location", current, status)
                    return None
                current = urljoin(current, location)
                continue
            if status >= 400:  # noqa: PLR2004 - HTTP error class
                logger.warning("skipping %s: HTTP %s", current, status)
                return None
            ctype = ""
            if hasattr(response, "headers"):
                ctype = (response.headers.get("content-type") or "").strip().lower()
            if not ctype.startswith("text/html"):
                logger.warning(
                    "skipping %s: content-type %r is not HTML", current, ctype or "unknown"
                )
                return None
            return self._read_capped(response, current, budget)
        logger.warning("skipping %s: exceeded %d redirects", url, _MAX_REDIRECTS)
        return None

    # ---------------------------------------------------------------- stream
    def stream(
        self,
        url: str,
        *,
        max_bytes: int,
        content_type_prefixes: tuple[str, ...] = DEFAULT_ASSET_CONTENT_TYPES,
    ) -> Iterator[bytes]:
        """Preflight every gate, then yield the asset in chunks.

        Raises :class:`UrlBlocked` / :class:`UnsupportedContentType` /
        :class:`OversizedAssetError` / OSError at call time whenever the gates
        can decide before bytes move; mid-stream size lies abort iteration.
        """
        host, ips = self.policy.validate(url)
        if not self._robots_allow(url):
            raise UrlBlocked(f"robots.txt disallows {url!r}")
        self._wait_turn(urlsplit(url).netloc.lower())
        client = self.client_factory()
        response = client.get(url, headers={"User-Agent": self.user_agent})
        self._check_peer(response, host, ips)
        status = getattr(response, "status_code", 0)
        if status >= 400:  # noqa: PLR2004 - HTTP error class
            raise OSError(f"HTTP {status} fetching {url!r}")
        ctype = ""
        if hasattr(response, "headers"):
            ctype = (response.headers.get("content-type") or "").strip().lower()
        if content_type_prefixes and not ctype.startswith(content_type_prefixes):
            raise UnsupportedContentType(
                f"{url!r} serves {ctype or 'unknown'!r}; not an asset stream"
            )
        length = ""
        if hasattr(response, "headers"):
            length = response.headers.get("content-length") or ""
        if length.isdigit() and int(length) > max_bytes:
            raise OversizedAssetError(
                f"{url!r} declares {length} bytes > the {max_bytes}-byte budget"
            )
        return self._chunks(response, url, max_bytes)

    def _chunks(self, response: Any, url: str, max_bytes: int) -> Iterator[bytes]:
        total = 0
        for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
            total += len(chunk)
            if total > max_bytes:
                raise OversizedAssetError(
                    f"{url!r} streamed past the {max_bytes}-byte budget; aborted"
                )
            yield chunk

    # ---------------------------------------------------------------- gates
    def _robots_allow(self, url: str) -> bool:
        netloc = urlsplit(url).netloc.lower()
        parser = self._robots(netloc)
        return parser is None or parser.can_fetch("*", url)

    def _robots(self, netloc: str) -> urllib.robotparser.RobotFileParser | None:
        """Cached parser; ``None`` means unreachable → allow-all (warned once)."""
        if netloc in self._robots_cache:
            return self._robots_cache[netloc]
        loader = self.robots_loader or _fetch_robots_text
        try:
            text = loader(netloc)
        except Exception as exc:  # noqa: BLE001 - loader faults degrade to allow-all
            logger.debug("robots loader failed for %s: %s", netloc, exc)
            text = None
        parser: urllib.robotparser.RobotFileParser | None = None
        if text:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(text.splitlines())
        else:
            logger.warning(
                "robots.txt unreachable for %s — proceeding with allow-all "
                "(politeness degraded, SSRF guards unchanged)",
                netloc,
            )
        self._robots_cache[netloc] = parser
        return parser

    def _wait_turn(
        self, netloc: str, parser: urllib.robotparser.RobotFileParser | None = None
    ) -> None:
        """Per-netloc spacing: max(min_interval, crawl-delay capped)."""
        if parser is None:
            parser = self._robots(netloc)
        delay = self.min_interval_s
        if parser is not None:
            crawl = parser.crawl_delay("*")
            if crawl:
                delay = max(delay, min(float(crawl), self.max_crawl_delay_s))
        last = self._last_request_at.get(netloc)
        now = self.clock()
        if last is not None and now - last < delay:
            self.sleeper(delay - (now - last))
        self._last_request_at[netloc] = self.clock()

    def _check_peer(self, response: Any, host: str, ips: list[str]) -> None:
        """Re-check the connected peer against the validated set (rebinding).

        Best-effort: transports that don't expose the peer name pass through;
        ones that do (httpx/httpcore streams) get a hard post-connect check so
        a rebinding DNS switch is caught before any body byte is consumed.
        """
        peername = None
        with contextlib.suppress(Exception):
            network_stream = response.extensions["network_stream"]
            peername = network_stream.get_extra_info("peername")
        if not peername:
            return
        try:
            peer_ip = ipaddress.ip_address(str(peername[0]).split("%", 1)[0])
        except ValueError:
            raise UrlBlocked(f"peer reported malformed address {peername!r}") from None
        allowed = {ipaddress.ip_address(raw.split("%", 1)[0]) for raw in ips}
        if is_blocked_ip(peer_ip) or peer_ip not in allowed:
            logger.warning(
                "SSRF guard: %s connected to %s but DNS validated %s — blocking",
                host,
                peer_ip,
                sorted(map(str, allowed)),
            )
            raise UrlBlocked(
                f"connected peer {peer_ip} is not among the validated addresses for {host!r}"
            )

    @staticmethod
    def _read_capped(response: Any, url: str, max_chars: int) -> str:
        buf = bytearray()
        truncated = False
        for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
            buf.extend(chunk)
            if len(buf) >= max_chars:
                truncated = True
                break
        if truncated:
            logger.warning("HTML body from %s truncated at %d chars", url, max_chars)
        return bytes(buf[:max_chars]).decode("utf-8", errors="replace")
