"""Direct-web media provider — SERP pages → polite GET → stdlib HTML harvest.

The no-platform-dependence tier: any page a browser can show becomes a source.
A :class:`~adh.search.SearchProvider` supplies candidate pages; each is fetched
through the :class:`~adh.net.PoliteFetcher` (SSRF guards, robots, spacing) and
parsed by a stdlib :class:`html.parser.HTMLParser` for ``<img>``/``<video>``/
``<source>``/``og:*``/media-extension links. Everything harvested carries an
honest ``"unknown"`` license — the spec's two-layer policy decides whether it
may enter, and the publish gate keeps it off public uploads unless overridden.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

from adh.search import SearchProvider

from .provider import MediaHit

if TYPE_CHECKING:
    from adh.net.httpfetch import PoliteFetcher

logger = logging.getLogger(__name__)

#: Extension → coarse media type. Absent extensions (.avif …) are skipped:
#: the probe verifier must be able to fail-closed on what we stage.
_EXT_TYPE: dict[str, str] = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".gif": "gif",
    ".mp4": "video",
    ".m4v": "video",
    ".mov": "video",
    ".webm": "video",
}

_SKIP_PREFIXES = ("data:", "javascript:", "blob:")

#: Recognized image containers we cannot stage (no probe support today):
#: skipping them outright beats mislabeling them as storable "image" hits.
_UNSTAGEABLE_EXTS = frozenset({".avif", ".svg"})


def _ext_type(url_path: str) -> str | None:
    tail = url_path.rsplit("/", 1)[-1]
    if "." not in tail:
        return None
    return _EXT_TYPE.get("." + tail.rsplit(".", 1)[1].lower())


def _typed(url: str, default: str) -> str | None:
    """Extension-derived type, else the tag's contextual default.

    A recognized-but-unstorable container (.avif/.svg) yields ``None`` so the
    caller skips the candidate entirely instead of guessing it storable.
    """
    tail = urlsplit(url.strip()).path.rsplit("/", 1)[-1].lower()
    if "." not in tail:
        return default
    ext = f".{tail.rsplit('.', 1)[1]}"
    if ext in _UNSTAGEABLE_EXTS:
        return None
    return _EXT_TYPE.get(ext, default)


def _srcset_candidates(value: str) -> list[tuple[str, int]]:
    """``srcset`` → (url, declared-width) pairs; width 0 when undeclared."""
    out: list[tuple[str, int]] = []
    for part in value.split(","):
        bits = part.strip().split()
        if not bits or not bits[0]:
            continue
        width = 0
        for token in bits[1:]:
            if token.endswith("w") and token[:-1].isdigit():
                width = int(token[:-1])
        out.append((bits[0], width))
    return out


class _HtmlMediaExtractor(HTMLParser):
    """Harvest absolute asset URLs + their inferred type from one page."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.candidates: list[tuple[str, str]] = []  # (absolute url, type)
        self._alts: dict[str, str] = {}
        self._in_title = False

    # ----------------------------------------------------------------- tags
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "img":
            self._add_img(a)
        elif tag == "video":
            if a.get("poster"):
                self._add(a["poster"], _typed(a["poster"], "image"))
            if a.get("src"):
                self._add(a["src"], _typed(a["src"], "video"))
        elif tag == "source":
            if a.get("src"):
                self._add(a["src"], _typed(a["src"], "video"))
        elif tag == "meta":
            prop = a.get("property", "").lower()
            content = a.get("content", "")
            if prop == "og:image" and content:
                self._add(content, "image")
            elif prop in {"og:video", "og:video:url", "og:video:secure_url"} and content:
                self._add(content, "video")
        elif tag == "a":
            href = a.get("href", "")
            kind = _ext_type(href)
            if kind:
                self._add(href, kind)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()

    # --------------------------------------------------------------- helpers
    def _add_img(self, attrs: dict[str, str]) -> None:
        srcset = attrs.get("srcset", "")
        if srcset.strip():
            candidates = _srcset_candidates(srcset)
            if candidates:
                best_url, _width = max(candidates, key=lambda c: c[1])
                self._add(best_url, _typed(best_url, "image"), alt=attrs.get("alt"))
                return  # srcset supersedes src
        src = attrs.get("src", "")
        if src:
            self._add(src, _typed(src, "image"), alt=attrs.get("alt"))

    def _add(self, raw_url: str, media_type: str | None, alt: str | None = None) -> None:
        if media_type is None:  # unstorable container: skip rather than guess
            return
        url = raw_url.strip()
        if not url or url.lower().startswith(_SKIP_PREFIXES):
            return
        absolute = urljoin(self.base_url, url)
        if not absolute.startswith(("http://", "https://")):
            return
        if absolute not in self._alts:  # first sighting wins the alt text
            self.candidates.append((absolute, media_type))
            self._alts[absolute] = (alt or "").strip()

    def alt_for(self, absolute_url: str) -> str:
        return self._alts.get(absolute_url, "")


@dataclass
class DirectWebMediaProvider:
    """Open-web harvest behind the standard :class:`MediaProvider` port."""

    search_provider: SearchProvider
    fetcher: PoliteFetcher
    supported_media_types: frozenset[str] = frozenset({"image", "gif", "video"})
    #: per-asset download budget handed to the fetcher's stream gate.
    max_asset_bytes: int = 512 * 1024 * 1024
    _state: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def search(self, query: str, *, limit: int, media_types=None) -> list[MediaHit]:
        wanted = frozenset(media_types) if media_types else None
        out: list[MediaHit] = []
        seen_urls: set[str] = set()
        for serp_hit in self.search_provider.search(query, limit=max(limit, 1)):
            page_url = serp_hit.url
            try:
                html = self.fetcher.fetch_html(page_url)
            except Exception as exc:  # noqa: BLE001 - one dead page ≠ dead acquisition
                logger.warning("direct-web: page %s failed (%s) — skipping", page_url, exc)
                continue
            if not html:
                logger.warning(
                    "direct-web: page %s unavailable (blocked/non-html/error) — skipping",
                    page_url,
                )
                continue
            extractor = _HtmlMediaExtractor(page_url)
            try:
                extractor.feed(html)
                extractor.close()
            except Exception as exc:  # noqa: BLE001 - hostile HTML must not crash acquisition
                logger.warning("direct-web: could not parse %s (%s)", page_url, exc)
                continue
            attribution = extractor.title or urlsplit(page_url).netloc
            for asset_url, media_type in extractor.candidates:
                if wanted is not None and media_type not in wanted:
                    continue
                if asset_url in seen_urls:
                    continue
                seen_urls.add(asset_url)
                alt = extractor.alt_for(asset_url)
                out.append(
                    MediaHit(
                        media_id=hashlib.sha256(asset_url.encode()).hexdigest()[:16],
                        page_url=page_url,
                        download_url=asset_url,
                        title=(alt or extractor.title or "")[:200],
                        tags=(alt,) if alt else (),
                        license="unknown",  # open web declares nothing; never invent
                        attribution=f"{attribution} ({page_url})",
                        media_type=media_type,
                    )
                )
                if len(out) >= limit:
                    return out
        return out

    def download(self, hit: MediaHit):
        """Stream the asset through the polite fetcher's bounded stream."""
        return self.fetcher.stream(hit.download_url, max_bytes=self.max_asset_bytes)
