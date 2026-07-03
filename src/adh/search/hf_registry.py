"""Hugging Face Hub dataset registry adapter (Phase 1.6, the ``datasets`` channel).

Reuse-first: searching HF Hub for an existing, licensed dataset beats re-scraping the
web (OUTLINE §1.6, RESEARCH §DISCOVERY). Uses ``huggingface_hub`` (an optional extra:
``pip install 'adh[datasets]'``) and is imported lazily so the core harness has no
hard dependency on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .provider import DatasetHit


class HfHubUnavailableError(RuntimeError):
    """Raised when the ``datasets`` extra (huggingface_hub) is not installed."""


def _require_hf():
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise HfHubUnavailableError(
            "huggingface_hub is required for the HF datasets channel; "
            "install with: pip install 'adh[datasets]'"
        ) from exc
    return HfApi


@dataclass
class HfDatasetRegistry:
    """Live HF Hub dataset search + sampling.

    ``api`` may be injected (e.g. a stub) for testing; otherwise an ``HfApi`` is
    created lazily on first use.
    """

    api: object | None = None
    token: str | None = None

    def _client(self):
        if self.api is None:
            HfApi = _require_hf()
            self.api = HfApi(token=self.token)
        return self.api

    def search(self, query: str, *, limit: int = 10) -> list[DatasetHit]:
        api = self._client()
        results = api.list_datasets(search=query, limit=limit, sort="downloads")
        hits: list[DatasetHit] = []
        for ds in results:
            ds_id = getattr(ds, "id", None) or getattr(ds, "id_or_path", "")
            card = getattr(ds, "card_data", None) or {}
            license_val = None
            if isinstance(card, dict):
                license_val = card.get("license")
            license_val = license_val or getattr(ds, "license", None)
            hits.append(
                DatasetHit(
                    dataset_id=ds_id,
                    locator=f"https://huggingface.co/datasets/{ds_id}",
                    title=ds_id,
                    description=" ".join(getattr(ds, "tags", []) or []),
                    license=license_val,
                    downloads=getattr(ds, "downloads", 0) or 0,
                    last_modified=str(getattr(ds, "last_modified", "") or "") or None,
                )
            )
        return hits

    def sample(self, dataset_id: str, *, max_chars: int = 2000) -> str:
        """Best-effort: fetch the dataset card README as the shape sample."""
        self._client()  # ensure the extra is installed before attempting a download
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            path = hf_hub_download(repo_id=dataset_id, filename="README.md", repo_type="dataset")
            with open(path, encoding="utf-8") as fh:
                return fh.read()[:max_chars]
        except Exception:  # pragma: no cover - network/availability dependent
            return ""
