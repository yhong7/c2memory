from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

@dataclass(frozen=True)
class WikipediaServiceSettings:
    base_url: str = os.getenv("WIKI_SERVICE_URL", "")
    default_top_k: int = int(os.getenv("WIKI_SERVICE_TOP_K", "10"))
    timeout_seconds: float = float(os.getenv("WIKI_SERVICE_TIMEOUT_SECONDS", "120"))


@dataclass(frozen=True)
class WikipediaHit:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float | None = None

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", "")).strip()


@dataclass(frozen=True)
class WikipediaLookupResult:
    title: str
    found: bool
    document: str | None
    metadata: dict[str, list[str]] | None
    lookup_seconds: float


class WikipediaServiceClient:
    def __init__(self, settings: WikipediaServiceSettings | None = None) -> None:
        self.settings = settings or WikipediaServiceSettings()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.base_url.strip():
            raise RuntimeError(
                "WIKI_SERVICE_URL must point to a deployed Wikipedia retrieval service"
            )
        response = requests.post(
            f"{self.settings.base_url.rstrip('/')}{path}",
            json=payload,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        return dict(response.json())

    def search(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        k = top_k or self.settings.default_top_k
        return self._post("/search", {"query": query, "top_k": k})

    def get_hits(self, query: str, *, top_k: int | None = None) -> list[WikipediaHit]:
        payload = self.search(query, top_k=top_k)
        hits: list[WikipediaHit] = []
        for raw in payload.get("hits", []):
            hits.append(
                WikipediaHit(
                    id=str(raw.get("id", "")),
                    document=str(raw.get("document", "")),
                    metadata=dict(raw.get("metadata", {}) or {}),
                    distance=raw.get("distance"),
                )
            )
        return hits

    def search_titles(self, query: str, *, top_k: int | None = None) -> list[str]:
        return [hit.title for hit in self.get_hits(query, top_k=top_k)]

    def lookup_title(self, title: str) -> WikipediaLookupResult:
        payload = self._post("/lookup", {"title": title})
        return WikipediaLookupResult(
            title=str(payload.get("title", title)),
            found=bool(payload.get("found", False)),
            document=payload.get("document"),
            metadata=dict(payload.get("metadata", {}) or {}) if payload.get("metadata") else None,
            lookup_seconds=float(payload.get("lookup_seconds", 0.0) or 0.0),
        )
