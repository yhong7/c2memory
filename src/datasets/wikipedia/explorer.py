from __future__ import annotations

import json

from .client import WikipediaServiceClient


class WikipediaActionExplorer:
    """Dataset-agnostic Wikipedia actions exposed to MAS environments."""

    def __init__(self, client: WikipediaServiceClient | None = None) -> None:
        self.client = client or WikipediaServiceClient()

    def search(self, query: str, *, top_k: int = 10) -> str:
        try:
            hits = self.client.get_hits(query, top_k=top_k)
        except Exception as exc:
            return f"Search service error for [{query}]: {type(exc).__name__}: {exc}"
        if not hits:
            return f"Could not find [{query}]."

        lines = ["Top matches:"]
        for idx, hit in enumerate(hits[:top_k], start=1):
            lines.append(f"{idx}. {hit.title}")
        return "\n".join(lines)

    def lookup(self, title: str) -> str:
        try:
            result = self.client.lookup_title(title)
        except Exception as exc:
            return f"Lookup service error for [{title}]: {type(exc).__name__}: {exc}"
        if not result.found or not result.metadata:
            return f"Could not lookup title [{title}]."

        payload = {
            "title": result.title,
            "document": result.document,
            "metadata": result.metadata,
        }
        return json.dumps(payload, ensure_ascii=False)
