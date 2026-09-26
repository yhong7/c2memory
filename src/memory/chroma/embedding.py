"""Client for an OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests


class OpenAICompatibleEmbedding:
    """Thin wrapper around an OpenAI-compatible embeddings API.

    Parameters
    ----------
    api_key:
        Provider API key.
    model:
        Embedding model name.
    base_url:
        API endpoint.
    batch_size:
        Maximum documents per API call.
    max_workers:
        Thread pool size for parallel batch embedding in ``embed_documents``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        batch_size: int = 32,
        max_workers: int = 4,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Vector dimension (lazy-resolved on first embed call)."""
        if self._dim is None:
            self._dim = len(self.embed_query("dim probe"))
        return self._dim

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self._call_api([query])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Large lists are split into batches and processed in parallel
        via a thread pool so that embedding does not block the main
        thread for long-running workloads.
        """
        if not documents:
            return []

        batches = [
            documents[i : i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]

        # Single batch — call directly, no thread-pool overhead
        if len(batches) == 1:
            return self._call_api(batches[0])

        # Multiple batches — parallelise
        results: dict[int, list[list[float]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._call_api, b): idx for idx, b in enumerate(batches)}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

        # Reconstitute in original order
        flat: list[list[float]] = []
        for i in sorted(results):
            flat.extend(results[i])
        return flat

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _call_api(self, inputs: list[str]) -> list[list[float]]:
        """Single API call for one batch."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "input": inputs,
        }

        max_retries = 3
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                # OpenAI-compatible response: data[i].embedding
                embeddings = [
                    item["embedding"] for item in data["data"]
                ]
                # Sort by index to preserve input order
                return [
                    emb for _, emb in sorted(
                        enumerate(embeddings),
                        key=lambda pair: data["data"][pair[0]].get("index", pair[0]),
                    )
                ]
            except Exception as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise last_err  # type: ignore[misc]
