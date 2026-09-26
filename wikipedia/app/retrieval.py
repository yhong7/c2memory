from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import ServiceSettings
from app.embedder import QueryEmbedder


class RetrievalService:
    def __init__(self, settings: ServiceSettings, embedder: QueryEmbedder):
        self.settings = settings
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self.collection = self.client.get_collection(settings.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Wikipedia collection {settings.collection_name!r} was not found in "
                f"{settings.chroma_dir}. Build or mount the index before starting the service."
            ) from exc

    @property
    def count(self) -> int:
        return self.collection.count()

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self.collection.metadata

    def warmup(self, query: str | None = None, top_k: int = 1) -> dict[str, Any]:
        return self.search(query or self.settings.warmup_query, top_k=top_k)

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        k = top_k or self.settings.default_top_k
        started = perf_counter()
        query_vector = self.embedder.embed_query(query)
        embedded = perf_counter()
        results = self.collection.query(query_embeddings=[query_vector], n_results=k)
        searched = perf_counter()

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        hits = [
            {
                "id": str(doc_id),
                "distance": distances[index] if index < len(distances) else None,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
            for index, doc_id in enumerate(ids)
        ]
        return {
            "query": query,
            "top_k": k,
            "embedding_seconds": round(embedded - started, 6),
            "search_seconds": round(searched - embedded, 6),
            "total_seconds": round(searched - started, 6),
            "hits": hits,
        }

    def lookup_title(self, title: str) -> dict[str, Any]:
        started = perf_counter()
        results = self.collection.get(
            where={"title": title},
            include=["documents", "metadatas"],
            limit=1,
        )
        finished = perf_counter()

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        found = bool(ids)
        raw_metadata = metadatas[0] if found and metadatas else {}
        raw_links = raw_metadata.get("links", [])
        if isinstance(raw_links, str):
            try:
                parsed_links = json.loads(raw_links)
            except json.JSONDecodeError:
                parsed_links = []
        elif isinstance(raw_links, list):
            parsed_links = raw_links
        else:
            parsed_links = []

        links = []
        for item in parsed_links:
            if isinstance(item, dict):
                target = str(item.get("target_title", "")).strip()
            else:
                target = str(item).strip()
            if target:
                links.append(target)

        return {
            "title": title,
            "found": found,
            "lookup_seconds": round(finished - started, 6),
            "document": documents[0] if found and documents else None,
            "metadata": {"links": links} if found else None,
        }

