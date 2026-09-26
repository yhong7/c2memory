from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

from app.config import ServiceSettings


class QueryEmbedder(ABC):
    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError


def _embeddings_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/embeddings"):
        return normalized
    return f"{normalized}/embeddings"


class APIQueryEmbedder(QueryEmbedder):
    def __init__(self, settings: ServiceSettings):
        self.api_key = settings.embedding_api_key
        self.endpoint = _embeddings_endpoint(settings.embedding_api_base)
        self.model = settings.embedding_model_name
        self.timeout = settings.request_timeout_seconds

    def embed_query(self, query: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": self.model, "input": [query]}
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        try:
            return [float(value) for value in body["data"][0]["embedding"]]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Embedding API returned an invalid response") from exc


class LocalQueryEmbedder(QueryEmbedder):
    def __init__(self, settings: ServiceSettings):
        from sentence_transformers import SentenceTransformer

        model_path = Path(settings.local_embedding_model).expanduser().resolve()
        self.model = SentenceTransformer(str(model_path))

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode([query], normalize_embeddings=False)
        return vector[0].tolist()


def build_query_embedder(settings: ServiceSettings) -> QueryEmbedder:
    if settings.embed_mode == "api":
        return APIQueryEmbedder(settings)
    if settings.embed_mode == "local":
        return LocalQueryEmbedder(settings)
    raise ValueError(f"Unsupported WIKI_SERVICE_EMBED_MODE: {settings.embed_mode}")
