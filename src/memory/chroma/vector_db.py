"""Vector database abstraction backed by ChromaDB."""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.config import Settings as ChromaSettings

from src.memory.chroma.embedding import OpenAICompatibleEmbedding


class _NoOpEmbedding(EmbeddingFunction):
    """Dummy embedding function to prevent ChromaDB downloading ONNX models."""

    def __call__(self, input: Documents) -> Embeddings:
        return [[0.0]] * len(input)


# ---------------------------------------------------------------------------
# abstract base
# ---------------------------------------------------------------------------


class VectorDBBase(ABC):
    """Minimal vector-database interface.

    All methods are synchronous.  Implementations are expected to handle
    embedding generation internally inside ``upsert``.
    """

    @abstractmethod
    def create_collection(self, name: str, dim: int) -> None:
        """Create a collection / table with the given vector dimension."""
        ...

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete a collection and all its data."""
        ...

    @abstractmethod
    def upsert(
        self,
        collection: str,
        records: list[dict],
        embedding_field: str = "text",
    ) -> None:
        """Insert or update records.

        For each record the value of *embedding_field* is extracted and
        embedded via the configured embedding model.  The resulting
        vectors are stored alongside the full record metadata.
        """
        ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by ID."""
        ...

    @abstractmethod
    def get(self, collection: str, ids: list[str]) -> list[dict[str, Any]]:
        """Read vectors and metadata by ID."""
        ...

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search."""
        ...

    @abstractmethod
    def store_raw(self, collection: str, records: list[dict]) -> None:
        """Store records WITHOUT embedding — pure document archival.

        Records are stored as documents + metadata only (no vectors).
        They can be retrieved by ID but not via similarity search.
        """
        ...


# ---------------------------------------------------------------------------
# ChromaDB implementation
# ---------------------------------------------------------------------------


@dataclass
class ChromaVectorDB(VectorDBBase):
    """``VectorDBBase`` backed by a local ChromaDB ``PersistentClient``.

    Embedding is delegated to an OpenAI-compatible embedding client.

    Parameters
    ----------
    persist_dir:
        ChromaDB persistence directory.
    embedding:
        Configured embedding client.
    """

    persist_dir: str = ".db/chroma"
    embedding: OpenAICompatibleEmbedding | None = None

    # -- private --
    _client: chromadb.PersistentClient | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cols: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def create_collection(self, name: str, dim: int) -> None:
        with self._lock:
            kwargs: dict = {"name": name, "metadata": {"hnsw:space": "cosine", "dimension": dim}}
            if self.embedding is None:
                kwargs["embedding_function"] = _NoOpEmbedding()
            self._cols[name] = self._client.create_collection(**kwargs)

    def drop_collection(self, name: str) -> None:
        with self._lock:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
            self._cols.pop(name, None)

    def upsert(
        self,
        collection: str,
        records: list[dict],
        embedding_field: str = "text",
    ) -> None:
        if not records:
            return

        col = self._get_collection(collection)

        # 1 — extract documents & generate vectors in background thread pool
        documents = [str(r.get(embedding_field, "")) for r in records]
        vectors = self.embedding.embed_documents(documents) if self.embedding else []

        # 2 — build payloads
        ids: list[str] = []
        metadatas: list[dict] = []
        for i, rec in enumerate(records):
            doc_id = rec.get("id") or _make_id(f"doc_{i}")
            ids.append(doc_id)
            # Flatten anything non-serialisable
            meta = {k: _serialize_val(v) for k, v in rec.items()}
            metadatas.append(meta)

        # 3 — upsert into chromadb
        with self._lock:
            col.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=vectors or None,
            )

    def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        col = self._get_collection(collection)
        with self._lock:
            col.delete(ids=ids)

    def get(self, collection: str, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        col = self._get_collection(collection)
        results = col.get(ids=ids)
        return _format_get_results(results)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        col = self._get_collection(collection)
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter,
        )
        return _format_query_results(results)

    def store_raw(self, collection: str, records: list[dict]) -> None:
        """Store records WITHOUT embedding — pure document archival."""
        if not records:
            return

        col = self._get_or_create_collection(collection)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for i, rec in enumerate(records):
            doc_id = rec.get("id") or _make_id(f"{collection}_{i}")
            ids.append(doc_id)
            documents.append(json.dumps(rec, ensure_ascii=False))
            metadatas.append({k: _serialize_val(v) for k, v in rec.items()})

        with self._lock:
            col.add(ids=ids, documents=documents, metadatas=metadatas)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _get_collection(self, name: str):
        if name not in self._cols:
            self._cols[name] = self._client.get_collection(name=name)
        return self._cols[name]

    def _get_or_create_collection(self, name: str):
        """Get existing collection or create one (dim=1 placeholder for raw storage)."""
        try:
            return self._get_collection(name)
        except Exception:
            self.create_collection(name, dim=1)
            return self._cols[name]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

import hashlib
import time as _time


def _make_id(prefix: str) -> str:
    ts = str(int(_time.time() * 1_000_000))
    raw = f"{prefix}_{ts}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{prefix}_{short_hash}"


def _serialize_val(v: Any, max_len: int = 4000) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        s = json.dumps(v, ensure_ascii=False)
        return s[:max_len]
    s = str(v)
    return s[:max_len]


def _format_get_results(results: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, doc_id in enumerate(results.get("ids", [])):
        rec: dict[str, Any] = {"id": doc_id}
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        embeds = results.get("embeddings") or []
        if i < len(docs):
            rec["document"] = docs[i]
        if i < len(metas):
            rec["metadata"] = metas[i]
        if i < len(embeds):
            rec["embedding"] = embeds[i]
        out.append(rec)
    return out


def _format_query_results(results: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]
    for i, doc_id in enumerate(ids_list):
        rec: dict[str, Any] = {"id": doc_id}
        if i < len(docs_list):
            rec["document"] = docs_list[i]
        if i < len(metas_list):
            rec["metadata"] = metas_list[i]
        if i < len(dists_list):
            rec["distance"] = dists_list[i]
        out.append(rec)
    return out
