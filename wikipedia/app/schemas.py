from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    id: str
    distance: float | None = None
    document: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    embedding_seconds: float
    search_seconds: float
    total_seconds: float
    hits: list[SearchHit]


class LookupRequest(BaseModel):
    title: str = Field(min_length=1)


class LookupResponse(BaseModel):
    title: str
    found: bool
    lookup_seconds: float
    document: str | None = None
    metadata: dict[str, list[str]] | None = None


class WarmupResponse(BaseModel):
    query: str
    embedding_seconds: float
    search_seconds: float
    total_seconds: float
    top_hit_title: str | None = None


class HealthResponse(BaseModel):
    status: str
    collection: str
    chroma_dir: str


class StatsResponse(BaseModel):
    collection: str
    count: int
    metadata: dict[str, Any] | None = None

