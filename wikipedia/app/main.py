from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Request
from loguru import logger

from app.config import load_settings
from app.embedder import build_query_embedder
from app.logging_utils import setup_logging
from app.retrieval import RetrievalService
from app.schemas import (
    HealthResponse,
    LookupRequest,
    LookupResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    WarmupResponse,
)


setup_logging("wikipedia_service")
settings = load_settings()
retrieval = RetrievalService(settings, build_query_embedder(settings))
app = FastAPI(title="C2Mem Wikipedia service", version="1.0.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    logger.info(
        "HTTP {} {} -> {} in {:.3f}s",
        request.method,
        request.url.path,
        response.status_code,
        perf_counter() - started,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        collection=settings.collection_name,
        chroma_dir=str(settings.chroma_dir),
    )


@app.get("/ready", response_model=StatsResponse)
@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(
        collection=settings.collection_name,
        count=retrieval.count,
        metadata=retrieval.metadata,
    )


@app.post("/warmup", response_model=WarmupResponse)
def warmup() -> WarmupResponse:
    result = retrieval.warmup()
    top_hit_title = None
    if result["hits"]:
        top_hit_title = result["hits"][0]["metadata"].get("title")
    return WarmupResponse(
        query=result["query"],
        embedding_seconds=result["embedding_seconds"],
        search_seconds=result["search_seconds"],
        total_seconds=result["total_seconds"],
        top_hit_title=top_hit_title,
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    return SearchResponse(**retrieval.search(request.query, top_k=request.top_k))


@app.post("/lookup", response_model=LookupResponse)
def lookup(request: LookupRequest) -> LookupResponse:
    return LookupResponse(**retrieval.lookup_title(request.title))
