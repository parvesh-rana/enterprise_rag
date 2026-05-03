"""HTTP routes."""

from __future__ import annotations

import time
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from api import metrics
from api.deps import enforce_rate_limit, get_bm25, get_llm
from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkOut,
    SourceResponse,
)
from core.config import get_settings
from core.logging import get_logger
from generation.answer import answer as run_answer
from generation.llm import LLMClient
from index.bm25 import BM25Index
from retrieval.dense import RetrievedChunk
from retrieval.filters import RetrievalFilter
from retrieval.pipeline import retrieve

log = get_logger(__name__)

router = APIRouter()

_PREVIEW_CHARS = 240


def _to_chunk_out(c: RetrievedChunk) -> RetrievedChunkOut:
    p = c.payload
    text = p.get("text", "")
    return RetrievedChunkOut(
        chunk_id=c.chunk_id,
        score=c.score,
        company=p.get("company", ""),
        year=int(p.get("year", 0)),
        item=p.get("item", ""),
        section_title=p.get("section_title", ""),
        text_preview=text[:_PREVIEW_CHARS] + ("…" if len(text) > _PREVIEW_CHARS else ""),
        source_url=p.get("source_url", ""),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
def query(
    body: QueryRequest,
    request: Request,
    llm: LLMClient = Depends(get_llm),
) -> QueryResponse:
    request_id: str = request.state.request_id
    settings = get_settings()
    filt = RetrievalFilter(company=body.company, year=body.year, item=body.item)

    t0 = time.perf_counter()
    chunks = retrieve(
        body.question,
        filt=filt,
        final_top_k=body.top_k or settings.final_top_k,
        use_reranker=body.use_reranker,
    )
    retrieval_s = time.perf_counter() - t0
    metrics.RETRIEVAL_LATENCY.observe(retrieval_s)

    t1 = time.perf_counter()
    ans = run_answer(body.question, chunks, request_id=request_id, client=llm)
    generation_s = time.perf_counter() - t1
    metrics.GENERATION_LATENCY.observe(generation_s)

    # Token usage is logged inside answer.run_answer; mirror it to Prometheus.
    # We pulled it via the structlog event there; re-derive from the LLM
    # response by exposing it via the Answer (kept minimal — we only have
    # totals on the Completion, not on Answer). For now, leave token metric
    # increments to the LLM client wrapper in a future phase.

    return QueryResponse(
        request_id=request_id,
        answer=ans.text,
        model=ans.model,
        citations=ans.citations,
        retrieved=[_to_chunk_out(c) for c in chunks],
        timings_ms={
            "retrieval": round(retrieval_s * 1000, 2),
            "generation": round(generation_s * 1000, 2),
            "total": round((retrieval_s + generation_s) * 1000, 2),
        },
    )


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    bm25: BM25Index | None = Depends(get_bm25),
) -> HealthResponse:
    settings = get_settings()
    qdrant_ok = False
    if settings.qdrant_url.startswith("local:"):
        try:
            from index.vector_store import _client

            collections = _client().get_collections().collections
            qdrant_ok = any(c.name == settings.qdrant_collection for c in collections)
        except Exception:
            qdrant_ok = False
    else:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{settings.qdrant_url}/readyz")
                qdrant_ok = r.status_code == 200
        except httpx.HTTPError:
            qdrant_ok = False

    bm25_ok = bm25 is not None
    overall: Literal["ok", "degraded"] = "ok" if (qdrant_ok and bm25_ok) else "degraded"
    return HealthResponse(
        status=overall,
        qdrant=qdrant_ok,
        bm25=bm25_ok,
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
    )


@router.get("/sources/{chunk_id}", response_model=SourceResponse)
def source(
    chunk_id: str, request: Request, bm25: BM25Index | None = Depends(get_bm25)
) -> SourceResponse:
    """Return the full text of a chunk by id. BM25 carries the full payload
    so we read from it rather than going back to Qdrant."""
    if bm25 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="bm25 index not loaded; run `make index` first",
        )

    try:
        idx = bm25.chunk_ids.index(chunk_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"chunk {chunk_id} not found",
        ) from exc

    p = bm25.payloads[idx]
    return SourceResponse(
        chunk_id=p["chunk_id"],
        company=p.get("company", ""),
        company_name=p.get("company_name", ""),
        year=int(p.get("year", 0)),
        item=p.get("item", ""),
        section_title=p.get("section_title", ""),
        text=p.get("text", ""),
        source_url=p.get("source_url", ""),
    )


@router.get("/metrics")
def metrics_endpoint() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
