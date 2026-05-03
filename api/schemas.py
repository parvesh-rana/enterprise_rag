"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.types import Citation


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    company: str | None = Field(default=None, description="Ticker filter, e.g. 'AAPL'")
    year: int | None = Field(default=None, ge=1990, le=2100)
    item: str | None = Field(default=None, description="10-K Item key, e.g. '1A'")
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_reranker: bool = True


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    score: float
    company: str
    year: int
    item: str
    section_title: str
    text_preview: str
    source_url: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    model: str
    citations: list[Citation]
    retrieved: list[RetrievedChunkOut]
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    vector_store: bool
    bm25: bool
    llm_provider: str
    embedding_model: str


class SourceResponse(BaseModel):
    chunk_id: str
    company: str
    company_name: str
    year: int
    item: str
    section_title: str
    text: str
    source_url: str


class ErrorResponse(BaseModel):
    request_id: str
    detail: str
