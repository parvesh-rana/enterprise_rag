"""Dense retriever: query embedding → Qdrant search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from index.embeddings import embed_query
from index.vector_store import DenseHit, search
from retrieval.filters import RetrievalFilter


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]
    source: str  # "dense" | "sparse"


def dense_search(query: str, *, top_k: int, filt: RetrievalFilter) -> list[RetrievedChunk]:
    qvec = embed_query(query)
    hits: list[DenseHit] = search(
        query_vector=qvec, top_k=top_k, qdrant_filter=filt.to_qdrant()
    )
    return [
        RetrievedChunk(chunk_id=h.chunk_id, score=h.score, payload=h.payload, source="dense")
        for h in hits
    ]
