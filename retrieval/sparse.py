"""Sparse retriever: BM25 over the on-disk index."""

from __future__ import annotations

from functools import lru_cache

from core.config import get_settings
from index.bm25 import BM25Index
from retrieval.dense import RetrievedChunk
from retrieval.filters import RetrievalFilter


@lru_cache(maxsize=1)
def _load_index() -> BM25Index:
    return BM25Index.load(get_settings().bm25_dir / "index.pkl")


def reset_index_cache() -> None:
    """Drop the cached BM25 index. Use in tests after rebuilding the index."""
    _load_index.cache_clear()


def sparse_search(query: str, *, top_k: int, filt: RetrievalFilter) -> list[RetrievedChunk]:
    bm25 = _load_index()
    hits = bm25.search(query, top_k=top_k, predicate=filt.to_predicate())
    return [
        RetrievedChunk(chunk_id=h.chunk_id, score=h.score, payload=h.payload, source="sparse")
        for h in hits
    ]
