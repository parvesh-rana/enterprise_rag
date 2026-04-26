"""End-to-end retrieval: dense + sparse → RRF fusion → cross-encoder rerank."""

from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger
from retrieval.dense import RetrievedChunk, dense_search
from retrieval.filters import RetrievalFilter
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from retrieval.sparse import sparse_search

log = get_logger(__name__)


def retrieve(
    query: str,
    *,
    filt: RetrievalFilter | None = None,
    final_top_k: int | None = None,
    rerank_top_k: int | None = None,
    candidate_pool: int | None = None,
    use_reranker: bool = True,
) -> list[RetrievedChunk]:
    """Hybrid retrieval entrypoint.

    Pipeline: dense + sparse (each top `candidate_pool`) → RRF (top `rerank_top_k`)
    → cross-encoder rerank (top `final_top_k`).

    Args:
        query: natural-language question.
        filt: optional metadata pre-filter (company / year / item).
        final_top_k: final number of chunks returned. Default settings.final_top_k.
        rerank_top_k: candidates fed into the reranker. Default settings.rerank_top_k.
        candidate_pool: hits each retriever returns before fusion. Default 2 * rerank_top_k.
        use_reranker: skip the cross-encoder if False (eval ablations).
    """
    settings = get_settings()
    filt = filt or RetrievalFilter()
    rerank_k = rerank_top_k or settings.rerank_top_k
    final_k = final_top_k or settings.final_top_k
    pool = candidate_pool or rerank_k * 2

    dense_hits = dense_search(query, top_k=pool, filt=filt)
    sparse_hits = sparse_search(query, top_k=pool, filt=filt)

    fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=rerank_k)

    if not use_reranker:  # noqa: SIM108 — explicit branches read clearer than a ternary here
        result = fused[:final_k]
    else:
        result = rerank(query, fused, top_k=final_k)

    log.info(
        "retrieve.done",
        query_len=len(query),
        n_dense=len(dense_hits),
        n_sparse=len(sparse_hits),
        n_fused=len(fused),
        n_final=len(result),
        reranker=use_reranker,
        filter=filt.model_dump(exclude_none=True),
    )
    return result
