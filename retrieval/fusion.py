"""Reciprocal Rank Fusion.

RRF score for a document d across rankings R is:

    RRF(d) = sum over r in R of 1 / (k + rank_r(d))

where rank starts at 1 for the top hit. The constant `k` (default 60, per
Cormack et al. 2009) damps the contribution of any single high rank, which is
why RRF is robust to score-scale mismatch between dense (cosine in [-1, 1])
and BM25 (unbounded positive).
"""

from __future__ import annotations

from collections.abc import Sequence

from retrieval.dense import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse multiple rankings into a single ordered list.

    Each ranking is treated independently: its position determines the RRF
    contribution. When the same chunk_id appears in multiple rankings, we keep
    the payload from the *first* ranking it appeared in (typically dense, by
    convention of caller order) and tag source="hybrid".
    """
    fused: dict[str, float] = {}
    seen: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            if hit.chunk_id not in seen:
                seen[hit.chunk_id] = hit

    ordered = sorted(fused.items(), key=lambda t: t[1], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]

    out: list[RetrievedChunk] = []
    for chunk_id, score in ordered:
        original = seen[chunk_id]
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                score=score,
                payload=original.payload,
                source="hybrid",
            )
        )
    return out
