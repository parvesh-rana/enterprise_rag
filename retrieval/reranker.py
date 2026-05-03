"""Cross-encoder reranker.

Bi-encoders (used in dense retrieval) score query and document independently.
Cross-encoders score them jointly, which is more accurate but quadratic in
candidates. We only run it on the fused top-N (default 20) → top-k (default 5).

The reranker is lazy-loaded the first time it's invoked, like the bi-encoder.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from core.config import get_settings
from core.logging import get_logger
from retrieval.dense import RetrievedChunk

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _model() -> Any:
    """Cache the CrossEncoder. Imported lazily to keep cold tests fast."""
    from sentence_transformers import CrossEncoder

    name = get_settings().reranker_model
    log.info("reranker.load", model=name)
    return CrossEncoder(name)


def rerank(
    query: str,
    candidates: Sequence[RetrievedChunk],
    *,
    top_k: int | None = None,
    score_fn: Callable[[str, list[tuple[str, str]]], list[float]] | None = None,
) -> list[RetrievedChunk]:
    """Re-score candidates with a cross-encoder; return top_k by new score.

    `score_fn` lets tests inject a deterministic scorer without loading the
    heavy CrossEncoder. Production calls pass `score_fn=None` and use the
    cached model.
    """
    if not candidates:
        return []
    settings = get_settings()
    k = top_k or settings.final_top_k

    pairs = [(query, c.payload.get("text", "")) for c in candidates]
    if score_fn is not None:
        scores = score_fn(query, pairs)
    else:
        scores = list(_model().predict(pairs, show_progress_bar=False))

    if len(scores) != len(candidates):
        raise RuntimeError("reranker scorer must return one score per candidate")

    rescored = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            score=float(s),
            payload=c.payload,
            source="reranked",
        )
        for c, s in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored[:k]
