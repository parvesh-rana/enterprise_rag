"""Sentence-transformers wrapper for dense embeddings.

The model name is configurable; the default `BAAI/bge-small-en-v1.5` outputs
384-d L2-normalized vectors and is small enough to run on CPU.
BGE models expect a query-side instruction prefix; we apply it for queries
and not for documents.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

# Per-model query instruction. Documents do NOT get a prefix.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> Any:
    """Lazy import + cache the SentenceTransformer model.

    Imported inside the function so unit tests that don't touch embeddings
    don't pay the model load cost or pull in torch. Return type is Any so
    we don't have to import SentenceTransformer at module scope.
    """
    from sentence_transformers import SentenceTransformer

    name = get_settings().embedding_model
    log.info("embeddings.load", model=name)
    return SentenceTransformer(name)


def _query_instruction() -> str:
    name = get_settings().embedding_model.lower()
    if "bge" in name:
        return _BGE_QUERY_INSTRUCTION
    return ""


def embed_documents(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    """Encode chunk texts. Returns a (n, dim) float32 array, L2-normalized."""
    arr = _model().encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    out: np.ndarray = arr.astype(np.float32, copy=False)
    return out


def embed_query(query: str) -> np.ndarray:
    """Encode a single query, prepending the model's instruction prefix."""
    text = _query_instruction() + query
    arr = _model().encode(
        [text],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    out: np.ndarray = arr[0].astype(np.float32, copy=False)
    return out
