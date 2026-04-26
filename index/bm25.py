"""BM25 sparse index over the same chunks as the vector store.

Persisted to disk as a single pickle: tokens + chunk_ids + payload metadata
needed for filtering at query time. Cheap to load (<200 ms for 5 filings).
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from core.types import Chunk

PayloadPredicate = Callable[[dict[str, Any]], bool]

# English stopword list small enough to inline. Removing them noticeably
# improves BM25 quality on financial prose where "the/of/and" dominate.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-\.]*")


def tokenize(text: str) -> list[str]:
    """Lowercase, alnum-keeping tokenizer with stopword filtering.

    Keeps internal punctuation that matters in financial filings:
      - decimals: "10.5"
      - tickers/abbrevs: "u.s.a.", "10-k"
      - hyphenated terms: "year-over-year"
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class SparseHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


@dataclass
class BM25Index:
    """Serializable BM25 index. `bm25` is recreated from `corpus` on load."""

    chunk_ids: list[str]
    payloads: list[dict[str, Any]]
    corpus: list[list[str]]
    bm25: BM25Okapi | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.bm25 is None:
            self.bm25 = BM25Okapi(self.corpus)

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> BM25Index:
        corpus = [tokenize(c.text) for c in chunks]
        payloads = [
            {
                "chunk_id": c.id,
                "company": c.company,
                "company_name": c.company_name,
                "year": c.year,
                "item": c.item,
                "section_title": c.section_title,
                "text": c.text,
                "source_url": c.source_url,
            }
            for c in chunks
        ]
        return cls(chunk_ids=[c.id for c in chunks], payloads=payloads, corpus=corpus)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Drop the BM25Okapi instance; it's rebuilt from corpus on load.
        with path.open("wb") as fh:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "payloads": self.payloads, "corpus": self.corpus},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as fh:
            data = pickle.load(fh)
        return cls(**data)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        predicate: PayloadPredicate | None = None,
    ) -> list[SparseHit]:
        """Return top-k hits, optionally filtered by a payload predicate.

        `predicate(payload) -> bool` is applied before truncation so filters
        compose with retrieval rather than masking the top results.
        """
        assert self.bm25 is not None
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)

        candidates: list[tuple[int, float]] = list(enumerate(scores))
        if predicate is not None:
            candidates = [(i, s) for i, s in candidates if predicate(self.payloads[i])]

        # Cheap top-k via partial sort.
        candidates.sort(key=lambda t: t[1], reverse=True)
        candidates = candidates[:top_k]

        return [
            SparseHit(
                chunk_id=self.chunk_ids[i],
                score=float(score),
                payload=self.payloads[i],
            )
            for i, score in candidates
            if score > 0.0
        ]
