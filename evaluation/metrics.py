"""Retrieval metrics: Recall@k and MRR.

A retrieved chunk counts as a hit for an example when:
  - its `chunk_id` is in `example.gold_chunk_ids`, OR
  - any string in `example.gold_substrings` appears (case-insensitive) in
    the chunk's text.

The substring path keeps the eval set robust to re-chunking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evaluation.dataset import QAExample
from retrieval.dense import RetrievedChunk


def is_hit(chunk: RetrievedChunk, example: QAExample) -> bool:
    if chunk.chunk_id in set(example.gold_chunk_ids):
        return True
    if not example.gold_substrings:
        return False
    text = chunk.payload.get("text", "").lower()
    return any(sub.lower() in text for sub in example.gold_substrings)


def recall_at_k(retrieved: Sequence[RetrievedChunk], example: QAExample, k: int) -> float:
    """1.0 if any of the top-k chunks is a gold hit, else 0.0.

    For evaluation over a small QA set with ~1 gold passage per question, the
    binary form is the right read of "did retrieval surface evidence at all."
    Multi-evidence questions can be split into multiple QAExamples.
    """
    for chunk in retrieved[:k]:
        if is_hit(chunk, example):
            return 1.0
    return 0.0


def reciprocal_rank(retrieved: Sequence[RetrievedChunk], example: QAExample) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if is_hit(chunk, example):
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalScores:
    recall_at_5: float
    recall_at_10: float
    mrr: float


def aggregate(per_example: list[dict[str, float]]) -> RetrievalScores:
    if not per_example:
        return RetrievalScores(0.0, 0.0, 0.0)
    n = len(per_example)
    return RetrievalScores(
        recall_at_5=sum(r["recall@5"] for r in per_example) / n,
        recall_at_10=sum(r["recall@10"] for r in per_example) / n,
        mrr=sum(r["mrr"] for r in per_example) / n,
    )
