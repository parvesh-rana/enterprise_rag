"""Tests for retrieval.fusion (RRF) and retrieval.filters."""

from __future__ import annotations

import math

import pytest

from retrieval.dense import RetrievedChunk
from retrieval.filters import RetrievalFilter
from retrieval.fusion import reciprocal_rank_fusion


def _hit(chunk_id: str, score: float = 0.0, source: str = "dense") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, score=score, payload={"chunk_id": chunk_id}, source=source
    )


# ---------- RRF ----------


def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_ranking_preserves_order() -> None:
    ranking = [_hit("a"), _hit("b"), _hit("c")]
    fused = reciprocal_rank_fusion([ranking])
    assert [h.chunk_id for h in fused] == ["a", "b", "c"]


def test_rrf_score_is_sum_of_reciprocals() -> None:
    dense = [_hit("a"), _hit("b")]
    sparse = [_hit("b"), _hit("a")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    by_id = {h.chunk_id: h.score for h in fused}
    # Both docs got rank 1 in one ranking and rank 2 in the other:
    # score = 1/(60+1) + 1/(60+2)
    expected = 1.0 / 61 + 1.0 / 62
    assert math.isclose(by_id["a"], expected, rel_tol=1e-9)
    assert math.isclose(by_id["b"], expected, rel_tol=1e-9)


def test_rrf_ranking_combination_promotes_overlap() -> None:
    # 'b' is rank 1 in BOTH rankings, beating 'a' which is rank 1 in one only.
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("b"), _hit("d"), _hit("e")]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].chunk_id == "b"
    assert {h.chunk_id for h in fused} == {"a", "b", "c", "d", "e"}


def test_rrf_truncates_to_top_k() -> None:
    dense = [_hit(f"d{i}") for i in range(10)]
    sparse = [_hit(f"s{i}") for i in range(10)]
    fused = reciprocal_rank_fusion([dense, sparse], top_k=5)
    assert len(fused) == 5


def test_rrf_marks_results_as_hybrid() -> None:
    dense = [_hit("a", source="dense")]
    sparse = [_hit("a", source="sparse")]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].source == "hybrid"


def test_rrf_keeps_payload_from_first_ranking_seen() -> None:
    dense_hit = RetrievedChunk(
        chunk_id="a", score=0.0, payload={"text": "from dense"}, source="dense"
    )
    sparse_hit = RetrievedChunk(
        chunk_id="a", score=0.0, payload={"text": "from sparse"}, source="sparse"
    )
    fused = reciprocal_rank_fusion([[dense_hit], [sparse_hit]])
    assert fused[0].payload["text"] == "from dense"


def test_rrf_respects_k_constant() -> None:
    # Larger k flattens differences; the relative ordering must still hold
    # even when absolute scores shrink.
    dense = [_hit("a"), _hit("b")]
    sparse = [_hit("a")]
    fused_small = reciprocal_rank_fusion([dense, sparse], k=10)
    fused_large = reciprocal_rank_fusion([dense, sparse], k=1000)
    assert fused_small[0].chunk_id == "a"
    assert fused_large[0].chunk_id == "a"
    assert fused_small[0].score > fused_large[0].score


# ---------- filters ----------


def test_empty_filter_is_noop() -> None:
    f = RetrievalFilter()
    assert f.is_empty()
    assert f.to_qdrant() is None
    pred = f.to_predicate()
    assert pred({"company": "AAPL"}) is True


def test_predicate_matches_company_year_item() -> None:
    f = RetrievalFilter(company="AAPL", year=2024, item="1A")
    pred = f.to_predicate()
    assert pred({"company": "AAPL", "year": 2024, "item": "1A"}) is True
    assert pred({"company": "MSFT", "year": 2024, "item": "1A"}) is False
    assert pred({"company": "AAPL", "year": 2023, "item": "1A"}) is False
    assert pred({"company": "AAPL", "year": 2024, "item": "7"}) is False


def test_partial_filter_only_constrains_specified_fields() -> None:
    f = RetrievalFilter(company="AAPL")
    pred = f.to_predicate()
    assert pred({"company": "AAPL", "year": 2024, "item": "anything"}) is True
    assert pred({"company": "AAPL", "year": 1999}) is True
    assert pred({"company": "MSFT"}) is False


def test_qdrant_filter_has_correct_must_clauses() -> None:
    f = RetrievalFilter(company="AAPL", year=2024)
    qf = f.to_qdrant()
    assert qf is not None
    keys = {c.key for c in qf.must}  # type: ignore[union-attr]
    assert keys == {"company", "year"}


@pytest.mark.parametrize(
    "filt,payload,expected",
    [
        (RetrievalFilter(year=2024), {"year": 2024}, True),
        (RetrievalFilter(year=2024), {"year": 2023}, False),
        (RetrievalFilter(item="7A"), {"item": "7A"}, True),
        (RetrievalFilter(item="7A"), {"item": "7"}, False),
    ],
)
def test_predicate_param(filt: RetrievalFilter, payload: dict, expected: bool) -> None:
    assert filt.to_predicate()(payload) is expected
