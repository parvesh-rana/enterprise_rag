"""Tests for retrieval.reranker — exercise the logic without loading the
heavy CrossEncoder by passing a `score_fn`."""

from __future__ import annotations

from collections.abc import Callable

from retrieval.dense import RetrievedChunk
from retrieval.reranker import rerank


def _hit(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, score=0.0, payload={"text": text}, source="hybrid")


def _fake_scorer_by_word(target_word: str) -> Callable[[str, list[tuple[str, str]]], list[float]]:
    """Score = number of times `target_word` appears in the candidate text."""

    def score(_q: str, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(doc.lower().count(target_word.lower())) for _, doc in pairs]

    return score


def test_rerank_empty_returns_empty() -> None:
    assert rerank("anything", [], score_fn=lambda *_: []) == []


def test_rerank_reorders_by_new_score() -> None:
    cands = [
        _hit("a", "rocket roller skates"),
        _hit("b", "anvils anvils anvils everywhere"),
        _hit("c", "fiscal year revenue summary"),
    ]
    out = rerank("anvil", cands, top_k=3, score_fn=_fake_scorer_by_word("anvil"))
    assert [c.chunk_id for c in out] == ["b", "a", "c"]


def test_rerank_truncates_to_top_k() -> None:
    cands = [_hit(f"c{i}", f"anvil {i}") for i in range(10)]
    out = rerank("anvil", cands, top_k=3, score_fn=_fake_scorer_by_word("anvil"))
    assert len(out) == 3


def test_rerank_marks_results_as_reranked() -> None:
    cands = [_hit("a", "anvil"), _hit("b", "skates")]
    out = rerank("anvil", cands, top_k=2, score_fn=_fake_scorer_by_word("anvil"))
    assert all(h.source == "reranked" for h in out)


def test_rerank_preserves_payload() -> None:
    cands = [
        RetrievedChunk(
            chunk_id="x",
            score=0.0,
            payload={"text": "anvil", "company": "ACME", "year": 2024},
            source="hybrid",
        )
    ]
    out = rerank("anvil", cands, top_k=1, score_fn=_fake_scorer_by_word("anvil"))
    assert out[0].payload["company"] == "ACME"
    assert out[0].payload["year"] == 2024


def test_rerank_score_replaces_rrf_score() -> None:
    cands = [_hit("a", "anvil anvil anvil")]
    out = rerank("anvil", cands, top_k=1, score_fn=_fake_scorer_by_word("anvil"))
    assert out[0].score == 3.0  # from the fake scorer, not the input 0.0


def test_rerank_raises_when_scorer_returns_wrong_length() -> None:
    cands = [_hit("a", "x"), _hit("b", "y")]
    bad = lambda _q, _pairs: [1.0]  # noqa: E731
    try:
        rerank("q", cands, top_k=2, score_fn=bad)
    except RuntimeError as e:
        assert "one score per candidate" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")
