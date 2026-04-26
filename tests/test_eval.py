"""Tests for evaluation.metrics, evaluation.dataset, and evaluation.judge."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.dataset import QAExample, load_qa_set
from evaluation.judge import judge_answer
from evaluation.metrics import (
    aggregate,
    is_hit,
    recall_at_k,
    reciprocal_rank,
)
from generation.llm import Completion, LLMClient, Message
from retrieval.dense import RetrievedChunk


def _hit(cid: str, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        score=0.5,
        payload={"chunk_id": cid, "text": text},
        source="hybrid",
    )


def _ex(
    *,
    id: str = "q",
    kind: str = "factoid",
    gold_ids: list[str] | None = None,
    gold_subs: list[str] | None = None,
) -> QAExample:
    return QAExample(
        id=id,
        question="Q?",
        kind=kind,  # type: ignore[arg-type]
        gold_chunk_ids=gold_ids or [],
        gold_substrings=gold_subs or [],
    )


# ---------- is_hit / recall@k / mrr ----------


def test_is_hit_matches_chunk_id() -> None:
    assert is_hit(_hit("a"), _ex(gold_ids=["a"])) is True
    assert is_hit(_hit("b"), _ex(gold_ids=["a"])) is False


def test_is_hit_matches_substring_case_insensitive() -> None:
    chunk = _hit("c1", text="Revenue grew 12% YoY in fiscal 2024.")
    assert is_hit(chunk, _ex(gold_subs=["Revenue grew"])) is True
    assert is_hit(chunk, _ex(gold_subs=["REVENUE"])) is True
    assert is_hit(chunk, _ex(gold_subs=["losses widened"])) is False


def test_is_hit_either_id_or_substring_counts() -> None:
    chunk = _hit("a", text="lorem ipsum")
    ex = _ex(gold_ids=["a"], gold_subs=["zzz"])
    assert is_hit(chunk, ex) is True


def test_recall_at_k_binary() -> None:
    retrieved = [_hit("a"), _hit("b"), _hit("c"), _hit("d"), _hit("e")]
    assert recall_at_k(retrieved, _ex(gold_ids=["c"]), k=5) == 1.0
    assert recall_at_k(retrieved, _ex(gold_ids=["zzz"]), k=5) == 0.0
    assert recall_at_k(retrieved, _ex(gold_ids=["c"]), k=2) == 0.0  # rank 3, not in top 2


def test_reciprocal_rank() -> None:
    retrieved = [_hit("a"), _hit("b"), _hit("c")]
    assert reciprocal_rank(retrieved, _ex(gold_ids=["a"])) == 1.0
    assert reciprocal_rank(retrieved, _ex(gold_ids=["b"])) == 0.5
    assert reciprocal_rank(retrieved, _ex(gold_ids=["c"])) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, _ex(gold_ids=["nope"])) == 0.0


def test_aggregate_means_per_metric() -> None:
    rows = [
        {"recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0},
        {"recall@5": 0.0, "recall@10": 1.0, "mrr": 0.5},
    ]
    s = aggregate(rows)
    assert s.recall_at_5 == 0.5
    assert s.recall_at_10 == 1.0
    assert s.mrr == 0.75


def test_aggregate_empty_returns_zeros() -> None:
    s = aggregate([])
    assert (s.recall_at_5, s.recall_at_10, s.mrr) == (0.0, 0.0, 0.0)


# ---------- dataset loader ----------


def test_load_qa_set_skips_blank_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text(
        "\n# header comment\n"
        '{"id":"a","question":"Q?","kind":"factoid","gold_substrings":["x"]}\n'
        "\n"
        '{"id":"b","question":"Q?","kind":"unanswerable"}\n',
        encoding="utf-8",
    )
    examples = load_qa_set(p)
    assert [e.id for e in examples] == ["a", "b"]
    assert examples[1].is_unanswerable() is True


def test_load_qa_set_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_qa_set(tmp_path / "nope.jsonl")


def test_shipped_qa_set_parses() -> None:
    """The committed qa_set.jsonl must always be valid."""
    path = Path(__file__).resolve().parent.parent / "evaluation" / "qa_set.jsonl"
    examples = load_qa_set(path)
    assert 30 <= len(examples) <= 50
    kinds = {e.kind for e in examples}
    assert {"factoid", "comparative", "multi_hop", "unanswerable"} <= kinds


# ---------- judge ----------


class FakeJudgeLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        return Completion(text=self.reply, model="fake-judge")


def test_judge_short_circuits_on_abstention_phrase() -> None:
    ex = _ex(kind="unanswerable")
    res = judge_answer(
        example=ex,
        answer_text="I don't have enough information in the provided filings to answer that.",
        cited_chunks=[],
        client=FakeJudgeLLM("SHOULD-NOT-BE-CALLED"),
    )
    assert res.verdict == "ABSTAINED"
    assert res.score == 1.0  # rewarded — the example is unanswerable


def test_judge_abstain_on_answerable_scores_zero() -> None:
    ex = _ex(kind="factoid")
    res = judge_answer(
        example=ex,
        answer_text="I don't have enough information in the provided filings to answer that.",
        cited_chunks=[],
        client=FakeJudgeLLM("ignored"),
    )
    assert res.verdict == "ABSTAINED"
    assert res.score == 0.0


@pytest.mark.parametrize(
    "reply,verdict,score",
    [
        ("SUPPORTED: every claim is in the passages.", "SUPPORTED", 1.0),
        ("PARTIAL: one claim is unsupported.", "PARTIAL", 0.5),
        ("UNSUPPORTED: nothing is grounded.", "UNSUPPORTED", 0.0),
        ("garbled output with no token", "UNSUPPORTED", 0.0),
    ],
)
def test_judge_parses_verdict_tokens(reply: str, verdict: str, score: float) -> None:
    res = judge_answer(
        example=_ex(),
        answer_text="Some grounded claim.",
        cited_chunks=[_hit("c1", text="evidence")],
        client=FakeJudgeLLM(reply),
    )
    assert res.verdict == verdict
    assert res.score == score
