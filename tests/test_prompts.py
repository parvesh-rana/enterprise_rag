"""Tests for generation.prompts (prompt construction + citation parsing)
and generation.answer (orchestration with a fake LLM client)."""

from __future__ import annotations

from typing import Any

from core.types import Answer
from generation.answer import answer as run_answer
from generation.llm import Completion, LLMClient, Message
from generation.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    parse_citations,
)
from retrieval.dense import RetrievedChunk


def _hit(chunk_id: str, text: str = "lorem ipsum", **payload: Any) -> RetrievedChunk:
    pl = {
        "chunk_id": chunk_id,
        "text": text,
        "company": "AAPL",
        "company_name": "Apple Inc.",
        "year": 2024,
        "section_title": "Item 7",
        **payload,
    }
    return RetrievedChunk(chunk_id=chunk_id, score=0.5, payload=pl, source="hybrid")


# ---------- build_messages ----------

def test_build_messages_emits_system_and_user_pair() -> None:
    msgs = build_messages("What was revenue?", [_hit("c1", "Revenue was $42M.")])
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == SYSTEM_PROMPT


def test_user_message_includes_chunk_ids_and_metadata() -> None:
    msgs = build_messages(
        "What was revenue?",
        [_hit("AAPL-2024-7-0001", "Revenue was $42M in fiscal 2024.")],
    )
    user = msgs[1].content
    assert "[AAPL-2024-7-0001]" in user
    assert "Apple Inc." in user
    assert "Item 7" in user
    assert "$42M" in user
    assert "Question: What was revenue?" in user


def test_user_message_handles_empty_chunks() -> None:
    msgs = build_messages("Anything?", [])
    assert "(no passages were retrieved" in msgs[1].content


def test_system_prompt_contains_citation_contract() -> None:
    # The exact wording matters; the parser depends on the bracket format.
    assert "square brackets" in SYSTEM_PROMPT
    assert "I don't have enough information" in SYSTEM_PROMPT


# ---------- parse_citations ----------

def test_parse_citations_extracts_single_id() -> None:
    text = "Revenue grew 12% YoY [AAPL-2024-7-0003]."
    assert parse_citations(text, allowed_ids={"AAPL-2024-7-0003"}) == ["AAPL-2024-7-0003"]


def test_parse_citations_handles_multi_id_in_one_bracket() -> None:
    text = "The two segments grew at different rates [a, b]."
    assert parse_citations(text, allowed_ids={"a", "b"}) == ["a", "b"]


def test_parse_citations_dedupes_in_first_occurrence_order() -> None:
    text = "Foo [a]. Bar [b]. Baz [a, c]. Quux [b]."
    assert parse_citations(text, allowed_ids={"a", "b", "c"}) == ["a", "b", "c"]


def test_parse_citations_drops_hallucinated_ids() -> None:
    text = "Real fact [a]. Made-up fact [zzz]."
    assert parse_citations(text, allowed_ids={"a"}) == ["a"]


def test_parse_citations_returns_empty_when_no_brackets() -> None:
    assert parse_citations("I don't have enough information.", allowed_ids={"a"}) == []


def test_parse_citations_ignores_brackets_without_known_ids() -> None:
    text = "Some prose with [random text inside] and no real citations."
    assert parse_citations(text, allowed_ids={"AAPL-2024-7-0001"}) == []


def test_parse_citations_tolerates_whitespace_inside_brackets() -> None:
    text = "Fact [ AAPL-2024-7-0001 ,  AAPL-2024-7-0002 ]."
    assert parse_citations(
        text, allowed_ids={"AAPL-2024-7-0001", "AAPL-2024-7-0002"}
    ) == ["AAPL-2024-7-0001", "AAPL-2024-7-0002"]


# ---------- answer() with a fake LLM ----------

class FakeLLM(LLMClient):
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append(messages)
        return Completion(
            text=self._response_text,
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=20,
        )


def test_answer_orchestrates_and_attaches_validated_citations() -> None:
    chunks = [
        _hit("AAPL-2024-7-0001", "Revenue was $42 million."),
        _hit("AAPL-2024-7-0002", "R&D expense rose 8%."),
    ]
    fake = FakeLLM(
        "Revenue grew to $42M [AAPL-2024-7-0001]. "
        "R&D rose 8% [AAPL-2024-7-0002, FAKE-ID]."
    )
    out: Answer = run_answer(
        "What happened?", chunks, request_id="req-123", client=fake
    )
    assert out.model == "fake-model"
    assert out.request_id == "req-123"
    assert [c.chunk_id for c in out.citations] == [
        "AAPL-2024-7-0001",
        "AAPL-2024-7-0002",
    ]
    # FAKE-ID was hallucinated and must be dropped.
    assert all(c.chunk_id != "FAKE-ID" for c in out.citations)
    # Citation scores carry the retriever score forward.
    for cit in out.citations:
        assert cit.score == 0.5


def test_answer_passes_messages_to_client() -> None:
    fake = FakeLLM("No citations here.")
    run_answer("Q?", [_hit("c1")], request_id="r", client=fake)
    assert len(fake.calls) == 1
    msgs = fake.calls[0]
    assert msgs[0].role == "system"
    assert "[c1]" in msgs[1].content


def test_answer_with_no_chunks_returns_empty_citations() -> None:
    fake = FakeLLM("I don't have enough information in the provided filings to answer that.")
    out = run_answer("Q?", [], request_id="r", client=fake)
    assert out.citations == []
