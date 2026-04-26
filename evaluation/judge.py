"""LLM-as-judge faithfulness scoring.

Given (question, answer, cited_chunks), the judge returns a verdict:
  - SUPPORTED: every factual claim in the answer is grounded in the cited chunks.
  - PARTIAL  : at least one claim is supported and at least one is not.
  - UNSUPPORTED: no claim is grounded in the cited chunks (hallucination).
  - ABSTAINED : the answer explicitly declines (the system prompt template).

Scoring (per example):
  SUPPORTED   = 1.0
  PARTIAL     = 0.5
  UNSUPPORTED = 0.0
  ABSTAINED   = 1.0 if the example was authored as `unanswerable`, else 0.0.

The judge calls the same `LLMClient` interface as generation; we use it via
`get_llm_client()` by default but accept an injected client for tests.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from evaluation.dataset import QAExample
from generation.llm import LLMClient, Message, get_llm_client
from retrieval.dense import RetrievedChunk

Verdict = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "ABSTAINED"]

_VERDICTS: tuple[Verdict, ...] = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "ABSTAINED")
_VERDICT_RE = re.compile(r"\b(SUPPORTED|PARTIAL|UNSUPPORTED|ABSTAINED)\b", re.IGNORECASE)

_ABSTAIN_MARKER = "i don't have enough information"

_JUDGE_SYSTEM = """\
You are a strict faithfulness judge for a financial-QA system. You see a
question, a candidate answer, and the only context passages the answerer was
allowed to use. Decide whether the answer is grounded in those passages.

Reply on a single line with one of these tokens, then a colon, then a one-
sentence reason:
  SUPPORTED   — every factual claim is directly supported by the passages.
  PARTIAL     — at least one claim supported, at least one not supported.
  UNSUPPORTED — no factual claim is grounded in the passages.
  ABSTAINED   — the answer explicitly declines to answer.

Do not consider whether the answer is *true* in the real world. Only judge
groundedness in the provided passages.
"""


@dataclass(frozen=True)
class JudgeResult:
    verdict: Verdict
    score: float
    reason: str


def _format_chunks(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "(no passages were cited)"
    parts = []
    for c in chunks:
        text = c.payload.get("text", "").strip()
        parts.append(f"[{c.chunk_id}]\n{text}")
    return "\n\n".join(parts)


def judge_answer(
    *,
    example: QAExample,
    answer_text: str,
    cited_chunks: Sequence[RetrievedChunk],
    client: LLMClient | None = None,
) -> JudgeResult:
    # Cheap pre-check for the canonical abstention phrase before paying for an LLM call.
    if _ABSTAIN_MARKER in answer_text.lower():
        score = 1.0 if example.is_unanswerable() else 0.0
        return JudgeResult(verdict="ABSTAINED", score=score, reason="Answer abstained.")

    client = client or get_llm_client()
    user = (
        f"Question:\n{example.question.strip()}\n\n"
        f"Answer:\n{answer_text.strip()}\n\n"
        f"Allowed context passages:\n{_format_chunks(cited_chunks)}\n\n"
        "Verdict:"
    )
    completion = client.complete(
        [Message(role="system", content=_JUDGE_SYSTEM), Message(role="user", content=user)],
        temperature=0.0,
        max_tokens=200,
    )
    verdict, reason = _parse_verdict(completion.text)

    if verdict == "ABSTAINED":
        score = 1.0 if example.is_unanswerable() else 0.0
    elif verdict == "SUPPORTED":
        score = 1.0
    elif verdict == "PARTIAL":
        score = 0.5
    else:
        score = 0.0
    return JudgeResult(verdict=verdict, score=score, reason=reason)


def _parse_verdict(text: str) -> tuple[Verdict, str]:
    match = _VERDICT_RE.search(text)
    verdict: Verdict = match.group(1).upper() if match else "UNSUPPORTED"  # type: ignore[assignment]
    if verdict not in _VERDICTS:
        verdict = "UNSUPPORTED"
    # Reason = the first 240 chars after the verdict token (or the whole reply).
    reason = text.strip().splitlines()[0] if text.strip() else ""
    return verdict, reason[:240]
