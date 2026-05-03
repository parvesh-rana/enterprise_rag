"""Prompt construction with citation enforcement.

Citation contract (enforced on the parser side):
  - Every factual sentence must end with one or more bracketed chunk ids,
    e.g. "Revenue grew 12% YoY [AAPL-2024-7-0003]."
  - Multiple citations: comma-separated inside one bracket pair, e.g. [a, b].
  - The model is allowed to abstain ("I don't know") when the context is
    insufficient — abstention is preferred to fabrication.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from generation.llm import Message
from retrieval.dense import RetrievedChunk

CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_ID_SPLIT_RE = re.compile(r"[,\s]+")

SYSTEM_PROMPT = """\
You are an analyst that answers questions about SEC 10-K filings using ONLY the
context passages provided. Follow these rules without exception:

1. Ground every factual claim in the provided context. Never invent numbers,
   dates, names, or relationships that are not directly supported by a passage.
2. After every sentence that asserts a fact, append a citation in square
   brackets containing the chunk id(s) that support it, e.g. [AAPL-2024-7-0003].
   Multiple ids in one bracket: [AAPL-2024-7-0003, AAPL-2024-7-0004].
3. If the context does not contain enough information to answer, reply
   exactly: "I don't have enough information in the provided filings to answer that."
   Do not speculate.
4. Be concise. 2-5 sentences for most questions; a short bulleted list for
   comparisons. No preamble, no restating the question.
5. Quote sparingly; paraphrase. When you do quote, keep it under ~15 words.
6. Use plain text only. Do NOT use markdown formatting such as bold (**),
   headers (#), or italics (*). Citations in brackets are the only special syntax.
"""

_CONTEXT_HEADER = (
    "Context passages from SEC 10-K filings. "
    "Each passage is preceded by its chunk id; cite by that id.\n"
)


def _format_chunk_block(c: RetrievedChunk) -> str:
    p = c.payload
    header = (
        f"[{c.chunk_id}] "
        f"{p.get('company_name', p.get('company', '?'))} "
        f"({p.get('year', '?')}) — {p.get('section_title', '?')}"
    )
    return f"{header}\n{p.get('text', '').strip()}"


def build_messages(query: str, chunks: Sequence[RetrievedChunk]) -> list[Message]:
    """Assemble the (system, user) message pair for the LLM."""
    if not chunks:
        # No retrieval results: the system prompt's abstention rule will trigger.
        context = "(no passages were retrieved for this query)"
    else:
        context = "\n\n".join(_format_chunk_block(c) for c in chunks)

    user = (
        f"{_CONTEXT_HEADER}\n"
        f"{context}\n\n"
        f"Question: {query.strip()}\n\n"
        "Answer (with citations):"
    )
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user),
    ]


def parse_citations(answer_text: str, *, allowed_ids: set[str]) -> list[str]:
    """Extract chunk ids from the answer.

    - Returns ids in their first-occurrence order (de-duped).
    - Drops any id not in `allowed_ids` (model hallucinated a citation).
    """
    seen: list[str] = []
    found: set[str] = set()
    for match in CITATION_RE.finditer(answer_text):
        for piece in _ID_SPLIT_RE.split(match.group(1).strip()):
            piece = piece.strip()
            if not piece or piece in found:
                continue
            if piece in allowed_ids:
                seen.append(piece)
                found.add(piece)
    return seen
