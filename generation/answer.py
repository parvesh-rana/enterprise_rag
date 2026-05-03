"""End-to-end answer generation: retrieve → prompt → LLM → parse citations."""

from __future__ import annotations

from collections.abc import Sequence

from core.logging import get_logger
from core.types import Answer, Citation
from generation.llm import Completion, LLMClient, get_llm_client
from generation.prompts import build_messages, parse_citations
from retrieval.dense import RetrievedChunk

log = get_logger(__name__)


def _score_by_id(chunks: Sequence[RetrievedChunk]) -> dict[str, float]:
    return {c.chunk_id: c.score for c in chunks}


def answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    request_id: str,
    client: LLMClient | None = None,
) -> Answer:
    """Build the prompt, call the LLM, and parse out validated citations."""
    client = client or get_llm_client()
    messages = build_messages(query, chunks)
    completion: Completion = client.complete(messages)

    allowed = {c.chunk_id for c in chunks}
    cited_ids = parse_citations(completion.text, allowed_ids=allowed)
    score_map = _score_by_id(chunks)

    citations = [Citation(chunk_id=cid, score=score_map.get(cid, 0.0)) for cid in cited_ids]

    log.info(
        "answer.done",
        request_id=request_id,
        n_chunks=len(chunks),
        n_citations=len(citations),
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        model=completion.model,
    )

    return Answer(
        text=completion.text,
        citations=citations,
        model=completion.model,
        request_id=request_id,
    )
