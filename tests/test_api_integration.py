"""End-to-end API integration test.

Stands up the FastAPI app against a tiny fixture corpus, swaps the LLM and
the dense retriever for fakes, and verifies:

  - /health reports BM25 loaded
  - /query returns an answer with valid citations and per-stage timings
  - /sources/{id} returns the full chunk payload
  - /metrics exposes the Prometheus registry
  - rate limiter rejects requests beyond its budget

Qdrant is not started; the dense retriever is monkeypatched so the real network
call never fires. The BM25 index is built in-process from the fixture filing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import create_app
from core.types import FilingDoc
from generation.llm import Completion, LLMClient, Message
from index.bm25 import BM25Index
from ingestion.chunker import ChunkerConfig, chunk_filing
from retrieval.dense import RetrievedChunk


# ---------- fakes & helpers ----------

class StubLLM(LLMClient):
    """Echoes a fixed answer that cites whatever id appears first in the user prompt."""

    last_messages: list[Message] | None = None

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        StubLLM.last_messages = messages
        # Pull the first chunk id mentioned in the user message.
        user = messages[-1].content
        first_id = None
        if "[" in user and "]" in user:
            first_id = user[user.index("[") + 1 : user.index("]")]
        cite = f"[{first_id}]" if first_id else ""
        return Completion(
            text=f"This is a stub answer grounded in the provided context {cite}.",
            model="stub-model",
            prompt_tokens=12,
            completion_tokens=8,
        )


def _build_index(doc: FilingDoc) -> BM25Index:
    chunks = chunk_filing(doc, ChunkerConfig(max_tokens=80, min_tokens=10, overlap_tokens=10))
    return BM25Index.build(chunks)


def _stub_dense(_query: str, *, top_k: int, filt: Any) -> list[RetrievedChunk]:
    """Empty dense results: the test relies on BM25 + RRF to surface chunks."""
    return []


# ---------- fixtures ----------

@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tiny_filing_doc: FilingDoc) -> Iterator[FastAPI]:
    bm25 = _build_index(tiny_filing_doc)

    # Replace dense_search at the call site (retrieval.pipeline imported it).
    import retrieval.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "dense_search", _stub_dense)

    # Skip the cross-encoder model load by short-circuiting the pipeline's reranker call.
    monkeypatch.setattr(
        pipeline_mod,
        "rerank",
        lambda _q, cands, top_k=None: list(cands)[: (top_k or 5)],
    )

    # Make sparse_search read from this BM25 instance instead of disk.
    import retrieval.sparse as sparse_mod

    monkeypatch.setattr(sparse_mod, "_load_index", lambda: bm25)

    # Stand up the app and overwrite lifespan-built state with our fakes.
    app = create_app()

    with TestClient(app) as client:
        app.state.bm25_index = bm25
        app.state.llm_client = StubLLM()
        # Loosen the limiter so the smoke tests don't trip it accidentally.
        from api.rate_limit import RateLimiter

        app.state.rate_limiter = RateLimiter(rate_per_minute=120)
        app.test_client_holder = client  # type: ignore[attr-defined]
        yield app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return app.test_client_holder  # type: ignore[attr-defined]


# ---------- tests ----------

def test_health_reports_bm25_loaded(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["bm25"] is True
    assert body["llm_provider"] in {"nvidia", "anthropic", "ollama"}
    # Qdrant isn't running in this test, so the overall status is "degraded".
    assert body["status"] in {"ok", "degraded"}


def test_query_returns_answer_with_citation_and_timings(client: TestClient) -> None:
    r = client.post(
        "/query",
        json={"question": "What does ACME make and sell?", "top_k": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["answer"].startswith("This is a stub answer")
    assert body["model"] == "stub-model"
    assert body["request_id"]
    assert body["citations"], "expected at least one validated citation"
    cited_id = body["citations"][0]["chunk_id"]
    retrieved_ids = {c["chunk_id"] for c in body["retrieved"]}
    assert cited_id in retrieved_ids

    timings = body["timings_ms"]
    assert {"retrieval", "generation", "total"} <= set(timings)
    assert timings["total"] >= 0.0


def test_query_validates_request_body(client: TestClient) -> None:
    r = client.post("/query", json={"question": "x"})  # too short
    assert r.status_code == 422


def test_query_request_id_is_propagated(client: TestClient) -> None:
    r = client.post(
        "/query",
        json={"question": "What does ACME make?"},
        headers={"x-request-id": "fixed-id-123"},
    )
    assert r.headers["x-request-id"] == "fixed-id-123"
    assert r.json()["request_id"] == "fixed-id-123"


def test_sources_endpoint_returns_full_text(client: TestClient, app: FastAPI) -> None:
    sample_id = app.state.bm25_index.chunk_ids[0]
    r = client.get(f"/sources/{sample_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["chunk_id"] == sample_id
    assert body["text"]
    assert body["company"] == "ACME"


def test_sources_endpoint_404_on_missing_id(client: TestClient) -> None:
    r = client.get("/sources/does-not-exist")
    assert r.status_code == 404


def test_metrics_endpoint_exposes_prometheus_format(client: TestClient) -> None:
    # Hit /query first so counters are non-zero.
    client.post("/query", json={"question": "What does ACME make and sell?"})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "rag_requests_total" in body
    assert "rag_request_latency_seconds" in body
    assert "rag_retrieval_latency_seconds" in body
    assert "rag_generation_latency_seconds" in body


def test_rate_limiter_rejects_excess_requests(client: TestClient, app: FastAPI) -> None:
    from api.rate_limit import RateLimiter

    # Tighten the limiter to 2/minute and slam it.
    app.state.rate_limiter = RateLimiter(rate_per_minute=2)
    payload = {"question": "What does ACME make and sell?"}

    r1 = client.post("/query", json=payload)
    r2 = client.post("/query", json=payload)
    r3 = client.post("/query", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
