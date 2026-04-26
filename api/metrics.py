"""Prometheus metrics. One module-level registry; routes import the metrics
they touch and increment/observe directly."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# Buckets tuned for an RAG workload: dense + reranker is typically 100-800 ms,
# generation 500 ms - 5 s. Don't bother with sub-10 ms buckets.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

REQUESTS = Counter(
    "rag_requests_total",
    "Total HTTP requests handled.",
    labelnames=("endpoint", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end HTTP request latency.",
    labelnames=("endpoint",),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Hybrid retrieval (dense + sparse + RRF + rerank) latency.",
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency.",
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "LLM token counts by kind.",
    labelnames=("kind",),  # "prompt" | "completion"
    registry=REGISTRY,
)

RATE_LIMITED = Counter(
    "rag_rate_limited_total",
    "Requests rejected by the in-memory rate limiter.",
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
