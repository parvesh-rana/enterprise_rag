"""FastAPI dependencies: rate limiter, retriever, LLM client, BM25 store.

Singletons live on `app.state` and are wired in api.main.lifespan; deps here
just project them out of `request.app.state` so handlers stay testable.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from api.metrics import RATE_LIMITED
from api.rate_limit import RateLimiter
from generation.llm import LLMClient
from index.bm25 import BM25Index


def _client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter  # type: ignore[no-any-return]


def enforce_rate_limit(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    if not limiter.allow(_client_key(request)):
        RATE_LIMITED.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
        )


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm_client  # type: ignore[no-any-return]


def get_bm25(request: Request) -> BM25Index | None:
    """Return the BM25 index attached at startup, or None if unavailable."""
    return getattr(request.app.state, "bm25_index", None)
