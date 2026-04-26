"""FastAPI app + lifespan.

Lifespan is responsible for:
  1. Configuring structlog once.
  2. Loading the BM25 index from disk (if present).
  3. Building the LLM client (lazy — provider import happens on demand).
  4. Constructing the rate limiter.

Anything that can fail (BM25 missing, LLM key missing) is logged and the app
still boots; /health will report the degradation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware import RequestContextMiddleware
from api.rate_limit import RateLimiter
from api.routes import router
from core.config import get_settings
from core.logging import configure_logging, get_logger
from generation.llm import Completion, LLMClient, Message

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    log.info("api.startup", provider=settings.llm_provider, port=settings.api_port)

    # Rate limiter — always available.
    app.state.rate_limiter = RateLimiter()

    # BM25 — optional (eg. fresh checkout before `make index`).
    bm25_path = settings.bm25_dir / "index.pkl"
    if bm25_path.exists():
        from index.bm25 import BM25Index

        try:
            app.state.bm25_index = BM25Index.load(bm25_path)
            log.info("api.bm25_loaded", path=str(bm25_path), n=len(app.state.bm25_index.chunk_ids))
        except Exception as exc:  # pragma: no cover - corrupt index
            log.error("api.bm25_load_failed", err=str(exc))
            app.state.bm25_index = None
    else:
        log.warning("api.bm25_missing", path=str(bm25_path))
        app.state.bm25_index = None

    # LLM client — lazy: build now so we fail fast on misconfig in non-test paths.
    try:
        from generation.llm import get_llm_client

        app.state.llm_client = get_llm_client()
    except Exception as exc:  # pragma: no cover
        log.error("api.llm_init_failed", err=str(exc))
        app.state.llm_client = _UnconfiguredLLM(str(exc))

    yield

    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app


class _UnconfiguredLLM(LLMClient):
    """Sentinel client used when LLM init fails. /query will surface a clean 503."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM unavailable: {self.reason}",
        )


app = create_app()
