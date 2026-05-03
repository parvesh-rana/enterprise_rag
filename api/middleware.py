"""Request middleware: assign a request_id, bind it to structlog's contextvars,
emit one JSON access log per request, and observe Prometheus latency."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from structlog.contextvars import bind_contextvars, clear_contextvars

from api.metrics import REQUEST_LATENCY, REQUESTS
from core.logging import get_logger

log = get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Inject request_id, time the request, log, and update Prometheus."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        clear_contextvars()
        bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)

        # Make the id available to handlers via request.state.
        request.state.request_id = request_id

        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            endpoint = _normalize_endpoint(request.url.path)
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            REQUESTS.labels(endpoint=endpoint, status=status).inc()
            log.info(
                "http.access",
                status=status,
                latency_ms=round(elapsed * 1000, 2),
                endpoint=endpoint,
            )
            clear_contextvars()


def _normalize_endpoint(path: str) -> str:
    """Collapse dynamic segments so per-id paths don't blow up label cardinality."""
    if path.startswith("/sources/"):
        return "/sources/{id}"
    return path
