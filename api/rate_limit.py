"""In-memory token-bucket rate limiter.

Per-client (keyed by IP), in-process only. This is appropriate for a single-
node demo; real deployments would back this with Redis. The bucket holds at
most `rate_per_minute` tokens and refills continuously.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from core.config import get_settings


@dataclass
class _Bucket:
    capacity: float
    tokens: float
    refill_per_sec: float
    last: float = field(default_factory=time.monotonic)


class RateLimiter:
    def __init__(self, rate_per_minute: int | None = None) -> None:
        rate = rate_per_minute or get_settings().rate_limit_per_minute
        self._capacity: float = float(rate)
        self._refill_per_sec: float = rate / 60.0
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> bool:
        """Consume one token; return True if allowed, False if exhausted."""
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(
                    capacity=self._capacity,
                    tokens=self._capacity,
                    refill_per_sec=self._refill_per_sec,
                )
                self._buckets[key] = b
            elapsed = now - b.last
            b.tokens = min(b.capacity, b.tokens + elapsed * b.refill_per_sec)
            b.last = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True
            return False
