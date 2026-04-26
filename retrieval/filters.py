"""Metadata pre-filters shared by dense and sparse retrievers.

A `RetrievalFilter` is provider-agnostic; each retriever translates it into
its native form (Qdrant `Filter` for dense, a Python predicate for BM25).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client.http import models as qm


class RetrievalFilter(BaseModel):
    """Optional metadata constraints. None means 'no constraint on this field'."""

    company: str | None = Field(default=None, description="Ticker, e.g. 'AAPL'")
    year: int | None = None
    item: str | None = Field(default=None, description="10-K Item key, e.g. '1A'")

    def is_empty(self) -> bool:
        return self.company is None and self.year is None and self.item is None

    def to_qdrant(self) -> qm.Filter | None:
        if self.is_empty():
            return None
        must: list[qm.FieldCondition] = []
        if self.company is not None:
            must.append(
                qm.FieldCondition(key="company", match=qm.MatchValue(value=self.company))
            )
        if self.year is not None:
            must.append(qm.FieldCondition(key="year", match=qm.MatchValue(value=self.year)))
        if self.item is not None:
            must.append(qm.FieldCondition(key="item", match=qm.MatchValue(value=self.item)))
        return qm.Filter(must=must)

    def to_predicate(self) -> Callable[[dict[str, Any]], bool]:
        if self.is_empty():
            return lambda _p: True

        company = self.company
        year = self.year
        item = self.item

        def _pred(p: dict[str, Any]) -> bool:
            if company is not None and p.get("company") != company:
                return False
            if year is not None and p.get("year") != year:
                return False
            if item is not None and p.get("item") != item:
                return False
            return True

        return _pred
