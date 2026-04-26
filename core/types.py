"""Shared data types used across ingestion, indexing, retrieval, and generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# 10-K Items we extract. Item 7A and 9A include alphabetic suffixes; the parser
# normalizes to these canonical strings.
ItemKey = Literal[
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "UNKNOWN",
]


class FilingDoc(BaseModel):
    """A parsed 10-K filing: full text plus the Item-level section spans."""

    company: str  # canonical ticker (e.g. "AAPL")
    company_name: str  # display name (e.g. "Apple Inc.")
    year: int  # fiscal year of the filing
    cik: str  # SEC CIK, zero-padded
    accession: str  # SEC accession number
    source_url: str  # canonical URL on EDGAR
    text: str  # cleaned plain text of the entire filing
    sections: list[Section]


class Section(BaseModel):
    """A contiguous span of `FilingDoc.text` belonging to one 10-K Item."""

    item: ItemKey
    title: str  # original heading, e.g. "Item 1A. Risk Factors"
    start: int  # inclusive char offset into FilingDoc.text
    end: int  # exclusive char offset into FilingDoc.text


class Chunk(BaseModel):
    """A retrievable unit of text with provenance back to its filing + section."""

    id: str = Field(..., description="Stable id: f'{company}-{year}-{item}-{ordinal:04d}'")
    text: str
    company: str
    company_name: str
    year: int
    item: ItemKey
    section_title: str
    char_start: int
    char_end: int
    token_count: int
    source_url: str

    @property
    def metadata(self) -> dict[str, str | int]:
        """Flat metadata payload suitable for Qdrant filtering."""
        return {
            "company": self.company,
            "year": self.year,
            "item": self.item,
            "section_title": self.section_title,
        }


class Citation(BaseModel):
    chunk_id: str
    score: float
    quote: str | None = None


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    model: str
    request_id: str
