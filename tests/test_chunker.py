"""Unit tests for ingestion.parser + ingestion.chunker."""

from __future__ import annotations

from itertools import pairwise

import pytest

from core.types import Chunk, FilingDoc, Section
from ingestion.chunker import ChunkerConfig, chunk_filing
from ingestion.parser import html_to_text

# ---------- parser ----------


def test_html_to_text_strips_script_and_style(tiny_filing_html: str) -> None:
    text = html_to_text(tiny_filing_html)
    assert "console.log" not in text
    assert "font-family" not in text
    assert "ACME Corporation" in text


def test_html_to_text_normalizes_nbsp(tiny_filing_html: str) -> None:
    text = html_to_text(tiny_filing_html)
    # NBSP between "December" and "31" should be normalized to a regular space.
    assert "December 31, 2024" in text
    assert "\u00a0" not in text


def test_find_sections_dedupes_toc_entries(tiny_filing_doc: FilingDoc) -> None:
    items = [s.item for s in tiny_filing_doc.sections]
    # Each Item should appear exactly once even though the ToC also lists them.
    assert items.count("1") == 1
    assert items.count("1A") == 1
    assert items.count("7") == 1


def test_sections_are_ordered_and_contiguous(tiny_filing_doc: FilingDoc) -> None:
    secs = tiny_filing_doc.sections
    assert secs == sorted(secs, key=lambda s: s.start)
    for a, b in pairwise(secs):
        assert a.end == b.start, "sections must tile the text without gaps or overlaps"
    assert secs[-1].end == len(tiny_filing_doc.text)


# ---------- chunker ----------


def test_chunks_respect_section_boundaries(tiny_filing_doc: FilingDoc) -> None:
    chunks = chunk_filing(
        tiny_filing_doc, ChunkerConfig(max_tokens=80, min_tokens=10, overlap_tokens=10)
    )
    by_item: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_item.setdefault(c.item, []).append(c)
    assert {"1", "1A", "7"}.issubset(by_item.keys())

    # No chunk's [char_start, char_end) may straddle two sections.
    for c in chunks:
        owning = [s for s in tiny_filing_doc.sections if s.start <= c.char_start < s.end]
        assert len(owning) == 1
        assert c.char_end <= owning[0].end


def test_chunk_ids_are_unique_and_well_formed(tiny_filing_doc: FilingDoc) -> None:
    chunks = chunk_filing(tiny_filing_doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.id.startswith(f"{c.company}-{c.year}-{c.item}-")


def test_token_budget_is_respected() -> None:
    # Build a synthetic doc with one section full of paragraphs.
    paragraphs = [" ".join(["word"] * 50) for _ in range(20)]
    body = "\n\n".join(paragraphs)
    doc = FilingDoc(
        company="ACME",
        company_name="ACME",
        year=2024,
        cik="0",
        accession="x",
        source_url="x",
        text=body,
        sections=[Section(item="1", title="Item 1", start=0, end=len(body))],
    )
    cfg = ChunkerConfig(max_tokens=120, min_tokens=20, overlap_tokens=20)
    chunks = chunk_filing(doc, cfg)
    assert chunks, "expected at least one chunk"
    # Every chunk stays within max_tokens + overlap headroom.
    for c in chunks:
        assert c.token_count <= cfg.max_tokens + cfg.overlap_tokens


def test_oversized_paragraph_is_hard_wrapped() -> None:
    big_para = " ".join(["lorem"] * 1000)  # one paragraph, far over the budget
    doc = FilingDoc(
        company="ACME",
        company_name="ACME",
        year=2024,
        cik="0",
        accession="x",
        source_url="x",
        text=big_para,
        sections=[Section(item="1", title="Item 1", start=0, end=len(big_para))],
    )
    cfg = ChunkerConfig(max_tokens=200, min_tokens=20, overlap_tokens=20)
    chunks = chunk_filing(doc, cfg)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= cfg.max_tokens + cfg.overlap_tokens


def test_overlap_does_not_cross_section_boundaries(tiny_filing_doc: FilingDoc) -> None:
    chunks = chunk_filing(
        tiny_filing_doc, ChunkerConfig(max_tokens=60, min_tokens=10, overlap_tokens=20)
    )
    # The first chunk of each section must not begin with text drawn from a
    # different section. We approximate by checking that the first chunk's
    # text appears verbatim inside its own section span.
    for item in {"1", "1A", "7"}:
        section_chunks = [c for c in chunks if c.item == item]
        section = next(s for s in tiny_filing_doc.sections if s.item == item)
        section_text = tiny_filing_doc.text[section.start : section.end]
        first_words = " ".join(section_chunks[0].text.split()[:5])
        assert first_words in " ".join(section_text.split())


def test_chunker_config_validation() -> None:
    with pytest.raises(ValueError):
        ChunkerConfig(max_tokens=100, overlap_tokens=200)
    with pytest.raises(ValueError):
        ChunkerConfig(max_tokens=100, min_tokens=200)


def test_chunker_handles_empty_sections() -> None:
    # A filing whose sections array is empty falls back to one big "UNKNOWN" section.
    doc = FilingDoc(
        company="ACME",
        company_name="ACME",
        year=2024,
        cik="0",
        accession="x",
        source_url="x",
        text="Just some prose. " * 50,
        sections=[],
    )
    chunks = chunk_filing(doc)
    assert chunks
    assert all(c.item == "UNKNOWN" for c in chunks)
