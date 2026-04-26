"""Tests for index.bm25 and retrieval.sparse — no Qdrant or torch required."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import Chunk
from index.bm25 import BM25Index, tokenize


def _chunk(
    cid: str, text: str, *, company: str = "ACME", year: int = 2024, item: str = "1"
) -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        company=company,
        company_name=f"{company} Inc.",
        year=year,
        item=item,  # type: ignore[arg-type]
        section_title=f"Item {item}",
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        source_url="https://example.com",
    )


def test_tokenizer_lowercases_and_drops_stopwords() -> None:
    toks = tokenize("The Quick brown fox JUMPS over the lazy dog.")
    assert "quick" in toks
    assert "jumps" in toks
    assert "the" not in toks  # stopword
    assert "for" not in toks  # stopword
    assert "by" not in toks  # stopword


def test_tokenizer_keeps_financial_token_shapes() -> None:
    toks = tokenize("Revenue grew 12.5% year-over-year on a 10-K basis.")
    assert "12.5" in toks
    assert "year-over-year" in toks
    assert "10-k" in toks


def test_bm25_ranks_keyword_match_highest() -> None:
    chunks = [
        _chunk("c1", "We design rocket-powered roller skates for professional coyotes."),
        _chunk("c2", "Our supply chain depends on a single Acme spring manufacturer."),
        _chunk("c3", "Fiscal 2024 revenue rose twelve percent on stronger anvil sales."),
    ]
    idx = BM25Index.build(chunks)
    hits = idx.search("anvil revenue", top_k=3)
    assert hits[0].chunk_id == "c3"


def test_bm25_returns_no_hits_for_empty_query() -> None:
    chunks = [_chunk("c1", "hello world")]
    idx = BM25Index.build(chunks)
    assert idx.search("", top_k=5) == []
    assert idx.search("the and of", top_k=5) == []  # all stopwords


def test_bm25_predicate_filters_before_truncation() -> None:
    # BM25 needs decoy docs that don't contain the query term so IDF stays
    # positive on tiny corpora. Without them, df ≈ N → idf goes negative and
    # search() drops zero-scored hits.
    chunks = [
        _chunk("a1", "anvils everywhere", company="AAPL", year=2024),
        _chunk("a2", "anvils for iphones", company="AAPL", year=2023),
        _chunk("m1", "anvils at microsoft", company="MSFT", year=2024),
        _chunk("d1", "feathers", company="AAPL", year=2024),
        _chunk("d2", "rockets", company="AAPL", year=2024),
        _chunk("d3", "skates", company="AAPL", year=2024),
        _chunk("d4", "banner", company="MSFT", year=2024),
        _chunk("d5", "telescope", company="AAPL", year=2024),
    ]
    idx = BM25Index.build(chunks)
    hits = idx.search("anvils", top_k=5, predicate=lambda p: p["company"] == "AAPL")
    assert {h.chunk_id for h in hits} == {"a1", "a2"}


def test_bm25_top_k_truncates() -> None:
    # 10 docs with the query term + 30 decoys keeps IDF positive (df < N/2).
    chunks = [_chunk(f"a{i}", "anvil") for i in range(10)] + [
        _chunk(f"d{i}", f"decoy{i}") for i in range(30)
    ]
    idx = BM25Index.build(chunks)
    assert len(idx.search("anvil", top_k=5)) == 5


def test_bm25_save_load_roundtrip(tmp_path: Path) -> None:
    chunks = [
        _chunk("c1", "rocket roller skates"),
        _chunk("c2", "anvil casting line yields"),
        _chunk("c3", "fiscal year revenue summary"),
        _chunk("c4", "geographic segment breakdown"),
    ]
    idx = BM25Index.build(chunks)
    out = tmp_path / "bm25.pkl"
    idx.save(out)

    reloaded = BM25Index.load(out)
    hits = reloaded.search("anvil", top_k=2)
    assert hits[0].chunk_id == "c2"
    assert reloaded.chunk_ids == idx.chunk_ids


def test_bm25_score_is_zero_for_unrelated_query() -> None:
    chunks = [_chunk("c1", "rocket roller skates and anvils")]
    idx = BM25Index.build(chunks)
    hits = idx.search("quarterly dividend yield", top_k=5)
    # No tokens match, so either no hits or a 0-scored hit; the search()
    # contract drops zero-scored hits.
    assert hits == []


@pytest.mark.parametrize(
    "query,expected_top",
    [
        ("rocket roller skates", "c1"),
        ("supply chain spring manufacturer", "c2"),
        ("revenue growth fiscal year", "c3"),
    ],
)
def test_bm25_topical_ranking(query: str, expected_top: str) -> None:
    chunks = [
        _chunk("c1", "We design rocket-powered roller skates for desert customers."),
        _chunk("c2", "Our supply chain depends on a single Acme spring manufacturer."),
        _chunk("c3", "Fiscal 2024 revenue rose twelve percent year over year."),
    ]
    idx = BM25Index.build(chunks)
    hits = idx.search(query, top_k=1)
    assert hits[0].chunk_id == expected_top
