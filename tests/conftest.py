"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import FilingDoc
from ingestion.parser import parse_filing

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_filing_html() -> str:
    return (FIXTURES / "tiny_filing.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def tiny_filing_doc(tiny_filing_html: str) -> FilingDoc:
    return parse_filing(
        html=tiny_filing_html,
        company="ACME",
        company_name="ACME Corporation",
        year=2024,
        cik="0000000000",
        accession="0000000000-24-000001",
        source_url="https://example.com/acme-10k.html",
    )
