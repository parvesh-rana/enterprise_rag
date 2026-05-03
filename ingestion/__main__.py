"""`python -m ingestion` (a.k.a. `make ingest`).

Default behavior: download the latest 10-K for AAPL, MSFT, AMZN, TSLA, NVDA,
parse them, chunk them, and persist chunks as JSONL under data/chunks/.

If EDGAR is unreachable (offline reviewers), falls back to any HTML files
present under data/sample/ so the pipeline still completes end-to-end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import orjson
from tqdm import tqdm

from core.config import get_settings
from core.logging import configure_logging, get_logger
from core.types import Chunk
from ingestion.chunker import ChunkerConfig, chunk_filing
from ingestion.edgar import COMPANY_CIK, fetch_all
from ingestion.parser import parse_filing

log = get_logger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"]


def _write_chunks(out_dir: Path, ticker: str, year: int, chunks: list[Chunk]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_{year}.jsonl"
    with path.open("wb") as fh:
        for c in chunks:
            fh.write(orjson.dumps(c.model_dump()))
            fh.write(b"\n")
    return path


async def _ingest_from_edgar(tickers: list[str], years: int = 1) -> int:
    settings = get_settings()
    refs_paths = await fetch_all(tickers, settings.raw_dir, years=years)
    total = 0
    for ref, html_path in tqdm(refs_paths, desc="parse+chunk"):
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        doc = parse_filing(
            html=html,
            company=ref.ticker,
            company_name=ref.company_name,
            year=ref.fiscal_year,
            cik=ref.cik,
            accession=ref.accession,
            source_url=ref.primary_url,
        )
        chunks = chunk_filing(doc, ChunkerConfig())
        path = _write_chunks(settings.chunks_dir, ref.ticker, ref.fiscal_year, chunks)
        log.info(
            "ingest.write",
            ticker=ref.ticker,
            year=ref.fiscal_year,
            n_sections=len(doc.sections),
            n_chunks=len(chunks),
            path=str(path),
        )
        total += len(chunks)
    return total


def _ingest_from_sample() -> int:
    """Fallback: parse any HTML files in data/sample/. Metadata is best-effort."""
    settings = get_settings()
    htmls = sorted(settings.sample_dir.glob("*.html")) + sorted(settings.sample_dir.glob("*.htm"))
    if not htmls:
        log.warning("ingest.no_sample", dir=str(settings.sample_dir))
        return 0
    total = 0
    for html_path in tqdm(htmls, desc="parse+chunk (sample)"):
        ticker = html_path.stem.split("_")[0].upper() or "SAMPLE"
        cik, company_name = COMPANY_CIK.get(ticker, ("0000000000", "Sample Company"))
        # Try to extract year from filename like "AAPL_2024.html"; default to 0.
        year_part = next((p for p in html_path.stem.split("_") if p.isdigit()), "0")
        doc = parse_filing(
            html=html_path.read_text(encoding="utf-8", errors="ignore"),
            company=ticker,
            company_name=company_name,
            year=int(year_part),
            cik=cik,
            accession="sample",
            source_url=f"file://{html_path}",
        )
        chunks = chunk_filing(doc, ChunkerConfig())
        _write_chunks(settings.chunks_dir, ticker, int(year_part), chunks)
        total += len(chunks)
    return total


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="ingestion")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to download from EDGAR.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Number of most recent 10-K filings per ticker to ingest (e.g. 4 for last 4 years).",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Skip EDGAR; ingest only files under data/sample/.",
    )
    args = parser.parse_args()

    if args.sample_only:
        n = _ingest_from_sample()
    else:
        try:
            n = asyncio.run(_ingest_from_edgar(args.tickers, years=args.years))
        except Exception as exc:  # network failures, EDGAR rate limits, etc.
            log.error("ingest.edgar_failed", err=str(exc))
            log.info("ingest.fallback_sample")
            n = _ingest_from_sample()

    print(json.dumps({"chunks_written": n}))


if __name__ == "__main__":
    main()
