"""SEC EDGAR downloader.

EDGAR rules we obey:
  - Send a real User-Agent (set via EDGAR_USER_AGENT). SEC blocks empty UAs.
  - Rate-limited to ~10 req/s; we go well below.
  - Use the JSON submissions feed to discover the latest 10-K accession,
    then fetch the primary document.

We download once and cache to data/raw/<TICKER>/<accession>/<file>. Subsequent
runs short-circuit if the file already exists.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

# Hand-maintained CIK map for the five companies in the brief.
# CIKs are stable; the SEC company_tickers.json is a fallback if we ever expand.
COMPANY_CIK: dict[str, tuple[str, str]] = {
    # ticker -> (zero-padded CIK, display name)
    "AAPL": ("0000320193", "Apple Inc."),
    "MSFT": ("0000789019", "Microsoft Corporation"),
    "AMZN": ("0001018724", "Amazon.com, Inc."),
    "TSLA": ("0001318605", "Tesla, Inc."),
    "NVDA": ("0001045810", "NVIDIA Corporation"),
}

EDGAR_DATA_HOST = "https://data.sec.gov"
EDGAR_ARCHIVE_HOST = "https://www.sec.gov"


@dataclass(frozen=True)
class FilingRef:
    """Pointer to a specific 10-K filing on EDGAR."""

    ticker: str
    company_name: str
    cik: str
    accession: str  # with dashes, e.g. "0000320193-24-000123"
    primary_document: str  # filename of the main 10-K HTML
    fiscal_year: int
    filing_date: str  # ISO date

    @property
    def primary_url(self) -> str:
        no_dashes = self.accession.replace("-", "")
        return (
            f"{EDGAR_ARCHIVE_HOST}/Archives/edgar/data/"
            f"{int(self.cik)}/{no_dashes}/{self.primary_document}"
        )


def _headers() -> dict[str, str]:
    ua = get_settings().edgar_user_agent.strip()
    if not ua:
        raise RuntimeError("EDGAR_USER_AGENT must be set; SEC requires identification.")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=10))
async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    resp = await client.get(url, headers=_headers(), timeout=30.0)
    resp.raise_for_status()
    return resp


async def latest_10k(client: httpx.AsyncClient, ticker: str) -> FilingRef:
    """Return the most recent 10-K filing reference for `ticker`."""
    refs = await recent_10ks(client, ticker, n=1)
    return refs[0]


async def recent_10ks(client: httpx.AsyncClient, ticker: str, n: int = 1) -> list[FilingRef]:
    """Return the `n` most recent 10-K filing references for `ticker`."""
    if ticker not in COMPANY_CIK:
        raise KeyError(f"Unknown ticker {ticker!r}; add to COMPANY_CIK")
    cik, name = COMPANY_CIK[ticker]

    submissions_url = f"{EDGAR_DATA_HOST}/submissions/CIK{cik}.json"
    data = (await _get(client, submissions_url)).json()
    recent = data["filings"]["recent"]

    forms: list[str] = recent["form"]
    accessions: list[str] = recent["accessionNumber"]
    docs: list[str] = recent["primaryDocument"]
    dates: list[str] = recent["filingDate"]

    results: list[FilingRef] = []
    for form, acc, doc, date in zip(forms, accessions, docs, dates, strict=True):
        if form == "10-K":
            results.append(FilingRef(
                ticker=ticker,
                company_name=name,
                cik=cik,
                accession=acc,
                primary_document=doc,
                fiscal_year=int(date.split("-")[0]),
                filing_date=date,
            ))
            if len(results) >= n:
                break

    if not results:
        raise LookupError(f"No 10-K found for {ticker}")
    return results


async def download_filing(client: httpx.AsyncClient, ref: FilingRef, dest_root: Path) -> Path:
    """Download the primary 10-K HTML to disk; return the local path."""
    out_dir = dest_root / ref.ticker / ref.accession.replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ref.primary_document

    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("edgar.cache_hit", ticker=ref.ticker, path=str(out_path))
        return out_path

    log.info("edgar.download", ticker=ref.ticker, url=ref.primary_url)
    resp = await _get(client, ref.primary_url)
    out_path.write_bytes(resp.content)
    return out_path


async def fetch_all(
    tickers: list[str], dest_root: Path, years: int = 1
) -> list[tuple[FilingRef, Path]]:
    """Concurrently fetch the last `years` 10-Ks for each ticker."""
    sem = asyncio.Semaphore(3)

    async with httpx.AsyncClient() as client:

        async def one(t: str) -> list[tuple[FilingRef, Path]]:
            async with sem:
                refs = await recent_10ks(client, t, n=years)
                pairs: list[tuple[FilingRef, Path]] = []
                for ref in refs:
                    path = await download_filing(client, ref, dest_root)
                    pairs.append((ref, path))
                return pairs

        nested = await asyncio.gather(*(one(t) for t in tickers))
        return [pair for sublist in nested for pair in sublist]
