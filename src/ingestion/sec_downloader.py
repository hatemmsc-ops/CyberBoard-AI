"""Download 10-K and 10-Q filings from SEC EDGAR."""

import csv
import time
import requests
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"

HEADERS = {"User-Agent": config.SEC_EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def get_filings_metadata(cik: str, filing_types: list[str] = None, max_filings: int = 8) -> list[dict]:
    """Fetch filing metadata from EDGAR submissions endpoint."""
    if filing_types is None:
        filing_types = ["10-K", "10-Q"]

    url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for form, date, accession, doc in zip(forms, dates, accessions, primary_docs):
        if form in filing_types and len(results) < max_filings:
            results.append({
                "form": form,
                "date": date,
                "accession": accession.replace("-", ""),
                "accession_display": accession,
                "primary_doc": doc,
            })
    return results


def download_filing(cik: str, filing: dict, output_dir: Path) -> Optional[Path]:
    """Download a single filing document."""
    url = EDGAR_ARCHIVES_URL.format(
        cik=cik.lstrip("0"),
        accession=filing["accession"],
        filename=filing["primary_doc"],
    )
    resp = requests.get(url, headers=HEADERS, timeout=60)
    if resp.status_code != 200:
        print(f"  Failed to download {url} (status {resp.status_code})")
        return None

    ext = Path(filing["primary_doc"]).suffix or ".html"
    filename = f"{filing['form']}_{filing['date']}{ext}"
    filepath = output_dir / filename
    filepath.write_bytes(resp.content)
    return filepath


def download_all(tickers: dict[str, str] = None, filing_types: list[str] = None, max_per_type: int = 4):
    """Download filings for all companies. max_per_type=4 gets ~2 fiscal years of 10-K and 10-Q."""
    if tickers is None:
        tickers = config.COMPANIES
    if filing_types is None:
        filing_types = ["10-K", "10-Q"]

    manifest_path = config.SEC_FILINGS_DIR / "manifest.csv"
    config.SEC_FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for ticker, cik in tickers.items():
        print(f"\n--- {ticker} (CIK: {cik}) ---")
        ticker_dir = config.SEC_FILINGS_DIR / ticker
        ticker_dir.mkdir(exist_ok=True)

        try:
            filings = get_filings_metadata(cik, filing_types, max_filings=max_per_type * len(filing_types))
        except Exception as e:
            print(f"  Error fetching metadata: {e}")
            continue

        counts = {ft: 0 for ft in filing_types}
        for filing in filings:
            ft = filing["form"]
            if counts.get(ft, 0) >= max_per_type:
                continue

            time.sleep(0.15)  # SEC rate limit: 10 requests/sec
            filepath = download_filing(cik, filing, ticker_dir)
            if filepath:
                counts[ft] = counts.get(ft, 0) + 1
                manifest_rows.append({
                    "ticker": ticker,
                    "cik": cik,
                    "filing_type": ft,
                    "date": filing["date"],
                    "accession": filing["accession_display"],
                    "file_path": str(filepath.relative_to(config.PROJECT_ROOT)),
                })
                print(f"  Downloaded {ft} ({filing['date']})")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "cik", "filing_type", "date", "accession", "file_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest saved: {manifest_path}")
    print(f"Total filings downloaded: {len(manifest_rows)}")
    return manifest_rows


if __name__ == "__main__":
    download_all()
