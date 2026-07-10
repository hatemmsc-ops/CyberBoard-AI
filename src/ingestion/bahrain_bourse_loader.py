"""Placeholder for Bahrain Bourse annual report ingestion.

Bahrain Bourse does not have a public API for bulk report downloads.
Reports must be manually collected from https://www.bahrainbourse.com/
and placed in data/bahrain_bourse/{company_name}/.

Supported formats: PDF (parsed via pdfplumber), HTML.
This module will be implemented once reports are collected.
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

BAHRAIN_DIR = config.DATA_DIR / "bahrain_bourse"


def list_available_reports() -> list[dict]:
    """List any manually placed Bahrain Bourse reports."""
    BAHRAIN_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for company_dir in sorted(BAHRAIN_DIR.iterdir()):
        if not company_dir.is_dir():
            continue
        for f in sorted(company_dir.iterdir()):
            if f.suffix in (".pdf", ".html", ".htm"):
                reports.append({
                    "company": company_dir.name,
                    "filename": f.name,
                    "path": str(f),
                    "format": f.suffix,
                })
    print(f"Found {len(reports)} Bahrain Bourse reports")
    for r in reports:
        print(f"  {r['company']}/{r['filename']}")
    return reports


def parse_pdf_report(filepath: Path) -> dict:
    """Parse a PDF annual report into sections. Requires pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        print("Install pdfplumber: pip install pdfplumber")
        return {}

    text_pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_pages.append(text)

    full_text = "\n\n".join(text_pages)
    return {
        "source_file": str(filepath),
        "full_text_length": len(full_text),
        "sections": {"full_document": {"text": full_text, "char_count": len(full_text)}},
    }


if __name__ == "__main__":
    list_available_reports()
