"""Parse SEC filing HTML into structured sections with clean text."""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


SECTION_PATTERNS = {
    "business_overview": [
        r"item\s*1[.\s]",
        r"business\b",
    ],
    "risk_factors": [
        r"item\s*1a[.\s]",
        r"risk\s*factors",
    ],
    "md_and_a": [
        r"item\s*7[.\s]",
        r"management.s\s*discussion",
        r"md\s*&\s*a",
    ],
    "financial_statements": [
        r"item\s*8[.\s]",
        r"financial\s*statements\s*and\s*supplementary",
    ],
    "controls_and_procedures": [
        r"item\s*9a[.\s]",
        r"controls\s*and\s*procedures",
    ],
}


def clean_text(text: str) -> str:
    """Remove excess whitespace and normalize text."""
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_html(html_content: str) -> str:
    """Extract plain text from HTML filing."""
    soup = BeautifulSoup(html_content, "lxml")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    return clean_text(soup.get_text(separator="\n"))


def identify_sections(full_text: str) -> dict[str, str]:
    """Split full text into named sections using regex patterns."""
    lines = full_text.split("\n")
    section_starts = []

    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        if len(line_lower) > 200 or len(line_lower) < 3:
            continue
        for section_name, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    section_starts.append((i, section_name, line.strip()))
                    break

    # Deduplicate: keep first occurrence of each section
    seen = set()
    unique_starts = []
    for idx, name, heading in section_starts:
        if name not in seen:
            seen.add(name)
            unique_starts.append((idx, name, heading))

    unique_starts.sort(key=lambda x: x[0])

    sections = {}
    for j, (start_idx, name, heading) in enumerate(unique_starts):
        end_idx = unique_starts[j + 1][0] if j + 1 < len(unique_starts) else len(lines)
        section_text = "\n".join(lines[start_idx:end_idx])
        section_text = clean_text(section_text)
        if len(section_text) > 100:
            sections[name] = section_text

    # If no sections found, put everything under "full_document"
    if not sections:
        sections["full_document"] = full_text

    return sections


def parse_filing(filepath: Path) -> dict:
    """Parse a single filing into structured sections."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    full_text = extract_text_from_html(content)
    sections = identify_sections(full_text)

    return {
        "source_file": str(filepath),
        "full_text_length": len(full_text),
        "sections": {name: {"text": text, "char_count": len(text)} for name, text in sections.items()},
    }


def parse_all_filings(filings_dir: Path = None) -> list[dict]:
    """Parse all downloaded filings."""
    if filings_dir is None:
        filings_dir = config.SEC_FILINGS_DIR

    results = []
    for ticker_dir in sorted(filings_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for filing_file in sorted(ticker_dir.glob("*.*")):
            if filing_file.suffix not in (".html", ".htm"):
                continue
            print(f"  Parsing {ticker}/{filing_file.name}...")
            parsed = parse_filing(filing_file)
            parsed["ticker"] = ticker
            parsed["filename"] = filing_file.name

            # Extract filing type and date from filename (e.g., "10-K_2024-10-31.htm")
            parts = filing_file.stem.split("_")
            if len(parts) >= 2:
                parsed["filing_type"] = parts[0]
                parsed["filing_date"] = parts[1]

            results.append(parsed)

            # Save JSON alongside HTML
            json_path = filing_file.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(parsed, f, indent=2)

    print(f"\nParsed {len(results)} filings")
    return results


if __name__ == "__main__":
    parse_all_filings()
