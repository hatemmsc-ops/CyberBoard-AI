"""Bahrain Bourse annual report ingestion (GCC cross-market evaluation corpus).

Bahrain Bourse does not have a public API for bulk report downloads. Reports are
manually collected and placed in data/bahrain_bourse/{company_name}/. This module
parses, chunks, embeds, and indexes them into a dedicated ChromaDB collection
(config.BAHRAIN_BOURSE_COLLECTION_NAME) so the SEC "sec_filings" corpus and its
reported chunk count stay reproducible and untouched.

Usage: python -m src.ingestion.bahrain_bourse_loader          # list reports
       python -m src.ingestion.bahrain_bourse_loader --ingest # parse+embed+index
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

BAHRAIN_DIR = config.BAHRAIN_BOURSE_DIR


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


def build_chunks_for_company(company_dir: Path) -> list[dict]:
    """Parse and chunk every PDF for one company into metadata-tagged chunks."""
    from src.pipeline.chunker import split_text, token_len

    ticker, year = config.BAHRAIN_BOURSE_COMPANIES.get(
        company_dir.name, (company_dir.name.upper(), "UNK"))

    chunks = []
    for pdf in sorted(company_dir.glob("*.pdf")):
        parsed = parse_pdf_report(pdf)
        full_text = parsed.get("sections", {}).get("full_document", {}).get("text", "")
        if not full_text.strip():
            print(f"  WARNING: no extractable text in {pdf.name}")
            continue
        pieces = split_text(full_text)
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "metadata": {
                    "ticker": ticker,
                    "company": company_dir.name,
                    "filing_type": "Annual Report",
                    "filing_date": f"{year}-12-31",
                    "section": "full_document",
                    "chunk_index": i,
                    "token_count": token_len(piece),
                    "source_file": str(pdf),
                    "market": "Bahrain Bourse",
                },
            })
    return chunks


def ingest():
    """Parse, chunk, embed, and index all Bahrain Bourse reports."""
    from src.pipeline.vector_store import HybridStore
    from src.pipeline.embedder import get_client, embed_batch

    if not config.gemini_available():
        print("GEMINI_API_KEY not set. Aborting.")
        return

    BAHRAIN_DIR.mkdir(parents=True, exist_ok=True)
    company_dirs = [d for d in sorted(BAHRAIN_DIR.iterdir()) if d.is_dir()]
    if not company_dirs:
        print(f"No company folders in {BAHRAIN_DIR}")
        return

    all_chunks = []
    for company_dir in company_dirs:
        chunks = build_chunks_for_company(company_dir)
        ticker = chunks[0]["metadata"]["ticker"] if chunks else "?"
        print(f"  {company_dir.name} [{ticker}]: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")
    if not all_chunks:
        return

    client = get_client()
    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks via Gemini ({config.EMBEDDING_DIM}d)...")
    embeddings = embed_batch(texts, client)
    for c, emb in zip(all_chunks, embeddings):
        c["embedding"] = emb

    store = HybridStore(collection_name=config.BAHRAIN_BOURSE_COLLECTION_NAME)
    store.add_chunks(all_chunks)
    print(f"\nStored in collection '{config.BAHRAIN_BOURSE_COLLECTION_NAME}': "
          f"{store.collection.count()} chunks")


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        ingest()
    else:
        list_available_reports()
