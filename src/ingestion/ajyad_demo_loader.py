"""Ingest Ajyad Capital annual reports into a standalone demo collection.

This is a showcase corpus for live demonstration purposes only — it is NOT
part of the thesis's formal evaluation methodology, which is scoped to SEC
EDGAR filings and (once collected) Bahrain Bourse-listed company filings for
the GCC evaluation set. Keeping this in its own ChromaDB collection means the
thesis's reported "18,136 chunks" figure and evaluation results stay
untouched and reproducible regardless of this corpus.

Usage: python -m src.ingestion.ajyad_demo_loader
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.ingestion.bahrain_bourse_loader import parse_pdf_report
from src.pipeline.chunker import split_text, token_len
from src.pipeline.vector_store import HybridStore
from src.pipeline.embedder import get_client, embed_batch


def build_chunks_for_report(filepath: Path) -> list[dict]:
    """Parse and chunk a single Ajyad Capital annual report PDF."""
    year_match = re.search(r"(\d{4})", filepath.stem)
    year = year_match.group(1) if year_match else "UNK"

    parsed = parse_pdf_report(filepath)
    full_text = parsed.get("sections", {}).get("full_document", {}).get("text", "")
    if not full_text.strip():
        print(f"  WARNING: no extractable text in {filepath.name}")
        return []

    pieces = split_text(full_text)
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append({
            "text": piece,
            "metadata": {
                "ticker": "AJYAD",
                "filing_type": "Annual Report",
                "filing_date": f"{year}-12-31",
                "section": "full_document",
                "chunk_index": i,
                "token_count": token_len(piece),
                "source_file": str(filepath),
            },
        })
    return chunks


def main():
    if not config.gemini_available():
        print("GEMINI_API_KEY not set. Aborting.")
        return

    config.AJYAD_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(config.AJYAD_DEMO_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.AJYAD_DEMO_DIR}")
        return
    print(f"Found {len(pdfs)} reports.")

    all_chunks = []
    for pdf in pdfs:
        print(f"Parsing {pdf.name}...")
        chunks = build_chunks_for_report(pdf)
        print(f"  {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    client = get_client()
    texts = [c["text"] for c in all_chunks]
    print("Embedding...")
    embeddings = embed_batch(texts, client)
    for c, emb in zip(all_chunks, embeddings):
        c["embedding"] = emb
    print(f"Embedded {len(embeddings)} chunks ({config.EMBEDDING_DIM}d)")

    store = HybridStore(collection_name=config.AJYAD_DEMO_COLLECTION_NAME)
    store.add_chunks(all_chunks)
    print(f"\nStored in ChromaDB collection '{config.AJYAD_DEMO_COLLECTION_NAME}': {store.collection.count()} chunks")


if __name__ == "__main__":
    main()
