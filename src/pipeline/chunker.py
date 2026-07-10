"""Split parsed filing sections into token-counted chunks with metadata."""

import json
from pathlib import Path

import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

ENC = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(ENC.encode(text))


def split_text(text: str, max_tokens: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Recursive split on paragraph -> sentence -> word boundaries."""
    if token_len(text) <= max_tokens:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) < 2:
            continue

        chunks = []
        current = parts[0]
        for part in parts[1:]:
            candidate = current + sep + part
            if token_len(candidate) > max_tokens:
                if current.strip():
                    chunks.append(current.strip())
                # overlap: keep tail tokens of previous chunk
                if overlap > 0 and current.strip():
                    tail_tokens = ENC.encode(current)[-overlap:]
                    current = ENC.decode(tail_tokens) + sep + part
                else:
                    current = part
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())

        if len(chunks) > 1:
            return chunks
    # last resort: hard split on tokens
    tokens = ENC.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(ENC.decode(chunk_tokens))
    return chunks


def chunk_filing(parsed: dict) -> list[dict]:
    """Chunk a single parsed filing JSON into metadata-tagged chunks."""
    ticker = parsed.get("ticker", "UNK")
    filing_type = parsed.get("filing_type", "UNK")
    filing_date = parsed.get("filing_date", "UNK")
    source_file = parsed.get("source_file", "")

    all_chunks = []
    for section_name, section_data in parsed.get("sections", {}).items():
        text = section_data["text"] if isinstance(section_data, dict) else section_data
        pieces = split_text(text)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "metadata": {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                    "section": section_name,
                    "chunk_index": i,
                    "token_count": token_len(piece),
                    "source_file": source_file,
                },
            })
    return all_chunks


def chunk_all_filings(filings_dir: Path = None) -> list[dict]:
    """Chunk all parsed JSON filings in the data directory."""
    if filings_dir is None:
        filings_dir = config.SEC_FILINGS_DIR

    all_chunks = []
    for json_file in sorted(filings_dir.rglob("*.json")):
        with open(json_file) as f:
            parsed = json.load(f)
        chunks = chunk_filing(parsed)
        all_chunks.extend(chunks)
        print(f"  {json_file.parent.name}/{json_file.name}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunk_all_filings()
