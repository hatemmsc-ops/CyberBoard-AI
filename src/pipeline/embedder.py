"""Batch embed chunks using Azure OpenAI text-embedding-3-small."""

import time
from openai import AzureOpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def get_client() -> AzureOpenAI:
    if not config.azure_available():
        raise RuntimeError("Azure OpenAI credentials not set. Add them to .env")
    return AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )


def embed_batch(texts: list[str], client: AzureOpenAI = None, batch_size: int = 64) -> list[list[float]]:
    """Embed texts in batches with rate limit handling."""
    if client is None:
        client = get_client()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 0
        while retries < 5:
            try:
                resp = client.embeddings.create(
                    input=batch,
                    model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                )
                all_embeddings.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2 ** retries
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    retries += 1
                else:
                    raise
        if i % (batch_size * 10) == 0 and i > 0:
            print(f"  Embedded {i + len(batch)}/{len(texts)} chunks")

    return all_embeddings


def embed_chunks(chunks: list[dict], client: AzureOpenAI = None) -> list[dict]:
    """Add embedding vectors to chunk dicts."""
    if not config.azure_available():
        print("WARNING: Azure OpenAI not configured. Skipping embedding.")
        return chunks

    if client is None:
        client = get_client()

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_batch(texts, client)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    print(f"Embedded {len(embeddings)} chunks ({config.EMBEDDING_DIM}d)")
    return chunks
