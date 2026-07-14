"""Batch embed chunks using Gemini's embedding model."""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from google import genai
from google.genai import types

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

REQUEST_TIMEOUT_SECONDS = 30
# google-genai's own http_options timeout is unreliable (known SDK issue: the
# value gets silently dropped and requests can hang indefinitely instead of
# raising), so timeouts are enforced with a thread pool here instead.
_executor = ThreadPoolExecutor(max_workers=4)


def _is_transient(error: Exception) -> bool:
    msg = str(error).lower()
    return any(s in msg for s in ("429", "503", "rate", "resource_exhausted", "unavailable", "overloaded"))


def get_client() -> genai.Client:
    if not config.gemini_available():
        raise RuntimeError("Gemini API key not set. Add GEMINI_API_KEY to .env")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _embed_one(client: genai.Client, text: str, embed_config: "types.EmbedContentConfig") -> list[float]:
    future = _executor.submit(
        client.models.embed_content,
        model=config.GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=embed_config,
    )
    resp = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
    return resp.embeddings[0].values


def embed_batch(texts: list[str], client: genai.Client = None, batch_size: int = 64) -> list[list[float]]:
    """Embed texts with rate limit and timeout handling.

    gemini-embedding-001 accepts one input per request, so each text is sent
    individually rather than as a batch; batch_size only controls the progress
    print interval here.
    """
    if client is None:
        client = get_client()

    embed_config = types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIM)
    all_embeddings = []
    for i, text in enumerate(texts):
        retries = 0
        while retries < 5:
            try:
                all_embeddings.append(_embed_one(client, text, embed_config))
                break
            except FutureTimeoutError:
                wait = 2 ** retries
                print(f"  Request timed out after {REQUEST_TIMEOUT_SECONDS}s, retrying in {wait}s...")
                time.sleep(wait)
                retries += 1
            except Exception as e:
                if _is_transient(e) and retries < 4:
                    wait = 2 ** retries
                    print(f"  Rate limited or unavailable, waiting {wait}s...")
                    time.sleep(wait)
                    retries += 1
                else:
                    raise
        else:
            raise RuntimeError(f"Failed to embed text after {retries} retries: {text[:80]!r}")
        if (i + 1) % batch_size == 0:
            print(f"  Embedded {i + 1}/{len(texts)} chunks")

    return all_embeddings


def embed_chunks(chunks: list[dict], client: genai.Client = None) -> list[dict]:
    """Add embedding vectors to chunk dicts."""
    if not config.gemini_available():
        print("WARNING: Gemini API not configured. Skipping embedding.")
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
