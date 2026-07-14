"""Re-embed all chunks in the ChromaDB collection with real Gemini embeddings.

Existing chunks were originally inserted without explicit embeddings (dense
access was never available until now), so ChromaDB silently auto-generated
384-dim vectors using its own default embedder. This script fetches every
chunk, embeds it with Gemini (matching config.EMBEDDING_DIM), and upserts the
result back by ID so the stale 384-dim vectors are overwritten.

Resumable: progress is checkpointed to data/reembed_checkpoint.jsonl after
every chunk, so an interrupted run can pick back up without re-embedding
chunks that already succeeded. The upsert step streams checkpoint lines in
batches rather than materializing every embedding as a second full copy in
memory (holding 18k x 1536-dim vectors twice was enough to get the process
OOM-killed on a laptop).

Run with no arguments to do the full embed+upsert flow. Run with
`--upsert-only` to skip straight to upserting an already-complete checkpoint
(useful if embedding finished but the upsert step crashed).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.pipeline.vector_store import HybridStore
from src.pipeline.embedder import get_client, embed_batch

CHECKPOINT_PATH = config.DATA_DIR / "reembed_checkpoint.jsonl"
DOCS_META_PATH = config.DATA_DIR / "reembed_docs_metas.jsonl"
UPSERT_BATCH_SIZE = 1000


def dump_docs_and_metas(store: HybridStore):
    """Save documents/metadatas for every chunk to a local file, decoupled
    from the collection, so it survives the collection being deleted below."""
    if DOCS_META_PATH.exists():
        print(f"{DOCS_META_PATH.name} already exists, skipping re-dump.")
        return
    print("Saving documents/metadatas before recreating the collection...")
    result = store.collection.get(include=["documents", "metadatas"])
    with open(DOCS_META_PATH, "w") as f:
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            f.write(json.dumps({"id": id_, "document": doc, "metadata": meta}) + "\n")
    print(f"Saved {len(result['ids'])} documents/metadatas to {DOCS_META_PATH.name}.")


def load_docs_and_metas() -> dict[str, dict]:
    out = {}
    with open(DOCS_META_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = {"document": rec["document"], "metadata": rec["metadata"]}
    return out


def recreate_collection(store: HybridStore):
    """ChromaDB locks a collection's embedding dimension at creation time, so
    the existing 384-dim collection can't just be upserted into with 1536-dim
    vectors (confirmed: InvalidArgumentError on the first upsert attempt).
    Delete and recreate empty; the new dimension is set on first insert."""
    name = store.collection.name
    print(f"Deleting and recreating collection '{name}' to reset its embedding dimension...")
    store.chroma_client.delete_collection(name)
    store.collection = store.chroma_client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"},
    )
    print("Collection recreated (empty).")


def load_checkpoint_ids() -> set[str]:
    """IDs only, to decide what's left to embed (cheap; avoids holding vectors)."""
    ids = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.add(json.loads(line)["id"])
    return ids


def iter_checkpoint_batches(batch_size: int = UPSERT_BATCH_SIZE):
    """Yield (ids, embeddings) batches by streaming the checkpoint file, so at
    most `batch_size` embeddings are ever held in memory at once."""
    batch_ids, batch_embs = [], []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            batch_ids.append(rec["id"])
            batch_embs.append(rec["embedding"])
            if len(batch_ids) >= batch_size:
                yield batch_ids, batch_embs
                batch_ids, batch_embs = [], []
    if batch_ids:
        yield batch_ids, batch_embs


def append_checkpoint(chunk_id: str, embedding: list[float]):
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps({"id": chunk_id, "embedding": embedding}) + "\n")


def do_embed(store: HybridStore):
    print("Fetching all chunk ids/documents from ChromaDB...")
    result = store.collection.get(include=["documents"])
    ids, docs = result["ids"], result["documents"]
    print(f"Found {len(ids)} chunks.")

    done_ids = load_checkpoint_ids()
    print(f"{len(done_ids)} already embedded from a previous run (resuming).")

    client = get_client()
    remaining = [(i, d) for i, d in zip(ids, docs) if i not in done_ids]
    print(f"{len(remaining)} chunks left to embed.")

    failed = []
    for n, (chunk_id, text) in enumerate(remaining, 1):
        try:
            emb = embed_batch([text], client)[0]
            append_checkpoint(chunk_id, emb)
        except Exception as e:
            print(f"  FAILED {chunk_id}: {e}")
            failed.append(chunk_id)

        if n % 100 == 0:
            done_so_far = len(done_ids) + n - len(failed)
            print(f"  {n}/{len(remaining)} embedded this run ({done_so_far}/{len(ids)} total, {len(failed)} failed)")

    print(f"\nEmbedding pass done. {len(failed)} failed this run.")
    if failed:
        print("Failed IDs:", failed[:20], "..." if len(failed) > 20 else "")
    return len(ids), failed


def do_upsert(store: HybridStore, docs_metas: dict[str, dict]):
    """Stream the checkpoint file and upsert in small batches, joining against
    the pre-saved documents/metadatas dump by ID."""
    print("Upserting embeddings into the fresh collection (streamed in batches)...")
    upserted = 0
    for batch_ids, batch_embs in iter_checkpoint_batches():
        docs = [docs_metas[i]["document"] for i in batch_ids]
        metas = [docs_metas[i]["metadata"] for i in batch_ids]

        store.collection.upsert(ids=batch_ids, embeddings=batch_embs, documents=docs, metadatas=metas)
        upserted += len(batch_ids)
        print(f"  Upserted {upserted} chunks so far...")

    print(f"Upserted {upserted} chunks total with {config.EMBEDDING_DIM}-dim Gemini embeddings.")
    return upserted


def main():
    if not config.gemini_available():
        print("GEMINI_API_KEY not set. Aborting.")
        return

    store = HybridStore()
    upsert_only = "--upsert-only" in sys.argv

    if not upsert_only:
        total_chunks, failed = do_embed(store)
    else:
        failed = []
        total_chunks = store.collection.count()

    # Save documents/metadatas BEFORE deleting the collection, then recreate
    # it so the new 1536-dim embeddings aren't rejected by the old 384-dim
    # HNSW index (dimension is locked in ChromaDB once a collection has data).
    dump_docs_and_metas(store)
    recreate_collection(store)
    docs_metas = load_docs_and_metas()

    upserted = do_upsert(store, docs_metas)

    if not failed and upserted == total_chunks:
        CHECKPOINT_PATH.unlink(missing_ok=True)
        DOCS_META_PATH.unlink(missing_ok=True)
        print("Checkpoint and docs/metas files removed (all chunks embedded and upserted successfully).")
    else:
        print(f"Checkpoint file kept at {CHECKPOINT_PATH} ({len(failed)} failed embeds; "
              f"{upserted}/{total_chunks} upserted).")

    print("Rebuilding BM25 index from the recreated collection...")
    store.rebuild_bm25_from_collection()
    print("Done.")


if __name__ == "__main__":
    main()
