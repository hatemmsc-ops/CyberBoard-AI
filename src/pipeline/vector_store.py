"""ChromaDB + BM25 hybrid retrieval with cross-encoder reranking."""

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def confidence_from_result(result: dict) -> float:
    """Map a retrieval result's score to a 0-1 confidence value for display.

    Rerank scores are cross-encoder logits (unbounded), so a sigmoid maps them
    to a pseudo-probability. Hybrid scores are already normalized to [0, 1].
    """
    if "rerank_score" in result:
        return float(1 / (1 + np.exp(-result["rerank_score"])))
    return float(result.get("score", 0.0))


class HybridStore:
    """Combines dense (ChromaDB) and sparse (BM25) retrieval with reranking."""

    def __init__(self, collection_name: str = config.COLLECTION_NAME, persist_dir: Path = None):
        if persist_dir is None:
            persist_dir = config.CHROMA_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_ids = []
        self.reranker = None

    def add_chunks(self, chunks: list[dict]):
        """Insert embedded chunks into ChromaDB and build BM25 index."""
        ids, embeddings, documents, metadatas = [], [], [], []

        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            chunk_id = f"{meta['ticker']}_{meta['filing_type']}_{meta['filing_date']}_{meta['section']}_{meta['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(meta)
            if "embedding" in chunk:
                embeddings.append(chunk["embedding"])

        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            batch_kwargs = {
                "ids": ids[i:i + batch_size],
                "documents": documents[i:i + batch_size],
                "metadatas": metadatas[i:i + batch_size],
            }
            if embeddings:
                batch_kwargs["embeddings"] = embeddings[i:i + batch_size]
            self.collection.upsert(**batch_kwargs)

        print(f"Stored {len(ids)} chunks in ChromaDB")
        self._build_bm25(ids, documents)

    def _build_bm25(self, ids: list[str], documents: list[str]):
        self.bm25_ids = ids
        self.bm25_corpus = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"Built BM25 index over {len(documents)} documents")

    def rebuild_bm25_from_collection(self):
        """Rebuild BM25 index from existing ChromaDB collection."""
        result = self.collection.get(include=["documents"])
        if result["ids"]:
            self._build_bm25(result["ids"], result["documents"])

    def search(self, query: str, query_embedding: list[float] = None, k: int = 10,
               alpha: float = config.HYBRID_ALPHA, rerank: bool = True, ticker: str = None,
               trace: list = None) -> list[dict]:
        """Hybrid search: dense + sparse, optionally reranked.

        Pass a list as `trace` to capture per-stage retrieval info for
        explainability (each stage appended as a dict).
        """
        where_filter = {"ticker": ticker} if ticker else None

        dense_scores = {}
        if query_embedding:
            chroma_kwargs = {"query_embeddings": [query_embedding], "n_results": k * 2}
            if where_filter:
                chroma_kwargs["where"] = where_filter
            results = self.collection.query(**chroma_kwargs)
            for doc_id, dist in zip(results["ids"][0], results["distances"][0]):
                dense_scores[doc_id] = 1.0 - dist  # cosine distance -> similarity

        sparse_scores = {}
        if self.bm25:
            bm25_scores = self.bm25.get_scores(query.lower().split())
            if ticker:
                candidate_idxs = [i for i, doc_id in enumerate(self.bm25_ids) if doc_id.startswith(ticker + "_")]
                ranked_idxs = sorted(candidate_idxs, key=lambda i: bm25_scores[i], reverse=True)[:k * 2]
            else:
                ranked_idxs = np.argsort(bm25_scores)[::-1][:k * 2]
            for idx in ranked_idxs:
                sparse_scores[self.bm25_ids[idx]] = float(bm25_scores[idx])

        if trace is not None:
            trace.append({
                "stage": "1. Candidate scoring",
                "dense_candidates": len(dense_scores),
                "sparse_candidates": len(sparse_scores),
                "dense_available": bool(dense_scores),
                "ticker_filter": ticker or "none",
                "top_sparse": [
                    {"id": doc_id, "bm25": round(score, 2)}
                    for doc_id, score in sorted(sparse_scores.items(), key=lambda x: -x[1])[:5]
                ],
            })

        # normalize scores to [0, 1]
        def normalize(scores: dict) -> dict:
            if not scores:
                return scores
            vals = list(scores.values())
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return {k: 1.0 for k in scores}
            return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        all_ids = set(dense_norm) | set(sparse_norm)
        combined = {}
        for doc_id in all_ids:
            d = dense_norm.get(doc_id, 0.0)
            s = sparse_norm.get(doc_id, 0.0)
            combined[doc_id] = alpha * d + (1 - alpha) * s

        top_ids = sorted(combined, key=combined.get, reverse=True)[:k * 2 if rerank else k]

        if trace is not None:
            trace.append({
                "stage": "2. Hybrid fusion",
                "alpha": alpha,
                "formula": f"{alpha} x dense + {round(1 - alpha, 2)} x sparse (scores normalized to 0..1)",
                "pool_size": len(all_ids),
                "kept_for_rerank": len(top_ids),
                "top_fused": [
                    {"id": doc_id, "fused_score": round(combined[doc_id], 3)}
                    for doc_id in top_ids[:5]
                ],
            })

        # fetch documents for top results
        if not top_ids:
            return []
        fetched = self.collection.get(ids=top_ids, include=["documents", "metadatas"])
        id_to_doc = dict(zip(fetched["ids"], fetched["documents"]))
        id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))

        candidates = [
            {"id": doc_id, "text": id_to_doc[doc_id], "metadata": id_to_meta[doc_id], "score": combined[doc_id]}
            for doc_id in top_ids if doc_id in id_to_doc
        ]

        if rerank and candidates:
            pre_order = [c["id"] for c in candidates[:5]]
            candidates = self._rerank(query, candidates, k)
            if trace is not None:
                trace.append({
                    "stage": "3. Cross encoder reranking",
                    "model": config.RERANKER_MODEL,
                    "order_before": pre_order,
                    "order_after": [c["id"] for c in candidates[:5]],
                    "rerank_scores": [
                        {"id": c["id"], "logit": round(c["rerank_score"], 2),
                         "confidence": round(confidence_from_result(c), 3)}
                        for c in candidates[:5]
                    ],
                })
        elif trace is not None:
            trace.append({"stage": "3. Cross encoder reranking", "skipped": True})

        return candidates[:k]

    def _rerank(self, query: str, candidates: list[dict], k: int) -> list[dict]:
        if self.reranker is None:
            self.reranker = CrossEncoder(config.RERANKER_MODEL)
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:k]
