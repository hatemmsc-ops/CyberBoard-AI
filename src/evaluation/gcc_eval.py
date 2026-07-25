"""Retrieval evaluation for the GCC (Bahrain Bourse) question set.

Mirrors financebench_eval.run_retrieval_eval but targets the bahrain_bourse
collection and the gcc_eval/questions.json schema (evidence_text is a single
string per question rather than an evidence list). Retrieval-only: needs query
embeddings and hybrid search, no agent LLM calls.
"""

import json
from pathlib import Path
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.evaluation.financebench_eval import EvalResult


def _retrieve(store, question: str, ticker: str, evidence_text: str, k: int = 10) -> dict:
    from src.pipeline.embedder import embed_batch, get_client

    query_emb = None
    if config.gemini_available():
        query_emb = embed_batch([question], get_client())[0]

    results = store.search(question, query_embedding=query_emb, k=k, ticker=ticker)
    retrieved_texts = [r["text"] for r in results]
    retrieved_sections = [
        {
            "section": r["metadata"].get("section"),
            "filing_type": r["metadata"].get("filing_type"),
            "filing_date": r["metadata"].get("filing_date"),
            "score": r.get("rerank_score", r.get("score", 0)),
        }
        for r in results
    ]

    # Hit = >30% of the first 20 evidence words appear in a retrieved chunk
    # (same rule as the FinanceBench harness).
    hit = False
    ev_words = set(evidence_text.lower().split()[:20])
    if ev_words:
        for rt in retrieved_texts:
            rt_words = set(rt.lower().split())
            if len(ev_words & rt_words) > len(ev_words) * 0.3:
                hit = True
                break
    return {"retrieved_texts": retrieved_texts,
            "retrieved_sections": retrieved_sections,
            "retrieval_hit": hit}


def run_gcc_retrieval_eval(k: int = 10) -> list[EvalResult]:
    from src.pipeline.vector_store import HybridStore

    path = config.DATA_DIR / "gcc_eval" / "questions.json"
    with open(path) as f:
        questions = json.load(f)["questions"]

    store = HybridStore(collection_name=config.BAHRAIN_BOURSE_COLLECTION_NAME)
    if not store.bm25:
        store.rebuild_bm25_from_collection()

    results = []
    print(f"Evaluating GCC retrieval on {len(questions)} questions (k={k})...\n")
    for i, q in enumerate(questions):
        ret = _retrieve(store, q["question"], q["ticker"], q.get("evidence_text", ""), k=k)
        er = EvalResult(
            question_id=q["id"],
            company=q["company"],
            ticker=q["ticker"],
            question=q["question"],
            gold_answer=q["answer"],
            retrieved_sections=ret["retrieved_sections"],
            retrieval_hit=ret["retrieval_hit"],
        )
        results.append(er)
        status = "HIT " if ret["retrieval_hit"] else "MISS"
        print(f"  [{i+1:2}/{len(questions)}] {q['ticker']:5} {status} {q['question'][:58]}")

    hits = sum(1 for r in results if r.retrieval_hit)
    print(f"\nRetrieval hit@{k}: {hits}/{len(results)} ({100*hits/len(results):.1f}%)")
    by = {}
    for r in results:
        by.setdefault(r.ticker, []).append(r)
    for t, rs in sorted(by.items()):
        h = sum(1 for r in rs if r.retrieval_hit)
        print(f"  {t:5}: {h}/{len(rs)}")
    return results


if __name__ == "__main__":
    res = run_gcc_retrieval_eval()
    out = config.DATA_DIR / "gcc_eval" / "retrieval_results.json"
    with open(out, "w") as f:
        json.dump([asdict(r) for r in res], f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")
