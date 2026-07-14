"""Evaluate the RAG pipeline against FinanceBench questions."""

import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


@dataclass
class EvalResult:
    question_id: str
    company: str
    ticker: str
    question: str
    gold_answer: str
    predicted_answer: str = ""
    retrieved_sections: list = field(default_factory=list)
    retrieval_hit: bool = False
    answer_correct: bool = False
    answer_faithful: bool = False
    latency_seconds: float = 0.0
    error: str = ""


def evaluate_retrieval(question: str, ticker: str, gold_evidence: list, k: int = 10) -> dict:
    """Evaluate retrieval quality for a single question."""
    from src.pipeline.vector_store import HybridStore
    from src.pipeline.embedder import embed_batch, get_client

    store = HybridStore()
    if not store.bm25:
        store.rebuild_bm25_from_collection()

    query_emb = None
    if config.gemini_available():
        query_emb = embed_batch([question], get_client())[0]

    results = store.search(question, query_embedding=query_emb, k=k, ticker=ticker)

    retrieved_texts = [r["text"] for r in results]
    retrieved_sections = [
        {
            "section": r["metadata"]["section"],
            "filing_type": r["metadata"]["filing_type"],
            "filing_date": r["metadata"]["filing_date"],
            "score": r.get("rerank_score", r.get("score", 0)),
        }
        for r in results
    ]

    # Check if any gold evidence overlaps with retrieved chunks
    hit = False
    if gold_evidence:
        for ev in gold_evidence:
            ev_text = ev.get("evidence_text", "") if isinstance(ev, dict) else str(ev)
            ev_words = set(ev_text.lower().split()[:20])
            for rt in retrieved_texts:
                rt_words = set(rt.lower().split())
                if len(ev_words & rt_words) > len(ev_words) * 0.3:
                    hit = True
                    break
            if hit:
                break

    return {
        "retrieved_texts": retrieved_texts,
        "retrieved_sections": retrieved_sections,
        "retrieval_hit": hit,
    }


async def evaluate_with_agent(question: str, ticker: str) -> tuple[str, float]:
    """Get agent answer and measure latency."""
    from src.agent.react_agent import create_agent
    from agents import Runner

    agent = create_agent()
    start = time.time()
    result = await Runner.run(agent, f"Company: {ticker}. Question: {question}")
    latency = time.time() - start
    return result.final_output, latency


def evaluate_answer_correctness(predicted: str, gold: str) -> bool:
    """Simple keyword overlap check for answer correctness."""
    if not predicted or not gold:
        return False

    gold_lower = gold.lower().strip()
    pred_lower = predicted.lower().strip()

    # Extract numbers from gold answer
    import re
    gold_numbers = re.findall(r'[\d,]+\.?\d*', gold_lower)
    if gold_numbers:
        for num in gold_numbers:
            clean_num = num.replace(",", "")
            if clean_num in pred_lower.replace(",", ""):
                return True

    # Keyword overlap
    gold_words = set(gold_lower.split())
    pred_words = set(pred_lower.split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for", "and", "or"}
    gold_content = gold_words - stop_words
    if not gold_content:
        return False
    overlap = len(gold_content & pred_words) / len(gold_content)
    return overlap > 0.4


def run_retrieval_eval(matched_path: Path = None) -> list[EvalResult]:
    """Evaluate retrieval only (no LLM needed)."""
    if matched_path is None:
        matched_path = config.DATA_DIR / "financebench" / "matched_questions.json"

    with open(matched_path) as f:
        questions = json.load(f)

    results = []
    print(f"Evaluating retrieval on {len(questions)} questions...\n")

    for i, q in enumerate(questions):
        ticker = q["ticker"]
        evidence = q.get("evidence", [])

        ret = evaluate_retrieval(q["question"], ticker, evidence)

        er = EvalResult(
            question_id=q["id"],
            company=q["company"],
            ticker=ticker,
            question=q["question"],
            gold_answer=q["answer"],
            retrieved_sections=ret["retrieved_sections"],
            retrieval_hit=ret["retrieval_hit"],
        )
        results.append(er)
        status = "HIT" if ret["retrieval_hit"] else "MISS"
        print(f"  [{i+1}/{len(questions)}] {ticker} - {status} - {q['question'][:60]}...")

    hits = sum(1 for r in results if r.retrieval_hit)
    print(f"\nRetrieval results: {hits}/{len(results)} hits ({100*hits/len(results):.1f}%)")

    by_ticker = {}
    for r in results:
        by_ticker.setdefault(r.ticker, []).append(r)
    for ticker, rs in sorted(by_ticker.items()):
        h = sum(1 for r in rs if r.retrieval_hit)
        print(f"  {ticker}: {h}/{len(rs)}")

    return results


def run_full_eval(matched_path: Path = None) -> list[EvalResult]:
    """Full evaluation with agent (requires Gemini API access)."""
    if not config.gemini_available():
        print("Gemini API not configured. Running retrieval-only evaluation.")
        return run_retrieval_eval(matched_path)

    if matched_path is None:
        matched_path = config.DATA_DIR / "financebench" / "matched_questions.json"

    with open(matched_path) as f:
        questions = json.load(f)

    results = []
    print(f"Full evaluation on {len(questions)} questions...\n")

    for i, q in enumerate(questions):
        ticker = q["ticker"]
        evidence = q.get("evidence", [])

        # Retrieval eval
        ret = evaluate_retrieval(q["question"], ticker, evidence)

        # Agent eval
        try:
            predicted, latency = asyncio.run(
                evaluate_with_agent(q["question"], ticker)
            )
            correct = evaluate_answer_correctness(predicted, q["answer"])
        except Exception as e:
            predicted = ""
            latency = 0
            correct = False
            print(f"  Agent error: {e}")

        er = EvalResult(
            question_id=q["id"],
            company=q["company"],
            ticker=ticker,
            question=q["question"],
            gold_answer=q["answer"],
            predicted_answer=predicted,
            retrieved_sections=ret["retrieved_sections"],
            retrieval_hit=ret["retrieval_hit"],
            answer_correct=correct,
            latency_seconds=latency,
        )
        results.append(er)
        status = "CORRECT" if correct else "WRONG"
        print(f"  [{i+1}/{len(questions)}] {ticker} - {status} ({latency:.1f}s) - {q['question'][:50]}...")

    # Summary
    hits = sum(1 for r in results if r.retrieval_hit)
    correct = sum(1 for r in results if r.answer_correct)
    avg_latency = sum(r.latency_seconds for r in results) / len(results) if results else 0

    print(f"\n{'='*50}")
    print(f"Retrieval hit rate: {hits}/{len(results)} ({100*hits/len(results):.1f}%)")
    print(f"Answer accuracy:    {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"Avg latency:        {avg_latency:.1f}s")

    return results


def save_results(results: list[EvalResult], output_path: Path = None):
    """Save evaluation results to JSON."""
    if output_path is None:
        output_path = config.DATA_DIR / "financebench" / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    results = run_retrieval_eval()
    save_results(results)
