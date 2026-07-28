"""Baseline comparison: agent vs naive RAG vs no retrieval.

Runs the same question sets through two baselines and scores them with the same
LLM judge used for the agent, so the three approaches can be compared head to head:

  - no_retrieval: the chat model answers from parametric knowledge only. No
    document is retrieved, so the answer is ungrounded by construction and its
    faithfulness (grounding in retrieved evidence) is not applicable.
  - naive_rag: a single top-k retrieval is stuffed into the prompt and the model
    answers in one shot. No agent loop, no tool selection, no multi-step reasoning.
  - agent: the full ReAct agent (scored separately by full_eval).

The point of the comparison is to show what the agent loop adds over vanilla RAG
and what retrieval adds over the bare model.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.evaluation.llm_judge import judge_answer
from src.evaluation.full_eval import _load_questions, JudgedResult


def _chat(prompt: str, max_retries: int = 4) -> str:
    from google.genai import types
    from src.pipeline.embedder import get_client
    client = get_client()
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return resp.text or ""
        except Exception as e:
            m = str(e).lower()
            transient = any(s in m for s in ("503", "429", "unavailable", "rate",
                                             "resource_exhausted", "overloaded",
                                             "connection reset", "read error", "high demand"))
            if transient and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def answer_no_retrieval(question: str, ticker: str) -> tuple[str, str]:
    prompt = (
        f"You are a financial analyst. Answer the following question about {ticker} "
        f"with the specific figure or fact requested, as precisely as you can.\n\n"
        f"Question: {question}"
    )
    return _chat(prompt), ""


def answer_naive_rag(question: str, ticker: str, k: int = 5) -> tuple[str, str]:
    from src.pipeline.vector_store import get_cached_store
    from src.pipeline.embedder import embed_batch, get_client

    store = get_cached_store(config.collection_for_ticker(ticker))
    emb = None
    if config.gemini_available():
        emb = embed_batch([question], get_client())[0]
    results = store.search(question, query_embedding=emb, k=k, ticker=ticker.upper())
    ctx = "\n\n---\n\n".join(
        f"[{r['metadata'].get('filing_type')} {r['metadata'].get('filing_date')}]\n{r['text']}"
        for r in results
    )
    prompt = (
        "Answer the question using ONLY the filing excerpts below. State the specific "
        "figure and cite which excerpt it comes from. If the answer is not present in "
        "the excerpts, say that it is not available.\n\n"
        f"Excerpts:\n{ctx}\n\nQuestion: {question}"
    )
    return _chat(prompt), ctx


ANSWER_FNS = {"no_retrieval": answer_no_retrieval, "naive_rag": answer_naive_rag}


def run_baseline(set_name: str, mode: str) -> list[JudgedResult]:
    _paths = {
        "gcc": config.DATA_DIR / "gcc_eval" / "questions.json",
        "sec_recent": config.DATA_DIR / "sec_recent" / "questions.json",
    }
    questions = _load_questions(_paths[set_name])
    answer_fn = ANSWER_FNS[mode]

    results = []
    print(f"\nBaseline '{mode}' on {set_name} ({len(questions)} questions)...")
    for i, q in enumerate(questions):
        try:
            start = time.time()
            pred, ctx = answer_fn(q["question"], q["ticker"])
            latency = time.time() - start
            j = judge_answer(q["question"], q["answer"], pred, ctx)
            r = JudgedResult(
                question_id=q["id"], company=q["company"], ticker=q["ticker"],
                question=q["question"], gold_answer=q["answer"], predicted_answer=pred,
                correct=j["correct"], faithful=j["faithful"], hallucinated=j["hallucinated"],
                rationale=j["rationale"], latency_seconds=latency,
            )
        except Exception as e:
            r = JudgedResult(question_id=q["id"], company=q["company"], ticker=q["ticker"],
                             question=q["question"], gold_answer=q["answer"],
                             error=f"{type(e).__name__}: {str(e)[:120]}")
        results.append(r)
        tag = "OK" if r.correct else "XX"
        print(f"  [{i+1:2}/{len(questions)}] {q['ticker']:5} {tag} {q['question'][:48]}")
        time.sleep(1.0)
    return results


def _acc_faith(results):
    scored = [r for r in results if not r.error]
    d = len(scored) or 1
    return (sum(r.correct for r in scored), sum(r.faithful for r in scored), len(scored),
            100 * sum(r.correct for r in scored) / d, 100 * sum(r.faithful for r in scored) / d)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["gcc", "sec_recent"], required=True)
    args = ap.parse_args()

    out_dir = config.DATA_DIR / ("gcc_eval" if args.set == "gcc" else "sec_recent")
    rows = {}
    for mode in ("no_retrieval", "naive_rag"):
        res = run_baseline(args.set, mode)
        json.dump([asdict(r) for r in res], open(out_dir / f"baseline_{mode}_results.json", "w"),
                  indent=2, ensure_ascii=False)
        rows[mode] = _acc_faith(res)

    # pull agent numbers from the existing full_eval results for the same set
    agent_path = out_dir / "full_eval_results.json"
    if agent_path.exists():
        rows["agent"] = _acc_faith([JudgedResult(**{k: v for k, v in r.items()})
                                    for r in json.load(open(agent_path))])

    print(f"\n{'='*64}\nBASELINE COMPARISON ({args.set})")
    print(f"{'Config':<14} {'Accuracy':>18} {'Faithfulness':>18}")
    print("-" * 64)
    for mode in ("no_retrieval", "naive_rag", "agent"):
        if mode in rows:
            c, f, n, ca, fa = rows[mode]
            print(f"{mode:<14} {f'{c}/{n} ({ca:.1f}%)':>18} {f'{f}/{n} ({fa:.1f}%)':>18}")
