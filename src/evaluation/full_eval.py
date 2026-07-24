"""Full agent evaluation with LLM-judge scoring.

Runs each question through the ReAct agent, captures the tool-output context the
agent actually conditioned on, and scores the answer with the LLM judge for
correctness and faithfulness (grounding). Reports answer accuracy, faithfulness
rate, hallucination rate and latency, overall and per ticker.

Works on either question set:
  - FinanceBench: data/financebench/matched_questions.json  (schema: flat list)
  - GCC:          data/gcc_eval/questions.json               (schema: {"questions": [...]})
"""

import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.evaluation.llm_judge import judge_answer


@dataclass
class JudgedResult:
    question_id: str
    company: str
    ticker: str
    question: str
    gold_answer: str
    predicted_answer: str = ""
    correct: bool = False
    faithful: bool = False
    hallucinated: bool = False
    rationale: str = ""
    latency_seconds: float = 0.0
    error: str = ""


def _load_questions(path: Path) -> list[dict]:
    data = json.load(open(path))
    return data["questions"] if isinstance(data, dict) else data


async def _run_agent(agent, ticker: str, question: str) -> tuple[str, str, float]:
    """Return (final_answer, tool_context, latency)."""
    from agents import Runner
    from agents.items import ToolCallOutputItem

    start = time.time()
    result = await Runner.run(agent, f"Company: {ticker}. Question: {question}")
    latency = time.time() - start
    ctx = "\n\n".join(
        str(it.output) for it in result.new_items if isinstance(it, ToolCallOutputItem)
    )
    return result.final_output, ctx, latency


def run_full_eval(question_path: Path = None, set_name: str = "gcc") -> list[JudgedResult]:
    if not config.gemini_available():
        raise RuntimeError("Gemini API not configured; the agent and judge both need it.")

    if question_path is None:
        question_path = (config.DATA_DIR / "gcc_eval" / "questions.json" if set_name == "gcc"
                         else config.DATA_DIR / "financebench" / "matched_questions.json")

    from src.agent.react_agent import create_agent
    agent = create_agent()
    questions = _load_questions(question_path)

    results = []
    print(f"Full agent eval ({set_name}) on {len(questions)} questions...\n")
    for i, q in enumerate(questions):
        ticker = q["ticker"]
        try:
            predicted, ctx, latency = asyncio.run(_run_agent(agent, ticker, q["question"]))
            j = judge_answer(q["question"], q["answer"], predicted, ctx)
            r = JudgedResult(
                question_id=q["id"], company=q["company"], ticker=ticker,
                question=q["question"], gold_answer=q["answer"],
                predicted_answer=predicted, correct=j["correct"],
                faithful=j["faithful"], hallucinated=j["hallucinated"],
                rationale=j["rationale"], latency_seconds=latency,
            )
        except Exception as e:
            r = JudgedResult(
                question_id=q["id"], company=q["company"], ticker=ticker,
                question=q["question"], gold_answer=q["answer"],
                error=f"{type(e).__name__}: {str(e)[:160]}",
            )
        results.append(r)
        tag = "OK " if r.correct else "XX "
        f = "F" if r.faithful else "h"
        print(f"  [{i+1:2}/{len(questions)}] {ticker:5} {tag}{f} ({r.latency_seconds:4.1f}s) {q['question'][:52]}")

    _summarize(results)
    return results


def _summarize(results: list[JudgedResult]):
    n = len(results)
    scored = [r for r in results if not r.error]
    correct = sum(r.correct for r in scored)
    faithful = sum(r.faithful for r in scored)
    halluc = sum(r.hallucinated for r in scored)
    errs = n - len(scored)
    avg_lat = sum(r.latency_seconds for r in scored) / len(scored) if scored else 0
    d = len(scored) or 1
    print(f"\n{'='*56}")
    print(f"Questions scored: {len(scored)}/{n}  (errors: {errs})")
    print(f"Answer accuracy:    {correct}/{len(scored)} ({100*correct/d:.1f}%)")
    print(f"Faithfulness rate:  {faithful}/{len(scored)} ({100*faithful/d:.1f}%)")
    print(f"Hallucination rate: {halluc}/{len(scored)} ({100*halluc/d:.1f}%)")
    print(f"Avg latency:        {avg_lat:.1f}s")
    by = {}
    for r in scored:
        by.setdefault(r.ticker, []).append(r)
    print("Per ticker (correct / faithful / n):")
    for t, rs in sorted(by.items()):
        print(f"  {t:5}: {sum(x.correct for x in rs)} / {sum(x.faithful for x in rs)} / {len(rs)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["gcc", "financebench"], default="gcc")
    args = ap.parse_args()
    res = run_full_eval(set_name=args.set)
    out = config.DATA_DIR / ("gcc_eval" if args.set == "gcc" else "financebench") / "full_eval_results.json"
    json.dump([asdict(r) for r in res], open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")
