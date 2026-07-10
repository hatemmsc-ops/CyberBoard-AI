"""Load FinanceBench dataset and filter for companies in our corpus."""

import json
from pathlib import Path
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

COMPANY_MAP = {
    "Amazon": "AMZN",
    "Coca-Cola": "KO",
    "Johnson & Johnson": "JNJ",
    "JPMorgan": "JPM",
    "Microsoft": "MSFT",
    "Pfizer": "PFE",
    "Walmart": "WMT",
}


def load_financebench(save: bool = True) -> list[dict]:
    """Load FinanceBench and return questions matching our 20 companies."""
    ds = load_dataset("PatronusAI/financebench", split="train")

    all_questions = []
    matched = []

    for ex in ds:
        record = {
            "id": ex["financebench_id"],
            "company": ex["company"],
            "question": ex["question"],
            "answer": ex["answer"],
            "justification": ex["justification"],
            "question_type": ex["question_type"],
            "question_reasoning": ex["question_reasoning"],
            "doc_name": ex["doc_name"],
            "doc_type": ex["doc_type"],
            "doc_period": ex["doc_period"],
            "evidence": ex["evidence"],
        }
        all_questions.append(record)

        ticker = COMPANY_MAP.get(ex["company"])
        if ticker:
            record["ticker"] = ticker
            matched.append(record)

    out_dir = config.DATA_DIR / "financebench"
    out_dir.mkdir(parents=True, exist_ok=True)

    if save:
        with open(out_dir / "all_questions.json", "w") as f:
            json.dump(all_questions, f, indent=2)
        with open(out_dir / "matched_questions.json", "w") as f:
            json.dump(matched, f, indent=2)

    print(f"FinanceBench: {len(all_questions)} total, {len(matched)} matched to our corpus")
    by_ticker = {}
    for q in matched:
        by_ticker.setdefault(q["ticker"], []).append(q)
    for t, qs in sorted(by_ticker.items()):
        print(f"  {t}: {len(qs)} questions")

    return matched


if __name__ == "__main__":
    load_financebench()
