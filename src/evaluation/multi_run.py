"""Repeat the agent evaluation N times and report mean and standard deviation.

The agent runs at temperature 0.1, so accuracy and faithfulness vary a little
between runs (a compound question occasionally drops a sub-part, and transient
provider errors leave a question or two unscored). Reporting mean and standard
deviation over several runs turns a single-run number into a defensible result.
"""

import json
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.evaluation.full_eval import run_full_eval, _load_questions


def _load_questions_count(set_name: str) -> list:
    path = (config.DATA_DIR / "gcc_eval" / "questions.json" if set_name == "gcc"
            else config.DATA_DIR / "sec_recent" / "questions.json")
    return _load_questions(path)


def multi_run(set_name: str, n_runs: int = 5) -> dict:
    per_run = []
    for i in range(n_runs):
        print(f"\n########## {set_name} run {i+1}/{n_runs} ##########")
        results = run_full_eval(set_name=set_name)
        scored = [r for r in results if not r.error]
        n = len(scored) or 1
        per_run.append({
            "run": i + 1,
            "scored": len(scored),
            "errors": len(results) - len(scored),
            "accuracy": 100 * sum(r.correct for r in scored) / n,
            "faithfulness": 100 * sum(r.faithful for r in scored) / n,
        })

    # A run where the provider cut off (few or no questions scored) is an outage,
    # not a measurement. Exclude runs that did not score most of the set, so a
    # rate-limit or spend-cap block cannot masquerade as a 0 percent result.
    threshold = 0.5 * len(_load_questions_count(set_name))
    valid = [r for r in per_run if r["scored"] >= threshold]

    def ms(key):
        vals = [r[key] for r in valid]
        if not vals:
            return 0.0, 0.0
        return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    acc_m, acc_s = ms("accuracy")
    f_m, f_s = ms("faithfulness")
    summary = {
        "set": set_name, "n_runs": n_runs, "valid_runs": len(valid),
        "excluded_runs": len(per_run) - len(valid), "per_run": per_run,
        "accuracy_mean": acc_m, "accuracy_std": acc_s,
        "faithfulness_mean": f_m, "faithfulness_std": f_s,
    }

    print(f"\n{'='*60}\nMULTI-RUN SUMMARY ({set_name})")
    for r in per_run:
        flag = "" if r["scored"] >= threshold else "  [EXCLUDED: provider cut-off]"
        print(f"  run {r['run']}: acc {r['accuracy']:.1f}%  faith {r['faithfulness']:.1f}%  "
              f"(scored {r['scored']}, errors {r['errors']}){flag}")
    print(f"  Valid runs: {len(valid)} of {n_runs}")
    print(f"  Accuracy:     {acc_m:.1f} +/- {acc_s:.1f} percent")
    print(f"  Faithfulness: {f_m:.1f} +/- {f_s:.1f} percent")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["gcc", "sec_recent"], required=True)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    summary = multi_run(args.set, args.runs)
    out_dir = config.DATA_DIR / ("gcc_eval" if args.set == "gcc" else "sec_recent")
    json.dump(summary, open(out_dir / "multi_run_summary.json", "w"), indent=2)
    print(f"\nSaved to {out_dir / 'multi_run_summary.json'}")
