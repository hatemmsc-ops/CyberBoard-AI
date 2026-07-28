"""Generate thesis figures from the saved evaluation results.

Produces two print-friendly PNGs in docs/figures/:
  fig_baseline_ladder.png : accuracy by configuration (no retrieval, naive RAG,
                            agent) for each corpus.
  fig_ablation.png        : retrieval hit@10 by configuration.

Colours use a ColorBrewer Blues sequential ramp, which is colourblind safe. The
baseline configurations form an ordered progression, so a single hue from light
to dark reads as increasing capability.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE_LIGHT, BLUE_MID, BLUE_DARK = "#9ecae1", "#4292c6", "#08519c"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#e6e6e6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def accuracy(path: Path) -> float:
    rows = json.load(open(path))
    scored = [r for r in rows if not r.get("error")]
    return 100 * sum(r["correct"] for r in scored) / (len(scored) or 1)


def baseline_ladder():
    corpora = [("GCC (Bahrain Bourse)", "gcc_eval"), ("SEC-recent (US)", "sec_recent")]
    configs = [
        ("No retrieval", "baseline_no_retrieval_results.json", BLUE_LIGHT),
        ("Naive RAG", "baseline_naive_rag_results.json", BLUE_MID),
        ("Agent (ReAct)", "full_eval_results.json", BLUE_DARK),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width, xs = 0.26, range(len(corpora))
    for j, (label, fname, colour) in enumerate(configs):
        vals = [accuracy(DATA / d / fname) for _, d in corpora]
        pos = [x + (j - 1) * width for x in xs]
        bars = ax.bar(pos, vals, width, label=label, color=colour,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels([c for c, _ in corpora])
    ax.set_ylabel("Answer accuracy (percent)"); ax.set_ylim(0, 108)
    ax.set_title("What retrieval and the agent loop add", fontsize=12.5, pad=12, loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_baseline_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ablation():
    # A dot plot, not a bar chart: the hit rates cluster in a narrow band (91 to
    # 100), and a bar chart with a non-zero baseline would exaggerate the gaps.
    # A dot plot shows position honestly, so a zoomed x-axis is legitimate.
    rows = json.load(open(DATA / "gcc_eval" / "ablation_results.json"))
    rows = sorted(rows, key=lambda r: r["hit_rate"])
    labels = [r["config_name"].replace("_", " ") for r in rows]
    vals = [100 * r["hit_rate"] for r in rows]
    ys = range(len(rows))
    colours = [BLUE_DARK if v == max(vals) else BLUE_MID for v in vals]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for y, v, c in zip(ys, vals, colours):
        ax.plot([85, v], [y, y], color=GRID, linewidth=1.4, zorder=1)  # leader line
        ax.scatter([v], [y], s=130, color=c, zorder=3, edgecolor="white", linewidth=1.0)
        ax.text(v + 0.4, y, f"{v:.1f}", ha="left", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(list(ys)); ax.set_yticklabels(labels)
    ax.set_xlabel("Retrieval hit@10 (percent)"); ax.set_xlim(85, 102)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("GCC retrieval ablation", fontsize=12.5, pad=12, loc="left")
    ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    baseline_ladder()
    ablation()
    print(f"Wrote figures to {OUT}")
