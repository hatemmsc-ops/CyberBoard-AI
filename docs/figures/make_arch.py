"""Regenerate Figure 1, the CyberBoard-AI architecture diagram.

The previous fig_architecture.png was a static export with no source file and
no canvas padding, so the rotated 'reads index' label and the right hand dashed
panel borders were clipped at the image edge. This rebuilds it from code with
explicit margins, keeps every box and label identical in wording, and reserves
a clear right hand channel for the 'reads index' feedback arrow so it no longer
crosses any box.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

NAVY, EDGE = "#121A2F", "#1a2740"
BOX, BOX2, PANEL = "#e4ecf7", "#cfdcee", "#f6f9fd"
GOLD, GOLDE, LABEL = "#c9a227", "#9d7d15", "#42597a"

W, H = 100.0, 136.0
fig, ax = plt.subplots(figsize=(10.6, 14.4))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.invert_yaxis()

def panel(x, y, w, h, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PANEL, edgecolor="#9db3cc",
                           linestyle=(0, (6, 4)), linewidth=1.3, zorder=1))
    ax.text(x + 2.6, y + 3.6, label, fontsize=11.5, fontweight="bold",
            color=LABEL, zorder=3, family="DejaVu Sans", va="center")

def box(x, y, w, h, lines, fc=BOX, tc=NAVY, bold_first=False, fs=10.5, ec=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                 facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2))
    step = 3.4; y0 = y + h / 2 - (len(lines) - 1) * step / 2
    for i, t in enumerate(lines):
        heavy = fc in (NAVY, GOLD) or (bold_first and i == 0)
        ax.text(x + w / 2, y0 + i * step, t, ha="center", va="center", fontsize=fs,
                color=tc, zorder=3, family="DejaVu Sans",
                fontweight="bold" if heavy else "normal")
    return (x, y, w, h)

def arrow(p1, p2, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                 linewidth=1.5, color=EDGE, zorder=4,
                 linestyle=(0, (5, 3)) if dashed else "solid",
                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

bot = lambda b: (b[0] + b[2] / 2, b[1] + b[3])
top = lambda b: (b[0] + b[2] / 2, b[1])
rgt = lambda b: (b[0] + b[2], b[1] + b[3] / 2)
lft = lambda b: (b[0], b[1] + b[3] / 2)

# ---------------- offline ----------------
panel(4, 3, 84, 56, "OFFLINE INDEXING")
sec   = box(9,  9,  31, 11, ["SEC EDGAR filings", "20 US companies, 10-K / 10-Q"], fs=10)
bhb   = box(46, 9,  31, 11, ["Bahrain Bourse", "GCC annual reports"], fs=10)
parse = box(9,  27, 21, 11, ["Parse into", "named sections"], fs=10)
chunk = box(34, 27, 23, 11, ["Recursive chunk", "512 tokens, 64 overlap"], fs=10)
emb   = box(61, 27, 19, 11, ["Embed", "gemini-embedding-001", "1536-dim"], fs=9.2)
chro  = box(12, 45, 64, 10, ["ChromaDB vector store",
            "18,136 chunks   ·   isolated US and GCC collections"], fc=NAVY, tc="white", fs=10.5)
arrow(bot(sec), (17.5, 27))
arrow(bot(bhb), (46, 27.6), rad=0.20)
arrow(rgt(parse), lft(chunk))
arrow(rgt(chunk), lft(emb))
arrow(bot(emb), (50, 45), rad=0.16)

# ---------------- online ----------------
panel(4, 64, 84, 68, "ONLINE QUERY AND REASONING")
qry   = box(9,  71,  21, 11, ["Board member", "query"], fs=10)
agent = box(36, 71,  34, 11, ["ReAct agent", "Gemini flash, temperature 0.1"],
            fc=BOX2, bold_first=True, fs=10.5)
fin   = box(9,  90,  21, 10, ["Financial", "summarizer"], fs=10)
risk  = box(33, 90,  19, 10, ["Risk", "identifier"], fs=10)
comp  = box(55, 90,  19, 10, ["Compliance", "checker"], fs=10)
hyb   = box(9,  106, 65, 12, ["Hybrid retrieval",
            "BM25 sparse  +  dense similarity   →   alpha-weighted fusion (0.7)",
            "→   cross-encoder rerank   →   top-k passages"], fs=9.4)
ans   = box(12, 121, 61, 9, ["Answer   +   source attribution (document, section)",
            "+   confidence score   +   advisory disclaimer"], fc=GOLD, tc=NAVY, ec=GOLDE, fs=10.2)

arrow(rgt(qry), lft(agent))
arrow((46, 82), top(fin), rad=0.0)
arrow((52, 82), top(risk), rad=0.0)
arrow((58, 82), top(comp), rad=0.0)
arrow(bot(fin),  (20, 106))
arrow(bot(risk), (42, 106))
arrow(bot(comp), (62, 106))
arrow(bot(hyb),  top(ans))

# 'reads index' runs up a clear right hand channel, crossing no box
arrow((74.5, 112), (80.0, 55.4), dashed=True, rad=-0.14)
ax.text(84.4, 84, "reads index", rotation=90, ha="center", va="center",
        fontsize=9.8, style="italic", color=LABEL, family="DejaVu Sans")

fig.savefig("/tmp/hcheck/fig/fig_architecture.png", dpi=300,
            bbox_inches="tight", pad_inches=0.30, facecolor="white")
print("ok")
