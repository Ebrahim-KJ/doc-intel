"""Chart nDCG@10 by retrieval method across the benchmark datasets."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [("scifact", "SciFact\n(science claims)"), ("nfcorpus", "NFCorpus\n(medical)"), ("fiqa", "FiQA\n(finance Q&A)")]
METHODS = [("BM25", "#8a8984"), ("Dense", "#2a78d6"), ("Hybrid (RRF)", "#eb6834")]


def main():
    res = {d: json.loads((ROOT / "results" / f"retrieval_{d}.json").read_text()) for d, _ in DATASETS}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.edgecolor": "#e6e5e1", "axes.grid": True, "axes.grid.axis": "y",
                         "grid.color": "#e6e5e1", "font.family": "DejaVu Sans", "ytick.left": False,
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(DATASETS))
    w = 0.24
    for i, (m, color) in enumerate(METHODS):
        vals = [res[d]["table"][m]["ndcg@10"] for d, _ in DATASETS]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=color, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(x, [label for _, label in DATASETS])
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 0.8)
    ax.set_title("Hybrid helps where keyword search is competitive, and hurts where it is not", loc="left", fontweight="bold", pad=22)
    ax.text(0, 1.03, "BEIR test queries with human relevance labels; dense = all-MiniLM-L6-v2",
            transform=ax.transAxes, color="#52514e", fontsize=9)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "retrieval_ndcg.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
