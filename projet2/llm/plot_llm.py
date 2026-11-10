#!/usr/bin/env python3
"""Generate comparison figure: LLM variants vs classical generator."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent.parent
TABLES   = BASE_DIR / "results" / "tables"
FIGURES  = BASE_DIR / "results" / "figures"


def plot_llm_comparison():
    path = TABLES / "llm_comparison.json"
    if not path.exists():
        print("llm_comparison.json not found — run compare.py first"); return

    data      = json.loads(path.read_text())
    classical = data["classical"]
    llm       = data["llm"]

    names     = ["classical"] + list(llm.keys())
    colors    = ["#238636", "#58a6ff", "#f0883e", "#bc8cff", "#3fb950"]

    def safe(d, key):
        v = d.get(key)
        return float(v) if v is not None and not isinstance(v, str) else 0.0

    alphas    = [safe(classical, "alpha")]   + [safe(r, "alpha")    for r in llm.values()]
    skews     = [safe(classical, "skewness")]+ [safe(r, "skewness") for r in llm.values()]
    times     = [safe(classical, "elapsed_s")]+ [safe(r, "elapsed_s") for r in llm.values()]
    n_txs     = [safe(classical, "n_transactions")] + [safe(r, "n_transactions") for r in llm.values()]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("LLM Variants vs Classical Generator", fontsize=14, fontweight="bold")

    def bar(ax, vals, title, ylabel, ref=None):
        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="#30363d", linewidth=0.8)
        if ref is not None:
            ax.axhline(ref, color="red", linestyle="--", linewidth=1.2, label=f"target={ref}")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.set_facecolor("#0d1117")
        ax.spines[["top","right"]].set_visible(False)
        for bar_, val in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height() + 0.01 * max(vals or [1]),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    bar(axes[0], alphas,  "Power-law Alpha",        "alpha",  ref=1.8)
    bar(axes[1], skews,   "Length Skewness",         "skewness", ref=1.0)
    bar(axes[2], times,   "Generation Time (s)",     "seconds")
    bar(axes[3], n_txs,   "Transactions Generated",  "count",
        ref=data["config"]["num_transactions"])

    # Add reference lines for alpha range
    axes[0].axhspan(1.5, 3.0, alpha=0.1, color="green", label="valid range [1.5, 3.0]")
    axes[0].legend(fontsize=7)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "llm_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#161b22")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    plot_llm_comparison()
