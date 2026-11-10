#!/usr/bin/env python3
"""Generate all figures from experiment results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLES  = Path(__file__).parent.parent / "results" / "tables"
FIGURES = Path(__file__).parent.parent / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DIM_LABELS = {
    "num_transactions": "# Transactions", "num_items": "# Items",
    "density": "Density (rho)", "avg_length": "Avg Length",
    "noise_ratio": "Noise Ratio", "num_patterns": "# Planted Patterns",
    "Q_max": "Max Quantity", "decouple_ratio": "Decouple Ratio",
    "profit_max": "Max Profit", "delta": "Utility Threshold",
}

def load(name):
    p = TABLES / name
    if not p.exists():
        print(f"  missing: {name}"); return None
    with open(p) as f: return json.load(f)

def save(fig, name):
    out = FIGURES / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

# ── FIM OFAT: Recall curves ───────────────────────────────────────────────────
def plot_fim_recall():
    data = load("ofat_fim.json")
    if not data: return
    dims = list(data.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("FIM OFAT — Recall@P vs Each Dimension", fontsize=14)
    for ax, dim in zip(axes.flatten(), dims):
        rows = data[dim]
        x  = [r["value"] for r in rows]
        y  = [r.get("recall") or 0 for r in rows]
        yf = [r.get("fpgrowth_recall") or 0 for r in rows]
        ax.plot(x, y,  marker="o", linewidth=2, color="steelblue",  label="Generator recall")
        ax.plot(x, yf, marker="s", linewidth=2, color="darkorange", linestyle="--", label="FP-Growth recall")
        ax.axhline(0.80, linestyle=":", color="red", linewidth=1, label="0.80 threshold")
        ax.set_title(DIM_LABELS.get(dim, dim)); ax.set_xlabel(DIM_LABELS.get(dim, dim))
        ax.set_ylabel("Recall@P"); ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for ax in axes.flatten()[len(dims):]: ax.set_visible(False)
    plt.tight_layout()
    save(fig, "fim_ofat_recall.png")

# ── FIM OFAT: Power-law alpha ─────────────────────────────────────────────────
def plot_fim_alpha():
    data = load("ofat_fim.json")
    if not data: return
    dims = list(data.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("FIM OFAT — Power-law Alpha vs Each Dimension", fontsize=14)
    for ax, dim in zip(axes.flatten(), dims):
        rows = data[dim]
        x = [r["value"] for r in rows]
        y = [r.get("powerlaw_alpha") or 0 for r in rows]
        ax.plot(x, y, marker="s", linewidth=2, color="darkorange")
        ax.axhline(1.5, linestyle="--", color="gray", linewidth=1)
        ax.axhline(3.0, linestyle="--", color="gray", linewidth=1, label="valid [1.5, 3.0]")
        ax.set_title(DIM_LABELS.get(dim, dim)); ax.set_xlabel(DIM_LABELS.get(dim, dim))
        ax.set_ylabel("Power-law alpha"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for ax in axes.flatten()[len(dims):]: ax.set_visible(False)
    plt.tight_layout()
    save(fig, "fim_ofat_alpha.png")

# ── FIM OFAT: FP-Growth vs Eclat runtime ─────────────────────────────────────
def plot_fim_runtime():
    data = load("ofat_fim.json")
    if not data: return
    rows = data.get("num_transactions", [])
    if not any(r.get("fpgrowth_ms") for r in rows): return
    x  = [r["value"] for r in rows]
    fp = [r.get("fpgrowth_ms") or 0 for r in rows]
    ec = [r.get("eclat_ms")    or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, fp, marker="o", linewidth=2, color="steelblue",  label="FP-Growth")
    ax.plot(x, ec, marker="s", linewidth=2, color="darkorange", label="Eclat")
    ax.set_title("FIM: FP-Growth vs Eclat Runtime (theta=0.10)")
    ax.set_xlabel("# Transactions"); ax.set_ylabel("Runtime (ms)")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "fim_runtime_comparison.png")

# ── HUIM OFAT: utility + decoupling ──────────────────────────────────────────
def plot_huim_ofat():
    data = load("ofat_huim.json")
    if not data: return
    dims = [d for d in data.keys() if d != "delta"]
    if not dims: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("HUIM OFAT — Total Utility & Decoupling", fontsize=14)
    for ax, dim in zip(axes.flatten(), dims[:4]):
        rows = data[dim]
        x    = [r["value"] for r in rows]
        util = [r.get("total_utility") or 0 for r in rows]
        B    = [r.get("B_rare_hui")      or 0 for r in rows]
        C    = [r.get("C_frequent_cheap") or 0 for r in rows]
        ax2  = ax.twinx()
        ax.bar(range(len(x)), util, color="steelblue", alpha=0.5, label="Total utility")
        ax2.plot(range(len(x)), B, marker="o", color="green", linewidth=2, label="B (rare HUI)")
        ax2.plot(range(len(x)), C, marker="s", color="red",   linewidth=2, label="C (freq cheap)")
        ax.set_xticks(range(len(x))); ax.set_xticklabels([str(v) for v in x], rotation=30)
        ax.set_title(DIM_LABELS.get(dim, dim)); ax.set_ylabel("Total utility")
        ax2.set_ylabel("# items in B / C")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7)
        ax.grid(True, alpha=0.2)
    for ax in axes.flatten()[len(dims):]: ax.set_visible(False)
    plt.tight_layout()
    save(fig, "huim_ofat_decoupling.png")

# ── HUIM delta sweep ──────────────────────────────────────────────────────────
def plot_delta_sweep():
    data = load("ofat_huim.json")
    if not data or "delta" not in data: return
    rows = data["delta"]
    if not any(r.get("n_huis") for r in rows): return
    x  = [r["value"]          for r in rows]
    n  = [r.get("n_huis")  or 0 for r in rows]
    rt = [r.get("runtime_ms") or 0 for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("HUIM: Effect of Utility Threshold Delta", fontsize=13)
    ax1.plot(x, n,  marker="o", color="steelblue",  linewidth=2)
    ax1.set_title("# HUIs vs Delta"); ax1.set_xlabel("Delta"); ax1.set_ylabel("# HUIs")
    ax1.grid(True, alpha=0.3)
    ax2.plot(x, rt, marker="s", color="darkorange", linewidth=2)
    ax2.set_title("Runtime vs Delta"); ax2.set_xlabel("Delta"); ax2.set_ylabel("Runtime (ms)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "huim_delta_sweep.png")

# ── FIM algorithm comparison ──────────────────────────────────────────────────
def plot_spmf_fim():
    data = load("spmf_fim_comparison.json")
    if not data: return
    valid = [r for r in data if r.get("fpgrowth_n") is not None]
    if not valid: return
    x     = [r["threshold"]        for r in valid]
    fp_n  = [r.get("fpgrowth_n")   or 0 for r in valid]
    ec_n  = [r.get("eclat_n")      or 0 for r in valid]
    fp_ms = [r.get("fpgrowth_ms")  or 0 for r in valid]
    ec_ms = [r.get("eclat_ms")     or 0 for r in valid]
    fp_r  = [r.get("fpgrowth_recall") or 0 for r in valid]
    ec_r  = [r.get("eclat_recall")    or 0 for r in valid]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("FIM: FP-Growth vs Eclat", fontsize=13)
    for ax, (yf, ye, title, ylabel) in zip(axes, [
        (fp_n,  ec_n,  "# Patterns Found", "# Patterns"),
        (fp_r,  ec_r,  "Recall@P",          "Recall"),
        (fp_ms, ec_ms, "Runtime (ms)",       "ms"),
    ]):
        ax.plot(x, yf, marker="o", color="steelblue",  linewidth=2, label="FP-Growth")
        ax.plot(x, ye, marker="s", color="darkorange", linewidth=2, label="Eclat")
        ax.set_title(title); ax.set_xlabel("Min Support"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "spmf_fim_comparison.png")

# ── HUIM algorithm comparison ─────────────────────────────────────────────────
def plot_spmf_huim():
    data = load("spmf_huim_comparison.json")
    if not data: return
    valid = [r for r in data if r.get("fhm_n") is not None or r.get("ihup_n") is not None]
    if not valid: return
    x      = [r["delta"]          for r in valid]
    fhm_n  = [r.get("fhm_n")   or 0 for r in valid]
    ihup_n = [r.get("ihup_n")  or 0 for r in valid]
    fhm_ms = [r.get("fhm_ms")  or 0 for r in valid]
    ihup_ms= [r.get("ihup_ms") or 0 for r in valid]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("HUIM: FHM vs IHUP", fontsize=13)
    ax1.plot(x, fhm_n,  marker="o", color="steelblue",  linewidth=2, label="FHM")
    ax1.plot(x, ihup_n, marker="s", color="darkorange", linewidth=2, label="IHUP")
    ax1.set_title("# HUIs Found"); ax1.set_xlabel("Delta"); ax1.set_ylabel("# HUIs")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(x, fhm_ms,  marker="o", color="steelblue",  linewidth=2, label="FHM")
    ax2.plot(x, ihup_ms, marker="s", color="darkorange", linewidth=2, label="IHUP")
    ax2.set_title("Runtime (ms)"); ax2.set_xlabel("Delta"); ax2.set_ylabel("ms")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "spmf_huim_comparison.png")

# ── Reproducibility ───────────────────────────────────────────────────────────
def plot_reproducibility():
    data = load("reproducibility.json")
    if not data: return
    metrics = list(data.keys())
    means   = [data[m]["mean"] for m in metrics]
    stds    = [data[m]["std"]  for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metrics, means, yerr=stds, capsize=5, color="steelblue", alpha=0.7)
    ax.set_title("Reproducibility: Mean +/- Std over 5 runs")
    ax.set_ylabel("Value"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save(fig, "reproducibility.png")

if __name__ == "__main__":
    print("Generating figures...")
    plot_fim_recall()
    plot_fim_alpha()
    plot_fim_runtime()
    plot_huim_ofat()
    plot_delta_sweep()
    plot_spmf_fim()
    plot_spmf_huim()
    plot_reproducibility()
    print("Done.")
