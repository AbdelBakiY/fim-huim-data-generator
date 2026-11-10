#!/usr/bin/env python3
"""Algorithm comparison: FP-Growth vs Eclat (FIM), FHM vs IHUP (HUIM)."""
import sys, json, subprocess, time, tempfile, shutil
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent / "generator"))
from generate import FIMGenerator, HUIMGenerator

SPMF = Path(__file__).parent.parent / "spmf.jar"
OUT  = Path(__file__).parent.parent / "results" / "tables"

BASE_FIM = {
    "num_transactions": 1000, "num_items": 50, "avg_length": 8, "density": 0.16,
    "noise_ratio": 0.15,
    "ground_truth": [
        {"itemset": [3, 7, 12], "target_support": 0.25},
        {"itemset": [5, 9],     "target_support": 0.15},
        {"itemset": [2, 8, 14], "target_support": 0.10},
    ],
}
BASE_HUIM = {
    "num_transactions": 1000, "num_items": 50, "avg_length": 8, "density": 0.16,
    "noise_ratio": 0.15,
    "ground_truth": [
        {"itemset": [3, 7, 12], "target_support": 0.20},
        {"itemset": [5, 9],     "target_support": 0.12},
    ],
    "huim": {
        "profit_distribution": "power_law", "profit_range": [1, 100],
        "quantity_distribution": "uniform", "quantity_range": [1, 5],
        "min_utility_threshold": 500, "decouple_ratio": 0.30,
        "ground_truth_huis": [{"itemset": [3, 7], "target_utility": 3000}],
    },
}

def run_spmf(algo, dat_path, param, timeout=120):
    out = dat_path.parent / f"{algo}_out.txt"
    cmd = ["java", "-jar", str(SPMF), "run", algo, str(dat_path), str(out), str(param)]
    t0  = time.time()
    r   = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    ms  = int((time.time() - t0) * 1000)
    return (out if out.exists() and r.returncode == 0 else None), ms

def parse_fim(path):
    s = set()
    with open(path) as f:
        for line in f:
            if "#SUP:" in line:
                s.add(frozenset(map(int, line.split("#SUP:")[0].strip().split())))
    return s

def parse_huim(path):
    s = set()
    with open(path) as f:
        for line in f:
            tag = "#UTIL:" if "#UTIL:" in line else ("#SUP:" if "#SUP:" in line else None)
            if tag:
                s.add(frozenset(map(int, line.split(tag)[0].strip().split())))
    return s

def recall(planted, found):
    if not planted: return None
    return round(sum(1 for p in planted if frozenset(p["itemset"]) in found) / len(planted), 3)

# ── FIM: FP-Growth vs Eclat across thresholds ─────────────────────────────────
def run_fim_comparison():
    thresholds = [0.05, 0.08, 0.10, 0.15, 0.20]
    results = []
    tmpdir  = Path(tempfile.mkdtemp())
    try:
        gen = FIMGenerator(BASE_FIM, seed=42)
        txs = gen.generate()
        dat, _ = gen.save(txs, tmpdir, prefix="fim_cmp")
        print(f"\n{'thresh':>8} | {'FPG n':>7} | {'FPG ms':>7} | {'FPG rec':>8} | {'Eclat n':>8} | {'Eclat ms':>9} | {'Eclat rec':>10}")
        print("-" * 75)
        for t in thresholds:
            row = {"threshold": t}
            fp, ms1 = run_spmf("FPGrowth_itemsets", dat, t)
            if fp:
                found1 = parse_fim(fp)
                row.update({"fpgrowth_n": len(found1), "fpgrowth_ms": ms1,
                             "fpgrowth_recall": recall(BASE_FIM["ground_truth"], found1)})
            ec, ms2 = run_spmf("Eclat", dat, t)
            if ec:
                found2 = parse_fim(ec)
                row.update({"eclat_n": len(found2), "eclat_ms": ms2,
                             "eclat_recall": recall(BASE_FIM["ground_truth"], found2)})
            results.append(row)
            print(f"  {t:>6} | {row.get('fpgrowth_n','?'):>7} | {row.get('fpgrowth_ms','?'):>7} | "
                  f"{row.get('fpgrowth_recall','?'):>8} | {row.get('eclat_n','?'):>8} | "
                  f"{row.get('eclat_ms','?'):>9} | {row.get('eclat_recall','?'):>10}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return results

# ── HUIM: FHM vs IHUP across delta ────────────────────────────────────────────
def run_huim_comparison():
    results = []
    tmpdir  = Path(tempfile.mkdtemp())
    try:
        gen = HUIMGenerator(BASE_HUIM, seed=42)
        txs = gen.generate_huim()
        dat, prof, _ = gen.save_huim(txs, tmpdir, prefix="huim_cmp")
        total_util = sum(u for _, _, u in txs)
        deltas = [max(500, int(total_util * f)) for f in [0.005, 0.010, 0.015, 0.020, 0.030]]
        print(f"\n{'delta':>8} | {'FHM n':>7} | {'FHM ms':>7} | {'IHUP n':>7} | {'IHUP ms':>8}")
        print("-" * 55)
        for delta in deltas:
            row = {"delta": delta}
            fhm, ms1 = run_spmf("FHM", dat, delta)
            if fhm:
                found1 = parse_huim(fhm)
                row.update({"fhm_n": len(found1), "fhm_ms": ms1})
            ihup, ms2 = run_spmf("IHUP", dat, delta)
            if ihup:
                found2 = parse_huim(ihup)
                row.update({"ihup_n": len(found2), "ihup_ms": ms2})
            results.append(row)
            print(f"  {delta:>6} | {row.get('fhm_n','?'):>7} | {row.get('fhm_ms','?'):>7} | "
                  f"{row.get('ihup_n','?'):>7} | {row.get('ihup_ms','?'):>8}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return results

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if not SPMF.exists():
        print(f"SPMF not found at {SPMF}"); exit(1)

    print("=== FIM: FP-Growth vs Eclat ===")
    fim_cmp = run_fim_comparison()
    with open(OUT / "spmf_fim_comparison.json", "w") as f: json.dump(fim_cmp, f, indent=2)
    print(f"Saved -> {OUT / 'spmf_fim_comparison.json'}")

    print("\n=== HUIM: FHM vs IHUP ===")
    huim_cmp = run_huim_comparison()
    with open(OUT / "spmf_huim_comparison.json", "w") as f: json.dump(huim_cmp, f, indent=2)
    print(f"Saved -> {OUT / 'spmf_huim_comparison.json'}")
