#!/usr/bin/env python3
"""OFAT experiments: FIM (6 dims) + HUIM (4 dims + delta sweep)."""
import sys, json, subprocess, shutil, time, random, tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "generator"))
from generate import FIMGenerator, HUIMGenerator
from validate import structural_check, statistical_check, groundtruth_check, decoupling_table

SPMF  = Path(__file__).parent.parent / "spmf.jar"
OUT   = Path(__file__).parent.parent / "results" / "tables"
DSETS = Path(__file__).parent.parent / "datasets"

# ── Base configs ───────────────────────────────────────────────────────────────
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

# ── OFAT dimensions ────────────────────────────────────────────────────────────
FIM_DIMS = {
    "num_transactions": [200, 500, 1000, 2000, 5000],
    "num_items":        [20,  50,  100,  200,  500],
    "density":          [0.02, 0.05, 0.10, 0.20, 0.35],
    "avg_length":       [3,   5,   8,   12,   18],
    "noise_ratio":      [0.0, 0.10, 0.20, 0.40, 0.60],
    "num_patterns":     [2,   5,   10,  20,   40],
}
HUIM_DIMS = {
    "Q_max":           [2,   5,  10,  20,   50],
    "decouple_ratio":  [0.0, 0.10, 0.25, 0.50, 0.80],
    "profit_max":      [10,  25,  50,  100,  200],
    "num_transactions":[200, 500, 1000, 2000, 5000],
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_patterns(n, num_items, seed=0):
    rng, patterns, used = random.Random(seed), [], set()
    cap = max(3, num_items - 1)
    for _ in range(n):
        avail = [i for i in range(2, cap + 1) if i not in used]
        if len(avail) < 2: break
        items = sorted(rng.sample(avail, min(rng.randint(2, 3), len(avail))))
        used.update(items)
        patterns.append({"itemset": items, "target_support": round(rng.uniform(0.08, 0.20), 2)})
    return patterns or BASE_FIM["ground_truth"]

def make_fim_cfg(dim, value):
    cfg = deepcopy(BASE_FIM)
    if   dim == "num_transactions": cfg["num_transactions"] = value
    elif dim == "num_items":
        cfg["num_items"] = value
        cfg["density"]   = round(BASE_FIM["avg_length"] / max(value, 1), 4)
        cfg["ground_truth"] = make_patterns(3, value)
    elif dim == "density":
        cfg["density"]    = value
        cfg["avg_length"] = max(2, int(value * BASE_FIM["num_items"]))
    elif dim == "avg_length":
        cfg["avg_length"] = value
        cfg["density"]    = round(value / BASE_FIM["num_items"], 4)
    elif dim == "noise_ratio":   cfg["noise_ratio"] = value
    elif dim == "num_patterns":  cfg["ground_truth"] = make_patterns(value, BASE_FIM["num_items"])
    return cfg

def make_huim_cfg(dim, value):
    cfg = deepcopy(BASE_HUIM)
    if   dim == "Q_max":           cfg["huim"]["quantity_range"] = [1, value]
    elif dim == "decouple_ratio":  cfg["huim"]["decouple_ratio"] = value
    elif dim == "profit_max":      cfg["huim"]["profit_range"]   = [1, value]
    elif dim == "num_transactions":cfg["num_transactions"]       = value
    return cfg

def run_spmf(algo, dat_path, param, timeout=60):
    out = dat_path.parent / "spmf_out.txt"
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

# ── FIM OFAT ───────────────────────────────────────────────────────────────────
def run_fim_ofat():
    results = {}
    for dim, values in FIM_DIMS.items():
        print(f"\n[FIM OFAT] {dim}")
        results[dim] = []
        for v in values:
            cfg = make_fim_cfg(dim, v)
            row = {"value": v}
            try:
                gen = FIMGenerator(cfg, seed=42)
                txs = gen.generate()
                save_dir = DSETS / "fim" / f"ofat_{dim}" / str(v).replace(".", "_")
                dat, gt  = gen.save(txs, save_dir, prefix="dataset")

                stat = statistical_check(txs)
                gtc  = groundtruth_check(txs, gt)
                row.update({"recall": gtc.get("recall"),
                             "powerlaw_alpha": stat["powerlaw_alpha"],
                             "skewness": stat["length_skewness"]})

                if SPMF.exists():
                    fp, ms = run_spmf("FPGrowth_itemsets", dat, 0.10)
                    if fp:
                        found = parse_fim(fp)
                        row.update({"fpgrowth_ms": ms, "fpgrowth_n": len(found),
                                    "fpgrowth_recall": recall(cfg["ground_truth"], found)})
                    ec, ms2 = run_spmf("Eclat", dat, 0.10)
                    if ec:
                        found2 = parse_fim(ec)
                        row.update({"eclat_ms": ms2, "eclat_n": len(found2),
                                    "eclat_recall": recall(cfg["ground_truth"], found2)})
            except Exception as e:
                row["error"] = str(e)
            results[dim].append(row)
            print(f"  {dim}={v} -> recall={row.get('recall')}, alpha={row.get('powerlaw_alpha')}, fp={row.get('fpgrowth_ms')}ms")
    return results

# ── HUIM OFAT ──────────────────────────────────────────────────────────────────
def run_huim_ofat():
    results = {}

    # Delta sweep on one fixed dataset
    print("\n[HUIM OFAT] delta sweep")
    tmpdir = Path(tempfile.mkdtemp())
    try:
        gen = HUIMGenerator(BASE_HUIM, seed=42)
        txs = gen.generate_huim()
        save_dir = DSETS / "huim" / "ofat_delta" / "base"
        dat, prof, gt = gen.save_huim(txs, save_dir, prefix="dataset")
        total_util  = sum(u for _, _, u in txs)
        delta_vals  = [max(200, int(total_util * f)) for f in [0.001, 0.003, 0.005, 0.010, 0.020]]
        results["delta"] = []
        for delta in delta_vals:
            row = {"value": delta}
            if SPMF.exists():
                out, ms = run_spmf("FHM", dat, delta)
                if out:
                    found = parse_huim(out)
                    row.update({"n_huis": len(found), "runtime_ms": ms,
                                "recall": recall(BASE_HUIM["huim"]["ground_truth_huis"], found)})
            results["delta"].append(row)
            print(f"  delta={delta} -> n_huis={row.get('n_huis')}, rt={row.get('runtime_ms')}ms")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Other HUIM dimensions
    for dim, values in HUIM_DIMS.items():
        print(f"\n[HUIM OFAT] {dim}")
        results[dim] = []
        for v in values:
            cfg = make_huim_cfg(dim, v)
            row = {"value": v}
            try:
                gen = HUIMGenerator(cfg, seed=42)
                txs = gen.generate_huim()
                save_dir = DSETS / "huim" / f"ofat_{dim}" / str(v).replace(".", "_")
                dat, prof, gt = gen.save_huim(txs, save_dir, prefix="dataset")
                total_util = sum(u for _, _, u in txs)
                delta      = max(200, int(total_util * 0.005))
                dt = decoupling_table(txs, gt)
                row.update({"total_utility": int(total_util),
                             "B_rare_hui": dt.get("B_rare_hui"),
                             "C_frequent_cheap": dt.get("C_frequent_cheap"),
                             "non_trivial": dt.get("non_trivial")})
                if SPMF.exists():
                    out, ms = run_spmf("FHM", dat, delta)
                    if out:
                        found = parse_huim(out)
                        row.update({"n_huis": len(found), "runtime_ms": ms,
                                    "recall": recall(cfg["huim"]["ground_truth_huis"], found)})
            except Exception as e:
                row["error"] = str(e)
            results[dim].append(row)
            print(f"  {dim}={v} -> util={row.get('total_utility')}, B={row.get('B_rare_hui')}, C={row.get('C_frequent_cheap')}")
    return results

# ── Reproducibility ────────────────────────────────────────────────────────────
def run_reproducibility(n_runs=5):
    metrics = {"recall": [], "powerlaw_alpha": [], "skewness": [], "avg_length_err": []}
    print("\n[Reproducibility] 5 runs with different seeds")
    for seed in range(n_runs):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            gen = FIMGenerator(BASE_FIM, seed=seed)
            txs = gen.generate()
            _, gt_path = gen.save(txs, tmpdir, prefix="tmp")
            stat   = statistical_check(txs)
            gtc    = groundtruth_check(txs, gt_path)
            struct = structural_check(txs, BASE_FIM)
            metrics["recall"].append(gtc.get("recall") or 0)
            metrics["powerlaw_alpha"].append(stat["powerlaw_alpha"] or 0)
            metrics["skewness"].append(stat["length_skewness"])
            metrics["avg_length_err"].append(struct["avg_length"]["error_pct"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    result = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        result[k] = {"mean": round(float(arr.mean()), 4),
                     "std":  round(float(arr.std()),  4),
                     "variance_pct": round(float(arr.std() / max(abs(arr.mean()), 1e-9) * 100), 2)}
        print(f"  {k}: mean={result[k]['mean']}, std={result[k]['std']}, var%={result[k]['variance_pct']}")
    return result

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    fim  = run_fim_ofat()
    with open(OUT / "ofat_fim.json", "w") as f: json.dump(fim, f, indent=2)
    print(f"\nFIM  -> {OUT / 'ofat_fim.json'}")

    huim = run_huim_ofat()
    with open(OUT / "ofat_huim.json", "w") as f: json.dump(huim, f, indent=2)
    print(f"HUIM -> {OUT / 'ofat_huim.json'}")

    repro = run_reproducibility()
    with open(OUT / "reproducibility.json", "w") as f: json.dump(repro, f, indent=2)
    print(f"Reproducibility -> {OUT / 'reproducibility.json'}")
