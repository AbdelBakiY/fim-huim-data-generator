#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np


def load_fim(path: Path) -> list:
    transactions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                transactions.append(list(map(int, line.split())))
    return transactions


def load_huim(path: Path) -> list:
    transactions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            items = list(map(int, parts[0].split()))
            tx_util = int(parts[1])
            qtys = list(map(int, parts[2].split()))
            transactions.append((items, dict(zip(items, qtys)), tx_util))
    return transactions


def items_from(transactions: list) -> list:
    if transactions and isinstance(transactions[0], tuple):
        return [i for t, _, _ in transactions for i in t]
    return [i for t in transactions for i in t]


def lengths_from(transactions: list) -> list:
    if transactions and isinstance(transactions[0], tuple):
        return [len(t) for t, _, _ in transactions]
    return [len(t) for t in transactions]


def structural_check(transactions: list, config: dict) -> dict:
    n = len(transactions)
    all_items = items_from(transactions)
    lengths = lengths_from(transactions)

    n_target = config["num_transactions"]
    ni_target = config["num_items"]
    al_target = config["avg_length"]

    n_items_actual = len(set(all_items))
    avg_len_actual = np.mean(lengths) if lengths else 0.0
    density_actual = len(all_items) / (n * ni_target) if n > 0 else 0.0
    density_target = config.get("density", density_actual)

    def check(actual, target, tol):
        err = abs(actual - target) / max(abs(target), 1e-9) * 100
        return {"target": target, "actual": round(actual, 4), "error_pct": round(err, 2), "ok": bool(err < tol * 100)}

    return {
        "num_transactions": check(n, n_target, 0.01),
        "num_items_vocabulary": check(n_items_actual, ni_target, 0.05),
        "avg_length": check(avg_len_actual, al_target, 0.10),
        "density": check(density_actual, density_target, 0.15),
    }


def statistical_check(transactions: list) -> dict:
    all_items = items_from(transactions)
    lengths = lengths_from(transactions)

    counts = Counter(all_items)
    freqs = sorted(counts.values(), reverse=True)

    alpha = None
    r_squared = None
    if len(freqs) > 5:
        log_r = np.log(np.arange(1, len(freqs) + 1, dtype=float))
        log_f = np.log(np.array(freqs, dtype=float) + 1e-9)
        A = np.vstack([log_r, np.ones(len(log_r))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, log_f, rcond=None)
        alpha = -coeffs[0]
        pred = A @ coeffs
        ss_res = np.sum((log_f - pred) ** 2)
        ss_tot = np.sum((log_f - log_f.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lengths_arr = np.array(lengths, dtype=float)
    std = lengths_arr.std()
    skewness = float(np.mean((lengths_arr - lengths_arr.mean()) ** 3) / (std ** 3 + 1e-9)) if std > 0 else 0.0

    return {
        "powerlaw_alpha": round(float(alpha), 3) if alpha is not None else None,
        "alpha_in_range_1_5_to_3_0": bool(1.5 <= alpha <= 3.0) if alpha is not None else False,
        "powerlaw_r2": round(float(r_squared), 3) if r_squared is not None else None,
        "length_skewness": round(skewness, 3),
        "length_right_skewed": bool(skewness > 1),
    }


def groundtruth_check(transactions: list, gt_path: Path) -> dict:
    if not gt_path.exists():
        return {"error": "groundtruth file not found"}

    with open(gt_path) as f:
        gt = json.load(f)

    n = len(transactions)
    trans_sets = [set(t) if not isinstance(t, tuple) else set(t[0]) for t in transactions]

    results = []
    for p in gt.get("patterns", []):
        s = set(p["itemset"])
        measured = sum(1 for t in trans_sets if s.issubset(t)) / n
        target = p["target_support"]
        err = abs(measured - target)
        results.append({
            "itemset": p["itemset"],
            "target_support": target,
            "measured_support": round(measured, 4),
            "absolute_error": round(err, 4),
            "ok": bool(err < 0.05),
        })

    recall = sum(1 for r in results if r["ok"]) / len(results) if results else None
    return {"patterns": results, "recall": round(recall, 3) if recall is not None else None, "recall_ok": bool(recall >= 0.80) if recall is not None else False}


def huim_utility_check(transactions: list, gt_path: Path) -> Optional[dict]:
    if not gt_path.exists() or not transactions or not isinstance(transactions[0], tuple):
        return None

    with open(gt_path) as f:
        gt = json.load(f)

    profit_table = {int(k): v for k, v in gt.get("profit_table", {}).items()}
    total_utility = sum(u for _, _, u in transactions)
    all_items = set(i for t, _, _ in transactions for i in t)
    missing_profits = [i for i in all_items if i not in profit_table]

    decoupling_rows = []
    for p in gt.get("huis", []):
        s = set(p["itemset"])
        n = len(transactions)
        freq = sum(1 for t, _, _ in transactions if s.issubset(t)) / n
        util = sum(
            sum(qtys[i] * profit_table.get(i, 0) for i in items if i in s)
            for items, qtys, _ in transactions
            if s.issubset(items)
        )
        decoupling_rows.append({
            "itemset": p["itemset"],
            "frequency": round(freq, 4),
            "utility": int(util),
            "target_utility": p.get("target_utility"),
        })

    return {
        "total_utility": int(total_utility),
        "missing_profits": missing_profits,
        "profit_integrity_ok": len(missing_profits) == 0,
        "hui_measures": decoupling_rows,
    }


def profit_distribution_check(gt_path: Path) -> dict:
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        gt = json.load(f)
    pt = {int(k): v for k, v in gt.get("profit_table", {}).items()}
    if not pt:
        return {}
    profits = sorted(pt.values(), reverse=True)
    total = sum(profits)
    if total == 0:
        return {}
    top_n = max(1, len(profits) // 10)
    share = sum(profits[:top_n]) / total
    return {"top_10pct_share": round(share, 3), "passes_80_20": bool(share > 0.60)}


def decoupling_table(transactions: list, gt_path: Path, freq_thresh: float = 0.10) -> dict:
    if not gt_path.exists() or not transactions or not isinstance(transactions[0], tuple):
        return {}
    with open(gt_path) as f:
        gt = json.load(f)
    pt = {int(k): v for k, v in gt.get("profit_table", {}).items()}
    n = len(transactions)
    items = set(i for t, _, _ in transactions for i in t)
    sup = {i: sum(1 for t, _, _ in transactions if i in t) / n for i in items}
    unit_profit = {i: pt.get(i, 0) for i in items}
    profit_thresh = sum(unit_profit.values()) / max(len(unit_profit), 1)
    A = sum(1 for i in items if sup[i] >= freq_thresh and unit_profit[i] >= profit_thresh)
    B = sum(1 for i in items if sup[i] <  freq_thresh and unit_profit[i] >= profit_thresh)
    C = sum(1 for i in items if sup[i] >= freq_thresh and unit_profit[i] <  profit_thresh)
    D = sum(1 for i in items if sup[i] <  freq_thresh and unit_profit[i] <  profit_thresh)
    return {
        "A_frequent_hui": A, "B_rare_hui": B,
        "C_frequent_cheap": C, "D_rare_cheap": D,
        "non_trivial": bool(B > 0 and C > 0),
        "freq_thresh": freq_thresh, "profit_thresh": int(profit_thresh),
    }


def validate(dataset_path: str, config_path: str, gt_path: Optional[str], is_huim: bool) -> dict:
    path = Path(dataset_path)
    cfg_path = Path(config_path)

    with open(cfg_path) as f:
        config = json.load(f)

    transactions = load_huim(path) if is_huim else load_fim(path)
    gt = Path(gt_path) if gt_path else path.parent / (path.stem + "_groundtruth.json")

    report = {
        "dataset": str(path),
        "structural": structural_check(transactions, config),
        "statistical": statistical_check(transactions),
        "groundtruth": groundtruth_check(transactions, gt),
    }

    if is_huim:
        report["huim_utility"] = huim_utility_check(transactions, gt)
        report["profit_dist"]  = profit_distribution_check(gt)
        report["decoupling"]   = decoupling_table(transactions, gt)

    return report


def print_report(report: dict):
    print("\n=== VALIDATION REPORT ===")
    print(f"Dataset: {report['dataset']}\n")

    print("[Structural]")
    for k, v in report["structural"].items():
        status = "OK" if v["ok"] else "FAIL"
        print(f"  {k}: target={v['target']}, actual={v['actual']}, err={v['error_pct']}%  [{status}]")

    print("\n[Statistical]")
    s = report["statistical"]
    print(f"  Power-law alpha: {s['powerlaw_alpha']} (in [1.5,3.0]: {s['alpha_in_range_1_5_to_3_0']}, R2={s['powerlaw_r2']})")
    print(f"  Length skewness: {s['length_skewness']} (right-skewed: {s['length_right_skewed']})")

    print("\n[Ground Truth]")
    gt = report["groundtruth"]
    if "error" in gt:
        print(f"  {gt['error']}")
    else:
        for p in gt.get("patterns", []):
            status = "OK" if p["ok"] else "FAIL"
            print(f"  {p['itemset']}: target={p['target_support']}, measured={p['measured_support']}, err={p['absolute_error']}  [{status}]")
        print(f"  Recall: {gt.get('recall')} (>=0.80: {gt.get('recall_ok')})")

    if "huim_utility" in report and report["huim_utility"]:
        print("\n[HUIM Utility]")
        hu = report["huim_utility"]
        print(f"  Total utility: {hu['total_utility']}")
        print(f"  Profit table integrity: {'OK' if hu['profit_integrity_ok'] else 'FAIL'}")
        for r in hu.get("hui_measures", []):
            print(f"  {r['itemset']}: freq={r['frequency']}, utility={r['utility']} (target={r['target_utility']})")

    if report.get("profit_dist"):
        pd_ = report["profit_dist"]
        print(f"\n[Profit Distribution 80/20]")
        print(f"  Top 10% items share: {pd_['top_10pct_share']} (>0.60: {pd_['passes_80_20']})")

    if report.get("decoupling"):
        dc = report["decoupling"]
        print(f"\n[Decoupling Table] (freq_thresh={dc['freq_thresh']}, profit_thresh={dc['profit_thresh']})")
        print(f"  A (frequent+HUI):   {dc['A_frequent_hui']}")
        print(f"  B (rare+HUI):       {dc['B_rare_hui']}")
        print(f"  C (frequent+cheap): {dc['C_frequent_cheap']}")
        print(f"  D (rare+cheap):     {dc['D_rare_cheap']}")
        print(f"  Non-trivial (B>0 and C>0): {dc['non_trivial']}")


def main():
    parser = argparse.ArgumentParser(description="Validate a generated transactional dataset")
    parser.add_argument("dataset", help="Path to .dat dataset file")
    parser.add_argument("config", help="Path to JSON configuration file used for generation")
    parser.add_argument("--groundtruth", default=None, help="Path to groundtruth JSON (optional)")
    parser.add_argument("--huim", action="store_true", help="Dataset is in HUIM format")
    parser.add_argument("--json", dest="json_out", default=None, help="Save report as JSON to this path")
    args = parser.parse_args()

    report = validate(args.dataset, args.config, args.groundtruth, args.huim)
    print_report(report)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.json_out}")


if __name__ == "__main__":
    main()
