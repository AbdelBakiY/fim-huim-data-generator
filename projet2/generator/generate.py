#!/usr/bin/env python3
import json
import math
import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np


def powerlaw_weights(n: int, alpha: float = 2.0) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=float)
    w = ranks ** (-alpha)
    return w / w.sum()


def lognormal_length(avg: float, rng: np.random.Generator) -> int:
    sigma = 0.6
    mu = math.log(max(avg, 1.0)) - 0.5 * sigma ** 2
    return max(1, int(round(rng.lognormal(mu, sigma))))


class FIMGenerator:
    def __init__(self, config: dict, seed: Optional[int] = None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self._rnd = random.Random(seed)

        self.n = config["num_transactions"]
        self.n_items = config["num_items"]
        self.avg_len = config["avg_length"]
        self.noise_ratio = config.get("noise_ratio", 0.1)
        self.patterns = config.get("ground_truth", [])
        self.items = np.arange(1, self.n_items + 1)
        self.item_probs = powerlaw_weights(self.n_items)

    def _fill_transaction(self, base: set) -> list:
        target_len = lognormal_length(self.avg_len, self.rng)
        need = max(0, target_len - len(base))
        if need > 0:
            mask = np.ones(self.n_items, dtype=bool)
            for i in base:
                mask[i - 1] = False
            pool = self.items[mask]
            probs = self.item_probs[mask]
            probs = probs / probs.sum()
            k = min(need, len(pool))
            extra = self.rng.choice(pool, size=k, replace=False, p=probs)
            base.update(extra.tolist())
        return sorted(base)

    def _build_transactions(self) -> list:
        transactions = []
        for _ in range(self.n):
            base = set()
            for p in self.patterns:
                if self._rnd.random() < p["target_support"] * (1 - self.noise_ratio):
                    base.update(p["itemset"])
            transactions.append(self._fill_transaction(base))
        self._rnd.shuffle(transactions)

        # Guarantee every item appears at least once (fixes vocabulary coverage)
        seen = set(i for t in transactions for i in t)
        missing = [int(i) for i in self.items if i not in seen]
        for item in missing:
            idx = self._rnd.randint(0, len(transactions) - 1)
            t = set(transactions[idx])
            t.add(item)
            transactions[idx] = sorted(t)

        return transactions

    def generate(self) -> list:
        return self._build_transactions()

    def measure_support(self, transactions: list, itemset: list) -> float:
        s = set(itemset)
        return sum(1 for t in transactions if s.issubset(t)) / len(transactions)

    def save(self, transactions: list, out_dir: Path, prefix: str = "dataset"):
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = out_dir / f"{prefix}.dat"
        gt_path = out_dir / f"{prefix}_groundtruth.json"

        with open(dataset_path, "w") as f:
            for t in transactions:
                f.write(" ".join(map(str, t)) + "\n")

        gt = []
        for p in self.patterns:
            measured = self.measure_support(transactions, p["itemset"])
            gt.append({
                "itemset": p["itemset"],
                "target_support": p["target_support"],
                "measured_support": round(measured, 4),
            })

        with open(gt_path, "w") as f:
            json.dump({"format": "fim", "patterns": gt}, f, indent=2)

        print(f"FIM dataset -> {dataset_path}")
        print(f"Ground truth -> {gt_path}")
        return dataset_path, gt_path


class HUIMGenerator(FIMGenerator):
    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.hcfg = config.get("huim", {})
        p_range = self.hcfg.get("profit_range", [1, 100])
        self.profits = self._build_profits(p_range)
        q_range = self.hcfg.get("quantity_range", [1, 5])
        self.qty_min, self.qty_max = q_range

    def _build_profits(self, p_range: list) -> dict:
        lo, hi = p_range
        w = powerlaw_weights(self.n_items, alpha=1.8)
        sorted_w = np.sort(w)[::-1]
        scaled = lo + (hi - lo) * (sorted_w / sorted_w[0])
        scaled = np.clip(np.round(scaled).astype(int), lo, hi)
        return {int(self.items[i]): int(scaled[i]) for i in range(self.n_items)}

    def _qty(self) -> int:
        dist = self.hcfg.get("quantity_distribution", "uniform")
        if dist == "geometric":
            v = int(self.rng.geometric(0.5))
            return int(np.clip(v, self.qty_min, self.qty_max))
        return self._rnd.randint(self.qty_min, self.qty_max)

    def generate_huim(self) -> list:
        transactions = self._build_transactions()
        result = []
        for t in transactions:
            qtys = {item: self._qty() for item in t}
            tx_util = sum(qtys[i] * self.profits[i] for i in t)
            result.append((t, qtys, tx_util))
        return result

    def save_huim(self, transactions: list, out_dir: Path, prefix: str = "dataset"):
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = out_dir / f"{prefix}.dat"
        profit_path = out_dir / f"{prefix}_profits.txt"
        gt_path = out_dir / f"{prefix}_groundtruth.json"

        with open(profit_path, "w") as f:
            line = " ".join(str(self.profits[i]) for i in range(1, self.n_items + 1))
            f.write(line + "\n")

        with open(dataset_path, "w") as f:
            for items, qtys, tx_util in transactions:
                items_str = " ".join(map(str, items))
                qtys_str = " ".join(str(qtys[i]) for i in items)
                f.write(f"{items_str}:{tx_util}:{qtys_str}\n")

        gt_huis = []
        for p in self.hcfg.get("ground_truth_huis", []):
            s = set(p["itemset"])
            util = sum(
                sum(qtys[i] * self.profits[i] for i in items if i in s)
                for items, qtys, _ in transactions
                if s.issubset(items)
            )
            gt_huis.append({
                "itemset": p["itemset"],
                "target_utility": p.get("target_utility"),
                "measured_utility": int(util),
            })

        # Also measure FIM patterns so groundtruth_check works on HUIM datasets
        flat = [items for items, _, _ in transactions]
        fim_patterns = []
        for p in self.patterns:
            measured = self.measure_support(flat, p["itemset"])
            fim_patterns.append({
                "itemset": p["itemset"],
                "target_support": p["target_support"],
                "measured_support": round(measured, 4),
            })

        with open(gt_path, "w") as f:
            json.dump({
                "format": "huim",
                "patterns": fim_patterns,
                "huis": gt_huis,
                "profit_table": {str(k): v for k, v in self.profits.items()},
            }, f, indent=2)

        print(f"HUIM dataset -> {dataset_path}")
        print(f"Profit table -> {profit_path}")
        print(f"Ground truth -> {gt_path}")
        return dataset_path, profit_path, gt_path


def run(config_path: str, output_dir: str, seed: Optional[int], prefix: str):
    with open(config_path) as f:
        config = json.load(f)

    out = Path(output_dir)
    is_huim = "huim" in config

    if is_huim:
        gen = HUIMGenerator(config, seed=seed)
        transactions = gen.generate_huim()
        gen.save_huim(transactions, out, prefix=prefix)
    else:
        gen = FIMGenerator(config, seed=seed)
        transactions = gen.generate()
        gen.save(transactions, out, prefix=prefix)

    n_actual = len(transactions)
    total_items = sum(len(t[0]) if isinstance(t, tuple) else len(t) for t in transactions)
    print(f"\nStats: |D|={n_actual}, avg_length={total_items/n_actual:.2f}, "
          f"density={total_items/(n_actual*config['num_items']):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic transactional dataset generator")
    parser.add_argument("config", help="Path to JSON configuration file")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("-p", "--prefix", default="dataset", help="Output file prefix")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    run(args.config, args.output, args.seed, args.prefix)


if __name__ == "__main__":
    main()
