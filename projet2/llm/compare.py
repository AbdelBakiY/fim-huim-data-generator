#!/usr/bin/env python3
"""Compare LLM variants vs classical FIMGenerator — including SPMF recall.

Usage:
  python compare.py                          # Ollama + mistral (default)
  python compare.py --model llama3           # Ollama + llama3
  python compare.py --backend hf --hf-token hf_xxx  # HuggingFace
"""
import sys, json, time, argparse, subprocess, tempfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "generator"))
sys.path.insert(0, str(Path(__file__).parent))

from generate       import FIMGenerator
from validate       import structural_check, statistical_check
from llm_generator  import run_all

OUT_DIR  = BASE_DIR / "results" / "tables"
LLM_DIR  = BASE_DIR / "datasets" / "llm"
SPMF_JAR = BASE_DIR / "spmf.jar"

# Small config — keeps LLM output within token limits
CFG = {
    "num_transactions": 100,
    "num_items":        20,
    "avg_length":       5,
    "density":          0.25,
    "noise_ratio":      0.10,
    "ground_truth": [
        {"itemset": [1, 5, 3],  "target_support": 0.25},
        {"itemset": [7, 2],     "target_support": 0.20},
    ],
}


def _np_clean(obj):
    import numpy as np
    if isinstance(obj, dict):        return {k: _np_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [_np_clean(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    return obj


def run_spmf_recall(dat_path: Path, cfg: dict, theta: float = 0.05) -> dict:
    """Run FP-Growth on a dataset and compute recall against ground truth patterns."""
    if not SPMF_JAR.exists():
        return {"recall": None, "n_patterns": None, "spmf_ms": None, "error": "spmf.jar not found"}

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "fpg_out.txt"
        cmd = ["java", "-jar", str(SPMF_JAR), "run", "FPGrowth_itemsets",
               str(dat_path), str(out_path), str(theta)]
        try:
            t0 = time.time()
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            ms = int((time.time() - t0) * 1000)

            # Parse output: "3 5 1 #SUP: 25" → frozenset({1,3,5})
            found = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    if "#SUP:" in line:
                        items_part = line.split("#SUP:")[0].strip()
                        items = frozenset(int(x) for x in items_part.split())
                        found.add(items)

            # Check which ground truth patterns are in the found itemsets
            gt_patterns = cfg.get("ground_truth", [])
            hits = 0
            for p in gt_patterns:
                target = frozenset(p["itemset"])
                # Pattern is "found" if it's a subset of any mined itemset
                if any(target.issubset(f) for f in found):
                    hits += 1

            recall = round(hits / len(gt_patterns), 3) if gt_patterns else None
            return {"recall": recall, "n_patterns": len(found), "spmf_ms": ms}

        except subprocess.TimeoutExpired:
            return {"recall": None, "n_patterns": None, "spmf_ms": None, "error": "timeout"}
        except Exception as e:
            return {"recall": None, "n_patterns": None, "spmf_ms": None, "error": str(e)}


def run_classical(cfg: dict) -> dict:
    import tempfile
    t0  = time.time()
    gen = FIMGenerator(cfg, seed=42)
    txs = gen.generate()
    elapsed = time.time() - t0
    stat  = _np_clean(statistical_check(txs))
    struc = _np_clean(structural_check(txs, cfg))

    # Save to temp file and run SPMF
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        for t in txs:
            f.write(" ".join(map(str, t)) + "\n")
        tmp_dat = Path(f.name)
    spmf = run_spmf_recall(tmp_dat, cfg)
    tmp_dat.unlink(missing_ok=True)

    return {
        "n_transactions": len(txs),
        "elapsed_s":      round(elapsed, 3),
        "alpha":          stat.get("powerlaw_alpha"),
        "alpha_ok":       stat.get("alpha_in_range_1_5_to_3_0"),
        "r2":             stat.get("powerlaw_r2"),
        "skewness":       stat.get("length_skewness"),
        "right_skewed":   stat.get("length_right_skewed"),
        "avg_length_err": struc.get("avg_length", {}).get("error_pct"),
        "avg_length_ok":  struc.get("avg_length", {}).get("ok"),
        "recall":         spmf.get("recall"),
        "n_patterns":     spmf.get("n_patterns"),
        "spmf_ms":        spmf.get("spmf_ms"),
    }


def enrich_with_spmf(llm_results: dict, cfg: dict) -> dict:
    """Run FP-Growth on each saved LLM .dat file and add recall."""
    for name, r in llm_results.items():
        if "error" in r:
            continue
        dat = Path(r.get("dat_path", ""))
        if not dat.is_absolute():
            dat = BASE_DIR / dat
        if not dat.exists():
            dat = LLM_DIR / f"llm_{name}.dat"   # fallback if the repo was moved
        if not dat.exists():
            r["recall"] = None
            continue
        print(f"  SPMF [{name}] ...", end=" ", flush=True)
        spmf = run_spmf_recall(dat, cfg)
        r["recall"]     = spmf.get("recall")
        r["n_patterns"] = spmf.get("n_patterns")
        r["spmf_ms"]    = spmf.get("spmf_ms")
        print(f"recall={r['recall']}  patterns={r['n_patterns']}")
    return llm_results


def print_table(classical: dict, llm: dict):
    line = "-" * 80
    print(f"\n{'Variant':<12} | {'n_tx':>5} | {'alpha':>6} | {'a_ok':>4} | {'skew':>6} | {'recall':>6} | {'t(s)':>7}")
    print(line)

    def row(name, r):
        if "error" in r:
            return f"{name:<12} | ERREUR: {r['error'][:40]}"
        return (f"{name:<12} | {r.get('n_transactions','?'):>5} | "
                f"{str(r.get('alpha') or '?'):>6} | "
                f"{'Y' if r.get('alpha_ok') else 'N':>4} | "
                f"{str(r.get('skewness') or '?'):>6} | "
                f"{str(r.get('recall') or '?'):>6} | "
                f"{r.get('elapsed_s','?'):>7}")

    print(row("classical", classical))
    print(line)
    for name, r in llm.items():
        print(row(name, r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",  default="ollama", choices=["ollama", "hf"])
    parser.add_argument("--model",    default=None,
                        help="Ollama: mistral/llama3/gemma2 (default: mistral) | "
                             "HF: model path (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (hf_xxx)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM generation, only run SPMF on existing .dat files")
    args = parser.parse_args()

    if args.model is None:
        args.model = "mistral" if args.backend == "ollama" else "meta-llama/Llama-3.1-8B-Instruct"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Backend : {args.backend}  |  Model : {args.model}")
    print("=" * 50)

    print("\n[1/3] Classical Generator + SPMF...")
    classical = run_classical(CFG)
    print(f"  alpha={classical['alpha']}  recall={classical['recall']}  t={classical['elapsed_s']}s")

    if args.skip_llm:
        # Only re-run SPMF on existing LLM datasets
        print("\n[2/3] SPMF on existing LLM datasets...")
        existing = json.loads((OUT_DIR / "llm_comparison.json").read_text())
        llm_results = existing.get("llm", {})
    else:
        print("\n[2/3] LLM Variants (4 prompts)...")
        llm_results = run_all(CFG, LLM_DIR,
                              backend=args.backend,
                              model=args.model,
                              token=args.hf_token)

    print("\n[3/3] SPMF recall on LLM datasets...")
    llm_results = enrich_with_spmf(llm_results, CFG)

    comparison = {
        "backend":   args.backend,
        "model":     args.model,
        "config":    CFG,
        "classical": classical,
        "llm":       _np_clean(llm_results),
    }
    out = OUT_DIR / "llm_comparison.json"
    out.write_text(json.dumps(comparison, indent=2, default=str))
    print(f"\nSaved -> {out}")

    print_table(classical, llm_results)
    print(f"\nPour générer la figure : python plot_llm.py")
