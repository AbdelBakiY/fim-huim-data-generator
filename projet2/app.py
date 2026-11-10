#!/usr/bin/env python3
"""Flask web interface for Subject 2 — FIM/HUIM synthetic dataset generator."""
import os, sys, json, threading, subprocess, time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "generator"))

from generate import FIMGenerator, HUIMGenerator
from validate import (structural_check, statistical_check, groundtruth_check,
                      huim_utility_check, profit_distribution_check, decoupling_table,
                      load_fim, load_huim)

app = Flask(__name__)

DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR  = BASE_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"
SPMF_JAR     = BASE_DIR / "spmf.jar"
EXPERIMENTS  = BASE_DIR / "experiments"
LLM_DIR      = BASE_DIR / "llm"

_tasks: dict = {}


# ── helpers ────────────────────────────────────────────────────────────────────

def _np_clean(obj):
    """Recursively convert numpy types to plain Python for JSON."""
    import numpy as np
    if isinstance(obj, dict):   return {k: _np_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_np_clean(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj


def _bg_run(task_id: str, cmd: list, cwd: str, env: dict | None = None):
    _tasks[task_id]["status"] = "running"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=600,
                           env={**os.environ, **(env or {})})
        _tasks[task_id]["status"] = "done" if r.returncode == 0 else "error"
        _tasks[task_id]["output"] = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["output"] = "Timeout (600s)"
    except Exception as e:
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["output"] = str(e)


def _build_cfg(form) -> tuple[dict, str]:
    """Build config dict + mode from POST form."""
    mode = form.get("mode", "fim")
    cfg = {
        "num_transactions": int(form.get("num_transactions", 1000)),
        "num_items":        int(form.get("num_items", 50)),
        "avg_length":       float(form.get("avg_length", 8)),
        "density":          float(form.get("density", 0.16)),
        "noise_ratio":      float(form.get("noise_ratio", 0.15)),
    }
    gt_items    = form.getlist("gt_itemset[]")
    gt_supports = form.getlist("gt_support[]")
    patterns = []
    for items_str, sup_str in zip(gt_items, gt_supports):
        if items_str.strip():
            try:
                patterns.append({"itemset": [int(x) for x in items_str.split(",")],
                                  "target_support": float(sup_str)})
            except ValueError:
                pass
    if patterns:
        cfg["ground_truth"] = patterns

    if mode == "huim":
        cfg["huim"] = {
            "profit_distribution": "power_law",
            "profit_range":  [int(form.get("profit_min", 1)), int(form.get("profit_max", 100))],
            "quantity_distribution": "uniform",
            "quantity_range": [int(form.get("qty_min", 1)), int(form.get("qty_max", 5))],
            "min_utility_threshold": int(form.get("min_utility", 500)),
            "decouple_ratio": float(form.get("decouple_ratio", 0.30)),
        }
    return cfg, mode


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    fim_count  = len(list(DATASETS_DIR.glob("fim/**/*.dat")))  if DATASETS_DIR.exists() else 0
    huim_count = len(list(DATASETS_DIR.glob("huim/**/*.dat"))) if DATASETS_DIR.exists() else 0
    figs       = len(list(FIGURES_DIR.glob("*.png")))          if FIGURES_DIR.exists() else 0
    tabs       = len(list(TABLES_DIR.glob("*.json")))          if TABLES_DIR.exists() else 0
    return render_template("index.html",
                           fim_count=fim_count, huim_count=huim_count,
                           figure_count=figs, table_count=tabs)


@app.route("/generate", methods=["GET", "POST"])
def generate():
    result = error = None
    if request.method == "POST":
        try:
            cfg, mode = _build_cfg(request.form)
            prefix = request.form.get("prefix", "dataset").strip() or "dataset"
            seed   = int(request.form.get("seed", 42))

            if mode == "fim":
                out_dir = DATASETS_DIR / "fim" / prefix
                out_dir.mkdir(parents=True, exist_ok=True)
                gen  = FIMGenerator(cfg, seed=seed)
                txs  = gen.generate()
                dat, _ = gen.save(txs, out_dir, prefix=prefix)
                (out_dir / f"{prefix}_config.json").write_text(json.dumps(cfg, indent=2))
                result = {"type": "FIM", "prefix": prefix, "dat": str(dat),
                          "n": len(txs),
                          "avg_len": round(sum(len(t) for t in txs)/len(txs), 2),
                          "density": round(sum(len(t) for t in txs)/(len(txs)*cfg["num_items"]), 4)}
            else:
                out_dir = DATASETS_DIR / "huim" / prefix
                out_dir.mkdir(parents=True, exist_ok=True)
                gen = HUIMGenerator(cfg, seed=seed)
                txs = gen.generate_huim()
                dat, _, _ = gen.save_huim(txs, out_dir, prefix=prefix)
                (out_dir / f"{prefix}_config.json").write_text(json.dumps(cfg, indent=2))
                result = {"type": "HUIM", "prefix": prefix, "dat": str(dat),
                          "n": len(txs),
                          "avg_len": round(sum(len(items) for items, _, _ in txs)/len(txs), 2),
                          "total_util": sum(u for _, _, u in txs)}
        except Exception as e:
            error = str(e)

    return render_template("generate.html", result=result, error=error)


@app.route("/validate", methods=["GET", "POST"])
def validate():
    fim_dats  = sorted(DATASETS_DIR.glob("fim/**/*.dat"))  if DATASETS_DIR.exists() else []
    huim_dats = sorted(DATASETS_DIR.glob("huim/**/*.dat")) if DATASETS_DIR.exists() else []
    report = error = None

    if request.method == "POST":
        dat_path = Path(request.form.get("dat_path", "")).resolve()
        is_huim  = "huim" in str(dat_path).replace("\\", "/")
        try:
            if not dat_path.is_relative_to(DATASETS_DIR.resolve()) or dat_path.suffix != ".dat":
                raise ValueError("Le fichier doit être un .dat situé dans datasets/")
            if is_huim:
                transactions = load_huim(dat_path)
                raw = [list(t[0]) for t in transactions]
            else:
                raw = load_fim(dat_path)
                transactions = raw

            gt_path  = dat_path.parent / (dat_path.stem + "_groundtruth.json")
            cfg_path = dat_path.parent / (dat_path.stem + "_config.json")

            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
            else:
                n_items = max(max(t) for t in raw) + 1
                avg_l   = sum(len(t) for t in raw) / len(raw)
                cfg = {"num_transactions": len(raw), "num_items": n_items,
                       "avg_length": avg_l, "density": avg_l / n_items,
                       "noise_ratio": 0.15, "ground_truth": []}

            report = {
                "dataset": dat_path.name,
                "structural":  _np_clean(structural_check(raw, cfg)),
                "statistical": _np_clean(statistical_check(raw)),
                "groundtruth": _np_clean(groundtruth_check(transactions, gt_path)),
            }
            if is_huim:
                prof_path = dat_path.parent / (dat_path.stem + "_profits.txt")
                report["huim_utility"] = _np_clean(huim_utility_check(transactions, gt_path))
                report["profit_80_20"] = _np_clean(profit_distribution_check(gt_path))
                report["decoupling"]   = _np_clean(decoupling_table(transactions, gt_path))
        except Exception as e:
            error = str(e)

    return render_template("validate.html",
                           fim_dats=fim_dats, huim_dats=huim_dats,
                           report=report, error=error)


@app.route("/experiments")
def experiments():
    exists = lambda f: (TABLES_DIR / f).exists()
    return render_template("experiments.html",
                           ofat_fim  = exists("ofat_fim.json"),
                           ofat_huim = exists("ofat_huim.json"),
                           repro     = exists("reproducibility.json"),
                           fim_cmp   = exists("spmf_fim_comparison.json"),
                           huim_cmp  = exists("spmf_huim_comparison.json"),
                           tasks     = _tasks)


@app.route("/api/run/<script>", methods=["POST"])
def run_script(script):
    scripts = {"ofat": "ofat.py", "comparison": "spmf_runner.py", "plot": "plot.py"}
    if script not in scripts:
        return jsonify({"error": "unknown script"}), 400
    task_id = f"{script}_{int(time.time())}"
    _tasks[task_id] = {"status": "pending", "output": "", "start": time.time()}
    t = threading.Thread(target=_bg_run,
                         args=(task_id, [sys.executable, scripts[script]], str(EXPERIMENTS)),
                         daemon=True)
    t.start()
    return jsonify({"task_id": task_id})


@app.route("/api/task/<task_id>")
def task_status(task_id):
    if task_id not in _tasks:
        return jsonify({"error": "not found"}), 404
    return jsonify(_tasks[task_id])


@app.route("/results")
def results():
    figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    tables  = {}
    if TABLES_DIR.exists():
        for jf in sorted(TABLES_DIR.glob("*.json")):
            try:   tables[jf.stem] = json.loads(jf.read_text())
            except (OSError, json.JSONDecodeError): pass
    return render_template("results.html", figures=figures, tables=tables)


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/llm", methods=["GET", "POST"])
def llm():
    existing = None
    error    = None
    task_id  = None
    llm_fig  = (FIGURES_DIR / "llm_comparison.png").exists()

    if (TABLES_DIR / "llm_comparison.json").exists():
        try:
            existing = json.loads((TABLES_DIR / "llm_comparison.json").read_text())
        except Exception:
            pass

    if request.method == "POST":
        backend   = request.form.get("backend", "hf")
        model     = request.form.get("model", "meta-llama/Llama-3.3-70B-Instruct").strip()
        hf_token  = request.form.get("hf_token", "").strip()
        skip_llm  = "skip_llm" in request.form

        cmd = [sys.executable, "compare.py", "--backend", backend, "--model", model]
        # Token passed through the environment, never on the command line
        env = {"HF_TOKEN": hf_token} if backend == "hf" and hf_token else None
        if skip_llm:
            cmd += ["--skip-llm"]

        task_id = f"llm_{int(time.time())}"
        _tasks[task_id] = {"status": "pending", "output": "", "start": time.time()}
        t = threading.Thread(target=_bg_run, args=(task_id, cmd, str(LLM_DIR), env), daemon=True)
        t.start()

    return render_template("llm.html",
                           existing=existing, error=error,
                           task_id=task_id, tasks=_tasks, llm_fig=llm_fig)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=int(os.environ.get("PORT", 5001)), use_reloader=False)
