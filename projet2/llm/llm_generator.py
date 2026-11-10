#!/usr/bin/env python3
"""LLM-based dataset generator — 4 prompt strategies.
Supports: Ollama (local, free) or HuggingFace (free token).

Usage:
  # Ollama (recommended - 100% free, no API key)
  ollama pull mistral   # or llama3, gemma2, etc.
  python compare.py --backend ollama --model mistral

  # HuggingFace (free token at huggingface.co)
  python compare.py --backend hf --hf-token hf_xxx
"""
import sys, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "generator"))
sys.path.insert(0, str(Path(__file__).parent))

from prompts import (minimal_prompt, narrative_prompt, fewshot_prompt,
                     iterative_prompt_step1, iterative_prompt_step2)


# ── Backend adapters ────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str = "mistral") -> str:
    import ollama
    r = ollama.generate(model=model, prompt=prompt)
    return r["response"]


def _call_hf(prompt: str, model: str = "meta-llama/Llama-3.1-8B-Instruct",
             token: str = None) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=model, token=token)
    r = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, temperature=0.7
    )
    return r.choices[0].message.content


def _make_caller(backend: str, model: str, token: str = None):
    """Return a (prompt -> text) function for the chosen backend."""
    if backend == "ollama":
        # Test connection
        try:
            import ollama
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                f"Ollama not running. Start it with: ollama serve\n"
                f"Then pull a model: ollama pull mistral\n({e})"
            )
        return lambda p: _call_ollama(p, model)

    elif backend == "hf":
        if not token:
            import os
            token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required.\n"
                "Get a free token at https://huggingface.co/settings/tokens\n"
                "Then: set HF_TOKEN=hf_xxx"
            )
        return lambda p: _call_hf(p, model, token)

    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'ollama' or 'hf'.")


# ── Response parser ─────────────────────────────────────────────────────────────

def _parse(text: str, max_item: int) -> list:
    """Extract transactions from LLM response — robust to noise."""
    transactions = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("{") or line.startswith("[") or line.startswith("#"):
            continue
        # Remove leading numbering like "1." or "- "
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        tokens = re.findall(r"\b\d+\b", line)
        if len(tokens) < 2:
            continue
        items = list(dict.fromkeys(
            int(t) for t in tokens if 1 <= int(t) <= max_item
        ))
        if items:
            transactions.append(items)
    return transactions


# ── 4 variants ──────────────────────────────────────────────────────────────────

def variant_minimal(cfg: dict, call) -> tuple[list, float, str]:
    prompt = minimal_prompt(cfg)
    t0 = time.time()
    text = call(prompt)
    return _parse(text, cfg["num_items"]), time.time() - t0, prompt


def variant_narrative(cfg: dict, call) -> tuple[list, float, str]:
    prompt = narrative_prompt(cfg)
    t0 = time.time()
    text = call(prompt)
    return _parse(text, cfg["num_items"]), time.time() - t0, prompt


def variant_fewshot(cfg: dict, call) -> tuple[list, float, str]:
    prompt = fewshot_prompt(cfg)
    t0 = time.time()
    text = call(prompt)
    return _parse(text, cfg["num_items"]), time.time() - t0, prompt


def variant_iterative(cfg: dict, call) -> tuple[list, float, str]:
    """Two-turn: first get frequency distribution, then generate transactions."""
    p1 = iterative_prompt_step1(cfg)
    t0 = time.time()
    freq_text = call(p1)                          # Step 1
    p2 = iterative_prompt_step2(cfg, freq_text)
    text = call(p2)                               # Step 2
    return _parse(text, cfg["num_items"]), time.time() - t0, p1 + "\n---\n" + p2


VARIANTS = {
    "minimal":   variant_minimal,
    "narrative": variant_narrative,
    "fewshot":   variant_fewshot,
    "iterative": variant_iterative,
}


def _rel(path: Path) -> str:
    """Path relative to projet2/ (portable across machines), absolute otherwise."""
    base = Path(__file__).resolve().parent.parent
    try:
        return Path(path).resolve().relative_to(base).as_posix()
    except ValueError:
        return str(path)


# ── Main runner ─────────────────────────────────────────────────────────────────

def run_all(cfg: dict, out_dir: Path,
            backend: str = "ollama",
            model: str = "mistral",
            token: str = None) -> dict:

    from validate import structural_check, statistical_check

    call = _make_caller(backend, model, token)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, fn in VARIANTS.items():
        print(f"  [{name}] ...", end=" ", flush=True)
        try:
            txs, elapsed, prompt_used = fn(cfg, call)

            dat = out_dir / f"llm_{name}.dat"
            dat.write_text("\n".join(" ".join(map(str, t)) for t in txs))
            (out_dir / f"llm_{name}_prompt.txt").write_text(prompt_used)

            stat  = statistical_check(txs) if txs else {}
            struc = structural_check(txs, cfg) if txs else {}

            results[name] = {
                "n_transactions": len(txs),
                "elapsed_s":      round(elapsed, 2),
                "alpha":          stat.get("powerlaw_alpha"),
                "alpha_ok":       stat.get("alpha_in_range_1_5_to_3_0"),
                "r2":             stat.get("powerlaw_r2"),
                "skewness":       stat.get("length_skewness"),
                "right_skewed":   stat.get("length_right_skewed"),
                "avg_length_err": struc.get("avg_length", {}).get("error_pct"),
                "avg_length_ok":  struc.get("avg_length", {}).get("ok"),
                "dat_path":       _rel(dat),
            }
            print(f"OK  n={len(txs)}  alpha={stat.get('powerlaw_alpha')}  t={elapsed:.1f}s")

        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"ERROR: {e}")

    return results
