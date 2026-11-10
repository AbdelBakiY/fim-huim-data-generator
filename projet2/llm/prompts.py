"""4 prompt variants for LLM-based transactional dataset generation."""


def minimal_prompt(cfg: dict) -> str:
    """Variant 1 — Minimal: short, direct instructions only."""
    n  = cfg["num_transactions"]
    ni = cfg["num_items"]
    al = cfg.get("avg_length", 8)
    return (
        f"Generate {n} transactions in SPMF format.\n"
        f"Items: integers from 1 to {ni}.\n"
        f"Average transaction length: {al}.\n"
        f"Output: one transaction per line, space-separated item IDs only. No other text."
    )


def narrative_prompt(cfg: dict) -> str:
    """Variant 2 — Narrative: rich context + statistical constraints."""
    n       = cfg["num_transactions"]
    ni      = cfg["num_items"]
    al      = cfg.get("avg_length", 8)
    density = cfg.get("density", round(al / ni, 3))
    noise   = cfg.get("noise_ratio", 0.15)
    gt      = cfg.get("ground_truth", [])

    gt_text = ""
    if gt:
        patterns = "; ".join(
            f"items [{','.join(map(str, p['itemset']))}] in ~{int(p['target_support']*100)}% of transactions"
            for p in gt
        )
        gt_text = f"\nEmbed these frequent patterns: {patterns}."

    return f"""You are a data scientist generating a synthetic transactional dataset for frequent itemset mining research.

Requirements:
- {n} transactions total
- {ni} unique items (IDs 1 to {ni})
- Average transaction length: {al} items
- Density: ~{density} (average items per transaction / total items)
- Item popularity must follow a power-law distribution: a few items appear very frequently (items 1-5), most appear rarely (items {ni-5}-{ni})
- Transaction lengths must follow a log-normal distribution: most transactions are short (2-6 items), a few are long (15+ items)
- Noise ratio: ~{noise} (some randomness in which items co-occur){gt_text}

Output format: one transaction per line, space-separated integer item IDs.
Output ONLY the transactions — no explanations, no headers, no numbering.

Example:
1 5 3 7
2 8 15 4 1 9
4 3 7 9 11 15 2 1 5 8
"""


def fewshot_prompt(cfg: dict, examples: list = None) -> str:
    """Variant 3 — Few-shot: show examples, ask to continue same pattern."""
    n  = cfg["num_transactions"]
    ni = cfg["num_items"]
    al = cfg.get("avg_length", 8)

    if examples is None:
        examples = [
            [1, 5, 3, 7],
            [2, 8],
            [4, 3, 7, 9, 11, 15, 2, 1, 5, 8],
            [1, 3, 7, 5, 2, 8, 9],
            [5, 2, 1],
            [1, 7, 3, 5, 9, 2, 8, 11, 4, 15, 6],
            [3, 1, 5],
            [2, 7, 1, 5, 3, 8],
        ]

    ex_str = "\n".join(" ".join(map(str, t)) for t in examples)

    return f"""Here are example transactions from a transactional dataset (items 1 to {ni}):

{ex_str}

Notice the pattern:
- Low-numbered items (1, 3, 5, 7) appear very frequently
- High-numbered items appear rarely
- Most transactions are short, a few are long

Generate {n} more transactions following exactly the same statistical pattern.
Use item IDs 1 to {ni} only. Average length should be around {al}.
Output only the transactions, one per line, space-separated integers. No other text.
"""


def iterative_prompt_step1(cfg: dict) -> str:
    """Variant 4 — Iterative step 1: ask LLM to define frequency distribution."""
    ni = cfg["num_items"]
    return (
        f"You are building a transactional dataset with {ni} items (IDs 1 to {ni}).\n\n"
        f"Step 1: Define a power-law frequency distribution for the {ni} items.\n"
        f"Item 1 should be the most frequent (~25% of transactions), "
        f"item {ni} the least frequent (~1%).\n\n"
        f"Output ONLY a JSON object mapping item IDs to probabilities that sum to 1.0.\n"
        f"Example format: {{\"1\": 0.12, \"2\": 0.09, ..., \"{ni}\": 0.01}}"
    )


def iterative_prompt_step2(cfg: dict, freq_dist_text: str) -> str:
    """Variant 4 — Iterative step 2: generate transactions using the distribution."""
    n  = cfg["num_transactions"]
    al = cfg.get("avg_length", 8)
    return (
        f"Using this item frequency distribution:\n{freq_dist_text}\n\n"
        f"Step 2: Generate {n} transactions.\n"
        f"- Sample transaction length from a log-normal distribution (mean={al})\n"
        f"- For each transaction, sample items without replacement according to their frequency\n\n"
        f"Output ONLY the transactions, one per line, space-separated item IDs. No other text."
    )
