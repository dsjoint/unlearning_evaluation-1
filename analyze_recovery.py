#!/usr/bin/env python
"""analyze_recovery.py - Summarize recovery rate vs LoRA rank from existing CSVs"""

import ast
import glob
import re
import sys


def parse_rank_from_path(path: str) -> int:
    """Extract LoRA rank from model path like 'rank4-sc20-...'"""
    match = re.search(r'rank(\d+)', str(path))
    return int(match.group(1)) if match else 0


def recovery_epoch(accs_str: str, threshold: float = 0.8) -> int:
    """First epoch where accuracy >= threshold"""
    try:
        accs = ast.literal_eval(accs_str) if isinstance(accs_str, str) else accs_str
        for epoch in sorted(accs.keys(), key=lambda x: int(x)):
            if accs[epoch] >= threshold:
                return int(epoch)
        return max(int(e) for e in accs.keys())
    except Exception:
        return -1


def main() -> None:
    import pandas as pd

    # Read all FT CSVs
    ft_files = glob.glob("evals/pipeline/ft/*.csv")
    if not ft_files:
        print("No FT CSV files found in evals/pipeline/ft/")
        sys.exit(1)

    rows = []
    for path in ft_files:
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                base_model = str(row.get("base_model", ""))
                if "LORA" in base_model or "rank" in base_model:
                    rank = parse_rank_from_path(base_model)
                    if rank > 0:
                        accs = row.get("forget_accs_local", "{}")
                        rec = recovery_epoch(accs, threshold=0.8)
                        if rec >= 0:
                            rows.append({"rank": rank, "recovery_epoch": rec})
        except Exception as exc:
            print(f"Warning: Error reading {path}: {exc}")

    if not rows:
        print("No LORA experiments found in FT CSVs")
        sys.exit(1)

    result = pd.DataFrame(rows)
    summary = result.groupby("rank")["recovery_epoch"].agg(
        ["mean", "std", "count"]
    )
    print("\nRecovery Epoch by LoRA Rank:")
    print("=" * 50)
    print(summary.to_string())
    print()

    summary.to_csv("recovery_vs_rank.csv")
    print("Saved to recovery_vs_rank.csv")


if __name__ == "__main__":
    main()
