#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_run_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _to_int_if_possible(x: Any) -> Any:
    try:
        return int(x)
    except Exception:
        return x


def _normalize_acc_to_pct(acc: Optional[float]) -> Optional[float]:
    if acc is None:
        return None
    if acc <= 1.0:
        return float(acc) * 100.0
    return float(acc)


def extract_accuracy(accs: Any, selection_rule: str) -> Optional[float]:
    if not accs or not isinstance(accs, dict):
        return None

    # Nested: {file: {epoch: acc}}
    first_val = next(iter(accs.values()), None)
    if isinstance(first_val, dict):
        per_file = []
        for _, epochs_dict in accs.items():
            if not epochs_dict:
                continue
            keys = list(epochs_dict.keys())
            keys_int = [_to_int_if_possible(k) for k in keys]
            # Map back to original key for lookup
            key_pairs = list(zip(keys_int, keys))
            if selection_rule == "final_epoch":
                selected_key = max(key_pairs, key=lambda p: p[0])[1]
                val = epochs_dict.get(selected_key)
            elif selection_rule == "max_epoch":
                val = max(epochs_dict.values())
            else:
                val = epochs_dict.get(selection_rule)
            if val is not None:
                per_file.append(float(val))
        if not per_file:
            return None
        return _normalize_acc_to_pct(float(np.mean(per_file)))

    # Flat: {epoch: acc}
    keys = list(accs.keys())
    keys_int = [_to_int_if_possible(k) for k in keys]
    key_pairs = list(zip(keys_int, keys))
    if selection_rule == "final_epoch":
        selected_key = max(key_pairs, key=lambda p: p[0])[1]
        val = accs.get(selected_key)
    elif selection_rule == "max_epoch":
        val = max(accs.values())
    else:
        val = accs.get(selection_rule)
    return _normalize_acc_to_pct(float(val)) if val is not None else None


def _pick_best_b(bs: List[Dict[str, Any]], selection_rule: str) -> Optional[Dict[str, Any]]:
    if not bs:
        return None
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for b in bs:
        fa = extract_accuracy(b.get("forget_accs"), selection_rule)
        scored.append(((fa if fa is not None else -1.0), b))
    return max(scored, key=lambda t: t[0])[1]


def load_raw_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_long_df(
    run_name: str,
    raw: Dict[str, Any],
    selection_rule: str,
) -> pd.DataFrame:
    a_rows = raw.get("A", []) or []
    b_rows = raw.get("B", []) or []

    # Index A by lora_rank
    a_by_rank: Dict[int, Dict[str, Any]] = {}
    for a in a_rows:
        if not isinstance(a, dict):
            continue
        rank = a.get("lora_rank")
        if rank is None:
            continue
        a_by_rank[int(rank)] = a

    # Group B by lora_rank (and pick best if multiple)
    b_by_rank: Dict[int, Dict[str, Any]] = {}
    tmp: Dict[int, List[Dict[str, Any]]] = {}
    for b in b_rows:
        if not isinstance(b, dict):
            continue
        rank = b.get("lora_rank")
        if rank is None:
            continue
        tmp.setdefault(int(rank), []).append(b)
    for rank, items in tmp.items():
        best = _pick_best_b(items, selection_rule)
        if best is not None:
            b_by_rank[rank] = best

    rows: List[Dict[str, Any]] = []
    for rank, a in sorted(a_by_rank.items()):
        b = b_by_rank.get(rank)
        row = {
            "run_name": run_name,
            "data_seed": a.get("data_seed"),
            "eval_split_ids": json.dumps(a.get("eval_split_ids")) if a.get("eval_split_ids") is not None else None,
            "dataset": a.get("dataset"),
            "unlearn_type": a.get("method"),
            "lora_rank": rank,
            "retain_coeff": a.get("retain_coeff"),
            "unlearn_lr": a.get("lr"),
            "unlearn_epochs": a.get("epochs"),
            "ft_skip_split": (b.get("skip_split") if isinstance(b, dict) else None),
            "ft_lr": (b.get("lr") if isinstance(b, dict) else None),
            "ft_epochs": (b.get("epochs") if isinstance(b, dict) else None),
            "ft_loss_type": (b.get("loss_type") if isinstance(b, dict) else None),
            "forget_acc_unlearn": extract_accuracy(a.get("forget_accs"), selection_rule),
            "retain_acc_unlearn": extract_accuracy(a.get("retain_accs"), selection_rule),
            "forget_acc_unlearn_rtt": extract_accuracy(b.get("forget_accs"), selection_rule) if b else None,
            "retain_acc_unlearn_rtt": extract_accuracy(b.get("retain_accs"), selection_rule) if b else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="evals/pipeline/summary")
    ap.add_argument("--run-name-prefix", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--selection-rule", default="final_epoch", choices=["final_epoch", "max_epoch"])
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    out_prefix = args.out_prefix or _safe_run_name(args.run_name_prefix)

    dfs = []
    for seed in seeds:
        run_name = f"{args.run_name_prefix}_seed{seed}"
        path = os.path.join(args.results_dir, f"raw_results_{_safe_run_name(run_name)}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raw results: {path}")
        raw = load_raw_results(path)
        dfs.append(build_long_df(run_name, raw, args.selection_rule))

    long_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if long_df.empty:
        raise RuntimeError("No rows found in raw results.")

    out_long = os.path.join(args.results_dir, f"seed_sweep_long_{out_prefix}.csv")
    long_df.to_csv(out_long, index=False)

    agg = (
        long_df.groupby(["dataset", "unlearn_type", "lora_rank"], dropna=False)
        .agg(
            n=("run_name", "count"),
            forget_acc_unlearn_mean=("forget_acc_unlearn", "mean"),
            forget_acc_unlearn_std=("forget_acc_unlearn", "std"),
            retain_acc_unlearn_mean=("retain_acc_unlearn", "mean"),
            retain_acc_unlearn_std=("retain_acc_unlearn", "std"),
            forget_acc_unlearn_rtt_mean=("forget_acc_unlearn_rtt", "mean"),
            forget_acc_unlearn_rtt_std=("forget_acc_unlearn_rtt", "std"),
            retain_acc_unlearn_rtt_mean=("retain_acc_unlearn_rtt", "mean"),
            retain_acc_unlearn_rtt_std=("retain_acc_unlearn_rtt", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "unlearn_type", "lora_rank"])
    )

    out_agg = os.path.join(args.results_dir, f"seed_sweep_agg_{out_prefix}.csv")
    agg.to_csv(out_agg, index=False)

    print(f"Wrote: {out_long}")
    print(f"Wrote: {out_agg}")

    if args.plot:
        import matplotlib.pyplot as plt

        ds = str(long_df["dataset"].iloc[0])
        ut = str(long_df["unlearn_type"].iloc[0])
        title = f"{ds} {ut} rank sweep ({len(seeds)} seeds)"

        plot_df = agg.copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        # Forget
        ax = axes[0]
        ax.errorbar(
            plot_df["lora_rank"],
            plot_df["forget_acc_unlearn_mean"],
            yerr=plot_df["forget_acc_unlearn_std"],
            marker="o",
            label="Unlearn",
        )
        ax.errorbar(
            plot_df["lora_rank"],
            plot_df["forget_acc_unlearn_rtt_mean"],
            yerr=plot_df["forget_acc_unlearn_rtt_std"],
            marker="o",
            label="Unlearn+RTT",
        )
        ax.set_title("Forget Accuracy (%)")
        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Retain
        ax = axes[1]
        ax.errorbar(
            plot_df["lora_rank"],
            plot_df["retain_acc_unlearn_mean"],
            yerr=plot_df["retain_acc_unlearn_std"],
            marker="o",
            label="Unlearn",
        )
        ax.errorbar(
            plot_df["lora_rank"],
            plot_df["retain_acc_unlearn_rtt_mean"],
            yerr=plot_df["retain_acc_unlearn_rtt_std"],
            marker="o",
            label="Unlearn+RTT",
        )
        ax.set_title("Retain Accuracy (%)")
        ax.set_xlabel("LoRA Rank")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.suptitle(title)
        fig.tight_layout()

        out_png = os.path.join(args.results_dir, f"seed_sweep_plot_{out_prefix}.png")
        fig.savefig(out_png, dpi=200)
        print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()

