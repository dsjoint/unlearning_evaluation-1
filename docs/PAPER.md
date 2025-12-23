# Paper Protocol

**Purpose**: Documentation of the paper's experimental protocol and current implementation status.  
**Audience**: Researchers understanding the experimental design, developers implementing protocol changes.  
**Canonical for**: A/B/C conditions, RTT protocol, implementation specifications.

---

## Paper Protocol (Minimal)

**Conditions:**

- **A) Unlearn:** Forget set accuracy after unlearning (no RTT).
- **B) Unlearn+RTT:** Forget set accuracy after unlearning followed by retraining-to-threshold (RTT).
- **C) Baseline+RTT:** Forget set accuracy after RTT starting from the base model (no unlearning).

**RTT Protocol (arXiv:2410.08827):**

- **T (Training Set):** All splits except the held-out validation split V.
- **V (Validation Set):** One held-out split used for validation.
- **V Runs:** Multiple independent runs, each with a different split as V.
- **LR/epochs sweep:** For each V run, sweep learning rates and epochs; select the best (LR, epochs) combination based on V accuracy.
- **Aggregation:** For each V run, record the best V accuracy (from best LR/epochs); average these best accuracies across all V runs.

**Ambiguity note:** The paper does not explicitly state whether "best LR" is selected per V run independently or globally. We interpret it as per-V-run selection (select best LR/epochs for each V, then average), which is more conservative and matches standard cross-validation practice.

## Current Repo Mapping (As-Is)

**A) Unlearn-only artifacts:**

- **CSV path pattern:** `evals/pipeline/unlearning/{timestamp}--num{i}.csv`
- **Field names:** `forget_accs` (nested dict: `{filepath: {epoch: accuracy}}` when multiple files evaluated), `val_files` (list of all split files)
- **Code location:** `pipeline.py:1105` (`write_metrics_to_csv()`)
- **Evaluation:** Evaluates on ALL `val_files` (all splits), not just eval_split_ids. See `pipeline.py:947` where `val_files=val_files` (full list) is passed to `unlearn_corpus.just_eval.remote()` or `unlearn.remote()`.

**B) Unlearn+RTT artifacts:**

- **CSV path pattern:** `evals/pipeline/ft/{timestamp}--num{i}.csv`
- **Field names:** `forget_accs_local` (dict: `{epoch: accuracy}` for a single split), `rtt_eval_split_id` (int), `rtt_train_split_ids` (list), `lr`, `epochs`, `rtt_signature`
- **Code location:** `pipeline.py:1297` (`write_metrics_to_csv()`)
- **V selection:** Uses `eval_split_ids` from config (`ft.eval_split_ids`) or randomly sampled (`ft.num_splits`). See `pipeline.py:1949-1973` (`resolve_eval_split_ids()`).
- **RTT execution:** For each `skip_split` in `eval_split_ids`, trains on all other splits, validates on `skip_split`. Sweeps over `ft.lrs` and `ft.epochs_lst` for each split. See `pipeline.py:1122-1184`.
- **Best selection:** `select_best_rtt_mean()` (`pipeline.py:293-324`) groups results by `(loss_type, lr, epochs)` across all V runs simultaneously, computes mean accuracy across all eval_split_ids for each group, then selects the group with highest mean. **This differs from paper protocol:** it selects best (LR, epochs) globally across all V runs, not per-V-run.

**C) Baseline+RTT artifacts:**

- **CSV path pattern:** `evals/pipeline/ft/{timestamp}--num{i}.csv` (same as B)
- **Field names:** Same as B, with `rtt_condition: "baseline+rtt"` and `rtt_start_model_path` pointing to base model.
- **Code location:** `pipeline.py:2414-2465` (processing baseline RTT results)
- **V selection:** Same `eval_split_ids` as B (computed once per dataset). See `pipeline.py:2141`.
- **RTT execution:** Same as B, but starts from base model. See `pipeline.py:2155-2217`.
- **Best selection:** Same `select_best_rtt_mean()` logic as B.

**Baseline evaluation (pre-unlearning check):**

- **Code location:** `pipeline.py:569` (`evaluate_baseline_model()`)
- **Evaluation:** Evaluates on ALL `val_files` (all splits), not filtered to eval_split_ids. See `pipeline.py:630` where all `val_files` are passed.

**Summary CSV:**

- **CSV path pattern:** `evals/pipeline/summary/{timestamp}.csv`
- **Field names:** `forget_acc_unlearn` (A), `forget_acc_unlearn_rtt` (B), `forget_acc_baseline_rtt` (C), `forget_acc_baseline` (pre-unlearning), `rtt_signature`, `num_rtt_splits_b`, `num_rtt_splits_c`
- **Code location:** `pipeline.py:142` (`write_summary_csv()`)
- **A extraction:** Extracts per-split accuracies from A's nested dict structure matching `held_out_split_ids` (from B/C) and averages them. A was evaluated on all splits, so this attempts to align with eval_split_ids used in B/C. See `pipeline.py:387-444`.

**Current V selection logic:**

- `eval_split_ids` are determined once per dataset via `resolve_eval_split_ids()` (`pipeline.py:1949-1973`):
  - If `ft.eval_split_ids` is set in config, use those (clipped to available splits).
  - Otherwise, randomly sample `min(ft.num_splits, num_total_splits)` splits using seed `ft.eval_seed` (defaults to `data_seed`).
- Same `eval_split_ids` are used for both B (unlearn+RTT) and C (baseline+RTT) for a given dataset.
- A (unlearn-only) and baseline evaluation do NOT use the same eval_split_ids; they evaluate on all splits.

## Spec (Target Behavior)

**Dataset split terminology:**

- **Total dataset splits:** All available splits (e.g., `split_0.jsonl` through `split_4.jsonl` = 5 total splits).
- **Evaluation splits (V runs):** Subset of total splits designated as validation splits. Each V run uses one split as V, all others as T.

**RTT protocol implementation:**

For each `eval_split_id` in `eval_split_ids`:

1. **Training:** Train on all splits except `eval_split_id` (T = all splits \ {eval_split_id}).
2. **Validation:** Validate on `eval_split_id` (V = {eval_split_id}).
3. **Hyperparameter sweep:** Sweep over all combinations of `ft.lrs` and `ft.epochs_lst`.
4. **Best selection per V run:** For this `eval_split_id`, select the (LR, epochs) combination that achieves the highest validation accuracy on V. Record this best accuracy.
5. **Aggregation:** After processing all `eval_split_ids`, average the best accuracies from step 4 across all V runs.

**Consistency requirement:**

- A (unlearn-only) must be evaluated on the same `eval_split_ids` used for RTT (B and C). Specifically, for each `eval_split_id`, evaluate A on that split only, then average across all eval_split_ids.
- Baseline evaluation (pre-unlearning check) must also use the same `eval_split_ids` for consistency, though this is primarily for reporting (baseline check can still use all splits for filtering).

**Implementation changes needed:**

1. **Modify `select_best_rtt_mean()` (`pipeline.py:293-324`):** Change from global best (LR, epochs) selection to per-V-run best selection, then average.
2. **Modify unlearn-only evaluation (`pipeline.py:940-1107`):** Instead of evaluating on all `val_files`, evaluate only on splits corresponding to `eval_split_ids`, compute per-split accuracy, then average.
3. **Modify baseline evaluation (`pipeline.py:569`):** Optionally filter to `eval_split_ids` for consistency (or keep all-splits for filtering, but report eval_split_ids subset).
4. **Update summary CSV generation (`pipeline.py:142`):** Ensure A is computed as average over eval_split_ids, matching B and C aggregation.

