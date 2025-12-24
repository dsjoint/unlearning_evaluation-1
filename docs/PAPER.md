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

- **Results:** Returned as dictionaries from `main()` remote function (`pipeline.py:510`)
- **Field names:** `forget_accs` (dict: `{epoch: accuracy}`), `retain_accs`, `forget_logits_dict`, etc.
- **Code location:** Results returned from `unlearn()` remote function (`pipeline.py:314`) which calls `unlearn_corpus.main()`
- **Evaluation:** Evaluates on ALL `val_files` (all splits), not just eval_split_ids. See `pipeline.py:1544` where `val_files=dataset_dict["val_files"]` (full list) is passed to `main.remote()`.

**B) Unlearn+RTT artifacts:**

- **Results:** Returned as dictionaries from `finetune_corpus.main()` remote function
- **Field names:** Results include accuracy metrics per epoch, split information, hyperparameters
- **Code location:** Results returned from `finetune_corpus.main()` called within `main()` remote function (`pipeline.py:510`)
- **V selection:** Uses `eval_split_ids` from config (`ft.eval_split_ids`) or randomly sampled (`ft.num_splits`). See `pipeline.py:1426-1450` (`resolve_eval_split_ids()`).
- **RTT execution:** For each `skip_split` in `eval_split_ids`, trains on all other splits, validates on `skip_split`. Sweeps over `ft.lrs` and `ft.epochs_lst` for each split. See `pipeline.py:510` and `finetune_corpus.py` for implementation.
- **Best selection:** Currently selects best (LR, epochs) globally across all V runs, not per-V-run. **This differs from paper protocol:** should select best per-V-run, then average.

**C) Baseline+RTT artifacts:**

- **Results:** Returned as dictionaries from `finetune_corpus.main()` remote function (baseline RTT runs)
- **Field names:** Same structure as B, starting from base model instead of unlearned model
- **Code location:** Baseline RTT runs scheduled in `run_pipeline()` (`pipeline.py:1610-1871`), results returned from `finetune_corpus.main()`
- **V selection:** Same `eval_split_ids` as B (computed once per dataset). See `pipeline.py:1453-1464`.
- **RTT execution:** Same as B, but starts from base model. See baseline RTT scheduling in `run_pipeline()`.
- **Best selection:** Same global selection logic as B (differs from paper protocol).

**Baseline evaluation (pre-unlearning check):**

- **Code location:** `pipeline.py:212` (`evaluate_baseline_model()`)
- **Evaluation:** Evaluates on ALL `val_files` (all splits), not filtered to eval_split_ids. See `pipeline.py:1377-1406` where baseline evaluation is performed.

**Summary/Results:**

- **Results:** Metrics are returned as dictionaries from Ray remote functions. Users can process these results to create summary statistics.
- **Field names:** Results include `forget_accs`, `retain_accs`, and other metrics from A, B, and C conditions
- **Code location:** Results collected via `ray.get()` calls in `run_pipeline()` (`pipeline.py:1847-1894`)
- **A extraction:** Currently A evaluates on all splits. To align with B/C, A should be evaluated only on `eval_split_ids` and averaged.

**Current V selection logic:**

- `eval_split_ids` are determined once per dataset via `resolve_eval_split_ids()` (`pipeline.py:1426-1450`):
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

1. **Modify best selection logic:** Change from global best (LR, epochs) selection to per-V-run best selection, then average. This logic is currently in `finetune_corpus.py` or needs to be implemented.
2. **Modify unlearn-only evaluation (`pipeline.py:510`):** Instead of evaluating on all `val_files`, evaluate only on splits corresponding to `eval_split_ids`, compute per-split accuracy, then average.
3. **Modify baseline evaluation (`pipeline.py:212`):** Optionally filter to `eval_split_ids` for consistency (or keep all-splits for filtering, but report eval_split_ids subset).
4. **Update results processing:** Ensure A is computed as average over eval_split_ids, matching B and C aggregation when processing returned metrics.

