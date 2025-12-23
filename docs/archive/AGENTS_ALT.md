# AGENTS.md — Quick Reference for Language Model Unlearning Evaluation

**Purpose**: Fast navigation guide for AI agents working with this codebase.  
**For detailed docs**: See `README.md`, `CONFIGURATION_GUIDE.md`, `DATA.md`

---

## 🎯 Quick Decision Tree

**I want to...** → **Go to...**

| Task | Entry Point | Key Files | Config |
|------|-------------|-----------|--------|
| Run full unlearn+RTT experiment | `pipeline.py` → `run_pipeline()` (1756) | `pipeline.py`, `unlearn_corpus.py`, `finetune_corpus.py` | `conf/default.yaml` |
| Evaluate existing model | `pipeline.py --config-name=just_eval` | `pipeline.py:1756`, `unlearn_corpus.py:just_eval()` (1146) | `conf/just_eval.yaml` |
| Fine-tune existing model | `pipeline.py --config-name=only_ft` | `pipeline.py:1756`, `finetune_corpus.py:main()` (283) | `conf/only_ft.yaml` |
| Add new unlearning method | `unlearn_corpus.py:main()` (514) | `unlearn_corpus.py`, `pipeline.py:672` (unlearn router) | `conf/default.yaml` → `unlearn.types_config` |
| Modify data loading | `unlearn_corpus.py:load_jsonl()` (479) | `unlearn_corpus.py`, `data/requirements.py` | `data/` directory structure |
| Change model/config | `conf/default.yaml` | All files (dynamically loaded) | `conf/*.yaml` |
| Debug metrics/CSV | `pipeline.py:write_summary_csv()` (142) | `pipeline.py`, `utils/metrics.py` | `results_dir` config |

---

## 📍 Critical Entry Points

### Main Pipeline
```python
# pipeline.py:1755-1756
@hydra.main(config_path="conf", config_name=config_file, version_base=None)
def run_pipeline(cfg: DictConfig) -> None:
```
**Flow**: Config → Validate → Ray Init → Baseline Check → Unlearn Loop → RTT Loop → Summary CSV

### Unlearning Router
```python
# pipeline.py:672
@ray.remote(num_gpus=1)
def unlearn(...) -> dict:
    if unlearn_type == GD/WHP/FWF/LORA:
        → unlearn_corpus.main()  # pipeline.py:728
    elif unlearn_type == CUT:
        → rmu.unlearn_pipeline.main()  # pipeline.py:304 (external)
```

### Core Unlearning
```python
# unlearn_corpus.py:514
def main(
    unlearn_type: UnlearnType,
    train_files: list[str],
    base_model: str,
    lr: float,
    epochs: int,
    lora_rank: int = 0,  # LoRA support
    ...
) -> dict:
```

### Fine-tuning (RTT)
```python
# finetune_corpus.py:282-283
@ray.remote(num_gpus=1)
def main(
    train_files: list[str],
    val_files: list[str],
    base_model: str,
    ...
) -> dict:
```

---

## 🔧 Configuration Quick Reference

### Execution Modes
| Flag | Effect | Use Case |
|------|--------|----------|
| `just_eval=true` | Evaluate only, no training | Test model on dataset |
| `only_ft=true` | Skip unlearning, only RTT | Fine-tune existing model |
| `dont_ft=true` | Skip RTT after unlearning | Unlearning only experiments |

### Key Config Paths
```yaml
# conf/default.yaml
model_id: "Qwen/Qwen2.5-3B-Instruct"  # Base model
datasets: [YEARS]                      # Which datasets
unlearn.types: [LORA]                  # Methods: [GD, WHP, FWF, CUT, LORA]
unlearn.lora_ranks: [16, 32]           # LoRA ranks (if LORA)
baseline_min_forget_acc: 0.3           # Pre-flight threshold (0=disable)
ft.num_splits: 2                       # Eval splits for RTT
ft.eval_split_ids: null                # Explicit splits or null for auto
```

### Config Presets
| File | Purpose | Key Override |
|------|---------|--------------|
| `default.yaml` | Full pipeline | Base config |
| `lora_smoketest.yaml` | Quick test | `testing=true, max_samples=32` |
| `just_eval.yaml` | Evaluation only | `just_eval=true` |
| `only_ft.yaml` | Fine-tune only | `only_ft=true` |

---

## 📂 Code Navigation by Task

### Task: Understand Pipeline Flow
1. **Entry**: `pipeline.py:1756` → `run_pipeline()`
2. **Config Loading**: `pipeline.py:1757-1815` (Hydra + OmegaConf resolvers)
3. **Data Validation**: `pipeline.py:1760` → `data/validate_data.py:validate_required_artifacts()`
4. **Ray Init**: `pipeline.py:1765-1766` → `ray.init(num_gpus=...)`
5. **Baseline Check**: `pipeline.py:1882-1944` → `evaluate_baseline_model()` (569)
6. **Unlearn Loop**: `pipeline.py:2000-2130` → `unlearn.remote()` (672)
7. **RTT Loop**: `pipeline.py:2135-2250` → `finetune_corpus.main.remote()` (283)
8. **Summary**: `pipeline.py:2508` → `write_summary_csv()` (142)

### Task: Add/Modify Unlearning Method
1. **Enum**: `pipeline.py:28-34` → Add to `UnlearnType`
2. **Router**: `pipeline.py:672-810` → Add dispatch in `unlearn()`
3. **Implementation**: `unlearn_corpus.py:514` → `main()` function
4. **Config**: `conf/default.yaml` → `unlearn.types_config.{METHOD}`

### Task: Modify Data Loading
1. **Dataset Dict**: `pipeline.py:733-1730` → `datasets_dict[Datasets.{NAME}]`
2. **Path Resolution**: `pipeline.py:99` → `resolve_dataset_dict_paths()`
3. **Loading**: `unlearn_corpus.py:479` → `load_jsonl()`
4. **Validation**: `data/validate_data.py` → `validate_required_artifacts()`

### Task: Debug Metrics/Output
1. **Unlearn Metrics**: `pipeline.py:1105` → `write_metrics_to_csv()` (132)
2. **FT Metrics**: `finetune_corpus.py:520` → `write_metrics_to_csv()`
3. **Summary CSV**: `pipeline.py:2508` → `write_summary_csv()` (142)
4. **Output Dir**: `evals/pipeline/{unlearning,ft,summary}/`

---

## 🗂️ File Structure Quick Map

| File | Purpose | Key Functions | Lines |
|------|---------|---------------|-------|
| `pipeline.py` | Orchestration | `run_pipeline()` (1756), `unlearn()` (672), `main()` (867) | 2527 |
| `unlearn_corpus.py` | Unlearning impl | `main()` (514), `just_eval()` (1146) | 1248 |
| `finetune_corpus.py` | RTT impl | `main()` (283) | 523 |
| `data/requirements.py` | Dataset specs | `resolve_dataset_path()` | 286 |
| `data/validate_data.py` | Validation | `validate_required_artifacts()` | 79 |
| `utils/metrics.py` | Metrics utils | `select_scalar_acc()` | ? |
| `utils/attention_backend.py` | Attention | `get_attn_implementation()` | ? |

---

## 🔑 Key Concepts

### Unlearning Methods
| Method | Enum | Implementation | Notes |
|--------|------|----------------|-------|
| **GD** | `UnlearnType.GD` | `unlearn_corpus.py` | Gradient Difference |
| **WHP** | `UnlearnType.WHP` | `unlearn_corpus.py` | Wrong Hypothesis Penalty (RIA) |
| **FWF** | `UnlearnType.FWF` | `unlearn_corpus.py` | Fixed Wrong Fact |
| **LORA** | `UnlearnType.LORA` | `unlearn_corpus.py` | LoRA adapters (rank configurable) |
| **CUT** | `UnlearnType.CUT` | `rmu.unlearn_pipeline` | External module required |

### Evaluation Metrics (A/B/C)
- **A**: Unlearn accuracy (from `evals/pipeline/unlearning/*.csv`)
- **B**: Unlearn+RTT accuracy (from `evals/pipeline/ft/*.csv`, `rtt_condition="unlearn+rtt"`)
- **C**: Baseline+RTT accuracy (from `evals/pipeline/ft/*.csv`, `rtt_condition="baseline+rtt"`)
- **Baseline**: Pre-unlearning accuracy (stored in summary CSV)
- **Recovery Rate**: B/C (computed in `write_summary_csv()`)

### RTT Split Selection
- `ft.num_splits`: Number of eval splits to average over (default: 2)
- `ft.eval_split_ids`: Explicit list (e.g., `[0, 3]`) or `null` for deterministic sampling
- `ft.eval_seed`: Seed for sampling (default: 0, falls back to `data_seed`)
- Same splits used for A, B, C, and baseline for fair comparison

---

## ⚠️ Common Gotchas

1. **Baseline Check**: Set `baseline_min_forget_acc=0` to disable. Default 0.3 may skip datasets.
2. **Data Root**: Override via `UNLEARN_DATA_ROOT` env var or `data_root` config.
3. **Ray GPUs**: Auto-detects, max 8. Override with `num_gpus=N` config.
4. **LoRA Models**: Saved with `merge_and_unload()` for RTT compatibility (`unlearn_corpus.py:999-1002`).
5. **External RMU**: CUT method requires `rmu` package (not in repo).
6. **Eval Splits**: Must be consistent across A/B/C. Check `eval_split_ids` in CSV metrics.

---

## 🚀 Common Workflows

### Run Default Experiment
```bash
python pipeline.py  # Uses conf/default.yaml
```

### Quick Smoketest
```bash
python pipeline.py --config-name=lora_smoketest
```

### Custom Experiment
```bash
python pipeline.py \
    datasets=[YEARS,MMLU] \
    unlearn.types=[LORA,GD] \
    unlearn.lora_ranks=[8,16] \
    baseline_min_forget_acc=0.5
```

### Materialize Data
```bash
python scripts/materialize_data.py datasets=[MMLU]
python scripts/check_data.py datasets=[MMLU]
```

---

## 📊 Data Formats

### MCQ Format (`split_*.jsonl`)
```json
{"question": "...", "choices": ["A", "B", "C", "D"], "answer": 2}
```

### Corpus Format (`corpus_split_*.jsonl`)
```json
{"text": "...", "split": "split_0"}
```

### Wrong Hypothesis (`whp_corpus_split_*.jsonl`)
```json
{"text": "...", "split": "split_0", "correct_answer": 1975, "wrong_answers": [...]}
```

---

## 🔍 Where to Find Things

| What | Where |
|------|-------|
| Dataset definitions | `pipeline.py:1338-1730` (`datasets_dict`) |
| Unlearn routing logic | `pipeline.py:714-810` |
| LoRA initialization | `unlearn_corpus.py:627-644` |
| Model saving (LoRA merge) | `unlearn_corpus.py:997-1002` |
| RTT signature computation | `pipeline.py:537` |
| Summary CSV logic | `pipeline.py:142` |
| Baseline evaluation | `pipeline.py:569` |
| OmegaConf resolvers | `pipeline.py:499-535` |

---

## 📝 Development Rules

- Always validate after changes: `python scripts/check_data.py datasets=[YEARS]`
- Smoketest: `python pipeline.py --config-name=lora_smoketest`
- Check for redundant code after modifications
- Prefer GPU-accelerated paths when available

---

**Last Updated**: Based on codebase as of commit (verify line numbers may shift)

