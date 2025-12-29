# AGENTS.md — Codebase Guide

**Purpose**: Comprehensive guide to the repository codebase for AI agents and developers.  
**Audience**: Developers, AI agents, contributors working with the codebase.  
**Canonical for**: Entry points, pipeline flow, I/O schemas, method reference, code navigation.

This document provides a comprehensive guide to the repository for the paper *"Do Unlearning Methods Remove Information from Language Model Weights?"*

For installation and quick start, see [README.md](README.md).  
For configuration details, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).  
For data formats and materialization, see [docs/DATA.md](docs/DATA.md).

---

## Quick Decision Tree

**I want to...** → **Go to...**

| Task | Entry Point | Key Files | Config |
|------|-------------|-----------|--------|
| Run full unlearn+RTT experiment | `pipeline.py` → `run_pipeline()` (1236) | `pipeline.py`, `unlearn_corpus.py`, `finetune_corpus.py` | `conf/default.yaml` |
| Evaluate existing model | `pipeline.py just_eval=true` | `pipeline.py:1236`, `unlearn_corpus.py:just_eval()` | `conf/default.yaml` with `just_eval=true` |
| Fine-tune existing model | `pipeline.py only_ft=true` | `pipeline.py:1236`, `finetune_corpus.py:main()` | `conf/default.yaml` with `only_ft=true` |
| Run year concept evaluation | `analyze_year_concept.ipynb` | `analyze_year_concept.ipynb` | Set `RUN_NAME` and `MODEL_ID` in notebook |
| Run matched forgetting (LoRA) | `pipeline.py` with `matched_forgetting.enabled=true` | `pipeline.py`, `unlearn_corpus.py` | `conf/default.yaml` → `matched_forgetting` |
| Add new unlearning method | `unlearn_corpus.py:main()` | `unlearn_corpus.py`, `pipeline.py:314` (unlearn router) | `conf/default.yaml` → `unlearn.types_config` |
| Modify data loading | `unlearn_corpus.py:load_jsonl()` | `unlearn_corpus.py`, `data/requirements.py` | `data/` directory structure |
| Change model/config | `conf/default.yaml` | All files (dynamically loaded) | `conf/*.yaml` |
| Debug metrics | Results returned from Ray remote functions | `pipeline.py`, `unlearn_corpus.py`, `finetune_corpus.py` | Results stored in Ray object store |

---

## Repository Map

| Path | Description | Evidence |
|------|-------------|----------|
| `pipeline.py` | Main entry point; Hydra-based orchestration, Ray distributed execution | (`pipeline.py:1905`) `if __name__ == "__main__": run_pipeline()` |
| `unlearn_corpus.py` | Core unlearning methods: GD, WHP, FWF, LORA | (`unlearn_corpus.py`) Contains unlearning implementations |
| `finetune_corpus.py` | Fine-tuning for RTT (Retraining To Threshold) evaluation | (`finetune_corpus.py`) Contains fine-tuning implementations |
| `analyze_year_concept.ipynb` | Year concept evaluation and visualization notebook | (`analyze_year_concept.ipynb`) Evaluates existing models and creates visualizations |
| `scripts/generate_year_concept_eval.py` | Year concept dataset generator | (`scripts/generate_year_concept_eval.py`) Generates evaluation dataset |
| `conf/` | Hydra configuration files (YAML) | (`pipeline.py:1236`) `@hydra.main(config_path="conf", ...)` |
| `conf/default.yaml` | Default experiment configuration | (`conf/default.yaml`) |
| `data/` | Dataset directory containing all JSONL data files | See [DATA.md](DATA.md) |
| `data/requirements.py` | Dataset requirements + aliasing | (`data/requirements.py`) |
| `data/validate_data.py` | Dataset validation helper | (`data/validate_data.py`) |

---

## Entry Points

### Primary Entry Point

**File:** `pipeline.py`  
**Function:** `run_pipeline()` (line 1236)  
**Invocation:** `python pipeline.py [hydra_overrides]`

```python
# (pipeline.py:1236-1237)
@hydra.main(config_path="conf", config_name=config_file, version_base=None)
def run_pipeline(cfg: DictConfig) -> None:
```

The pipeline supports three main modes controlled by config flags:

| Mode | Config Flags | Description | Evidence |
|------|--------------|-------------|----------|
| **Unlearn + RTT** | `just_eval=false, only_ft=false, dont_ft=false` | Full pipeline: unlearn then fine-tune | (`pipeline.py:1236-1896`) |
| **Evaluation Only** | `just_eval=true` | Evaluate existing model on datasets | (`pipeline.py:1236-1846`) |
| **Fine-tune Only** | `only_ft=true` | Only perform RTT on existing model | (`pipeline.py:1236-1896`) |

### Secondary Entry Points

| File | Function | Purpose | Evidence |
|------|----------|---------|----------|
| `unlearn_corpus.py` | `main()` | Direct unlearning (GD/WHP/FWF/LORA) | (`unlearn_corpus.py`) |
| `unlearn_corpus.py` | `just_eval()` | Evaluation-only remote function | (`unlearn_corpus.py`) |
| `finetune_corpus.py` | `main()` | Direct fine-tuning | (`finetune_corpus.py`) |
| `analyze_year_concept.ipynb` | Evaluation + visualization | Year concept evaluation and analysis | (`analyze_year_concept.ipynb`) |
| `pipeline.py` | `evaluate_baseline_model()` | Baseline model pre-flight check | (`pipeline.py:212`) |
| `pipeline.py` | `main()` | Remote function for unlearning + RTT | (`pipeline.py:510`) |
| `pipeline.py` | `unlearn()` | Router to unlearning implementations | (`pipeline.py:314`) |

---

## End-to-End Pipeline

### Pipeline Narrative

1. **Initialization**: Ray cluster starts with available GPUs (`pipeline.py:1246-1247`)
2. **Config Loading**: Hydra loads and resolves YAML configuration (`pipeline.py:1237-1356`)
3. **Validation**: Required artifacts are validated before running (`pipeline.py:1241-1244`, see `data/validate_data.py`)
4. **Baseline Pre-flight Check** (if `baseline_min_forget_acc > 0`): Evaluates baseline model on forget sets to ensure it knows the information before unlearning (`pipeline.py:1365-1418`)
   - Skips datasets where baseline accuracy < threshold
   - Stores baseline accuracies for later analysis
5. **Experiment Loop**: For each (unlearn_type × dataset × hyperparameter combination):
   - **Unlearning Phase**: Call `main()` remote function which calls `unlearn()` → saves model to `models/`
   - **Metrics**: Results returned from remote functions (stored in Ray object store)
6. **RTT Phase** (if `dont_ft=false`): For each fine-tuning configuration:
   - Call `finetune_corpus.main()` remote function (both unlearned and baseline models)
   - Results returned from remote functions
8. **Cleanup**: Ray shutdown (`pipeline.py:1897`)

### Pipeline Stages with Code Locations

```
User Invocation (CLI)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Config Load (Hydra + OmegaConf)                             │
│ (pipeline.py:1237-1356)                                      │
│ - Resolves ${get_log_range:...}, ${get_num_layers:...}      │
│ - Logs config to wandb                                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Data Validation                                              │
│ (data/validate_data.py:1-80)                                 │
│ - Ensures required artifacts exist                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Ray Initialization                                           │
│ (pipeline.py:1765-1766) ray.init(num_gpus=...)              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Baseline Pre-flight Check (optional)                         │
│ (pipeline.py:1365-1418)                                      │
│                                                              │
│ evaluate_baseline_model() @ray.remote                       │
│ (pipeline.py:212)                                            │
│ - Evaluates baseline model on forget sets                   │
│ - Filters datasets below baseline_min_forget_acc threshold  │
│ - Stores baseline accuracies for later analysis              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Unlearning Phase (per config combination)                   │
│ (pipeline.py:1466-1610)                                      │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ main() @ray.remote → calls unlearn() @ray.remote        ││
│ │ (pipeline.py:510)                                         ││
│ │                                                          ││
│ │ unlearn() routes to:                                     ││
│ │ - GD/WHP/FWF/LORA → unlearn_corpus.main()               ││
│ │   (unlearn_corpus.py)                                    ││
│ │ - CUT → rmu.unlearn_pipeline.main()                     ││
│ │   (pipeline.py:462) [EXTERNAL MODULE]                   ││
│ └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Data Loading                                                 │
│ (unlearn_corpus.py:653-682)                                  │
│ - load_jsonl() for train/val/retain datasets                │
│ - make_k_shot() for few-shot prompts                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Model Loading                                                │
│ (unlearn_corpus.py:582-624)                                  │
│ - AutoModelForCausalLM.from_pretrained(base_model, ...)     │
│ - Flash Attention 2, bfloat16                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ LoRA Initialization (if lora_rank > 0)                      │
│ (unlearn_corpus.py:627-644)                                  │
│ - LoraConfig setup                                          │
│ - get_peft_model()                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Training Loop                                                │
│ (unlearn_corpus.py:908-993)                                  │
│ - Lion optimizer                                             │
│ - get_loss() for forget/retain loss computation             │
│ - Warmup scheduling                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Evaluation                                                   │
│ (unlearn_corpus.py:722-902)                                  │
│ - MCQ accuracy on forget/retain sets                        │
│ - Calibrated accuracy                                        │
│ - 5-shot evaluation (optional)                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Model Saving                                                 │
│ (unlearn_corpus.py:997-1002)                                 │
│ model.save_pretrained(save_name)                            │
│ tokenizer.save_pretrained(save_name)                        │
│ - LoRA merge_and_unload() if applicable                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Results Returned                                             │
│ Results stored in Ray object store, returned from remote    │
│ functions. Metrics include forget_accs, retain_accs, etc.   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ RTT Phase (Fine-tuning)                                      │
│ (pipeline.py:1610-1871)                                      │
│                                                              │
│ finetune_corpus.main() @ray.remote                          │
│ (finetune_corpus.py)                                         │
│ - Fine-tune unlearned model on forget set                   │
│ - Measure accuracy recovery                                  │
│ - Results returned from remote functions                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Results Processing                                           │
│ Results collected from Ray remote functions                 │
│ Metrics include forget_accs, retain_accs, etc.              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Pipeline Completion                                          │
│ All results collected and processed                          │
│ Analysis can be performed on returned metrics                │
└─────────────────────────────────────────────────────────────┘
```

---

## Inputs and Outputs

### I/O Inventory Table

| Artifact | Produced By | Consumed By | Format | Default Path/Pattern | Config Key(s) | Evidence |
|----------|-------------|-------------|--------|---------------------|---------------|----------|
| Hydra Config | User/defaults | `run_pipeline()` | YAML | `conf/*.yaml` | `--config-name` | (`pipeline.py:1048`) |
| Training Data (MCQ) | External/scripts | `load_jsonl()` | JSONL | `data/{dataset}/split_*.jsonl` | `datasets_dict[*]["val_files"]` | (`pipeline.py:733-1130`) |
| Training Data (Corpus) | GPT-4o scripts | `load_jsonl()` | JSONL | `data/{dataset}/corpus_split_*.jsonl` | `datasets_dict[*]["unlearn_files"]` | (`pipeline.py:733-1130`) |
| Wrong Hypothesis Data | GPT-4o scripts | `load_jsonl()` | JSONL | `data/{dataset}/whp_corpus_split_*.jsonl` | `datasets_dict[*]["wrong_unlearn_files"]` | (`pipeline.py:733-1130`) |
| Dev Set | External | `make_k_shot()` | JSONL | `data/{dataset}/dev.jsonl` | `datasets_dict[*]["dev_file"]` | (`pipeline.py:733-1130`) |
| Base Model | HuggingFace Hub | `AutoModelForCausalLM` | HF format | N/A | `model_id` | (`conf/default.yaml:18`) |
| Unlearned Model | `unlearn_corpus.main()` | RTT / evaluation | HF format | `models/{run_name}/{method}/{dataset}/{project}/rank{rank}-sc{sc}-{model_id}-rc{rc}-lr{lr}-epochs{epochs}` | `save_name`, `unlearn.save_unlearn_model`, `run_name` | (`pipeline.py:1559-1567`) |
| Fine-tuned Model | `finetune_corpus.main()` | Evaluation | HF format | `models/{run_name}/fted/{method}/{dataset}/{project}/{loss_type}/ft-skip_split{skip}/lr{lr}-epoch{epochs}` | `ft.save_models`, `run_name` | (`pipeline.py:722-728`) |
| Unlearning Metrics | Returned from `main()` remote function | Analysis | dict | Returned from Ray remote functions | `results_dir` | (`pipeline.py:510`) |
| Fine-tuning Metrics | Returned from `finetune_corpus.main()` | Analysis | dict | Returned from Ray remote functions | `results_dir` | (`finetune_corpus.py`) |
| Baseline Evaluation | `evaluate_baseline_model()` | Pre-flight check | dict | In-memory (returned from remote function) | `baseline_min_forget_acc` | (`pipeline.py:212`) |
| Year Concept Dataset | `scripts/generate_year_concept_eval.py` | Year concept evaluation | JSONL | `data/year_concept_eval/year_concept.jsonl` | N/A | (`scripts/generate_year_concept_eval.py`) |
| Year Concept Metrics | `analyze_year_concept.ipynb` | Analysis | CSV | `evals/pipeline/year_concept/{timestamp}--num{i}.csv` | `RUN_NAME`, `MODEL_ID` in notebook | (`analyze_year_concept.ipynb`, `pipeline.py:260-316`) |
| Checkpoint Manifest | `write_checkpoint_manifest_entry()` | Analysis | JSON | `models/{run_name}/manifest.json` | `run_name` | (`pipeline.py:214-293`) |
| Data Manifest | Manual/External | Analysis | JSON | `data/MANIFEST.json` | `data_root` | See [DATA.md](DATA.md) |
| WandB Logs | `wandb.log()` | Monitoring | WandB | N/A | `wandb_project_name` | (`unlearn_corpus.py:699-712`) |
| Error Logs | Exception handler | Debugging | Text | `pipeline_error.log` | N/A | (`pipeline.py:610-617`) |

### Data File Schemas

See [DATA.md](DATA.md) for complete data format documentation.

#### MCQ Format (`split_*.jsonl`)
```json
{
  "question": "When did the following event happen?\nThe First Congo War begins.",
  "choices": ["1994", "1995", "1996", "1997"],
  "answer": 2,
  "correct_year": 1996
}
```
**Evidence:** (`data/dates-years-trimmed/split_0.jsonl:1`)

#### Corpus Format (`corpus_split_*.jsonl`)
```json
{
  "text": "Juan Carlos I became King of Spain in 1975.",
  "split": "split_0"
}
```
**Evidence:** (`data/dates-years-trimmed/corpus_split_0.jsonl:1`)

#### Wrong Hypothesis Format (`whp_corpus_split_*.jsonl`)
```json
{
  "text": "Juan Carlos I became King of Spain in 1978.",
  "split": "split_0",
  "correct_answer": 1975,
  "wrong_answers": [1978, 1971, 1973]
}
```
**Evidence:** (`data/dates-years-trimmed/whp_corpus_split_0.jsonl:1`)

### Output Metrics Schema (CSV)

**Unlearning metrics columns** (`evals/pipeline/unlearning/*.csv`):
```python
{
    "model_path": str,          # Name/path of the model
    "dataset": str,             # Dataset enum name (e.g., "YEARS")
    "forget_accs": dict,        # {epoch: accuracy} for forget set
    "forget_accs_calibrated": dict,
    "forget_logits_dict": dict,
    "retain_accs": dict,        # {epoch: accuracy} for retain set
    "retain_accs_calibrated": dict,
    "retain_logits_dict": dict,
    "retain_accs_5_shot": dict,
    "retain_accs_5_shot_calibrated": dict,
    "retain_logits_5_shot_dict": dict,
    "unlearn_type": str,        # e.g., "GD", "WHP", "CUT"
    "unlearn_files": list,
    "wrong_unlearn_files": list,
    "val_files": list,
    "dev_file": str,
    "retain_files": list,
    "val_retain_files": list,
    "retain_dev_file": str,
    "base_model": str,
    "lr": float,
    "epochs": int,
    "batch_size": int,
    "val_batch_size": int,
    "retain_coeff": float,
    "warmup_steps": int,
    "data_seed": int,
    "eval_every": int,
    "save_name": str,
    "wandb_project_name": str,
    "samples": dict,            # Generated text samples
    "time": str,                # UTC timestamp
    "time_sf": str,             # SF timezone timestamp
    "start_time": str,
    "start_time_sf": str,
    "hydra_dict": dict,         # Full config
    "steering_coeff": float,    # For RMU/CUT
    "max_samples": int
}
```
**Note:** These metrics are returned as dictionaries from Ray remote functions. The actual structure may vary slightly depending on the unlearning method used.

---

## Files Generated by Pipeline

This section documents all files and directories created when running `pipeline.py`.

### 1. Model Checkpoints

#### Unlearned Models (Condition A)
**Location:** `models/{run_name}/{unlearn_type}/{dataset}/{wandb_project_name}/rank{lora_rank}-sc{steering_coeff}-{model_id}-rc{retain_coeff}-lr{lr}-epochs{epochs}/`

**Generated by:** `unlearn_corpus.main()` (called via `unlearn()` remote function)

**Code reference:** ```1559:1567:pipeline.py```

**Example path:**
```
models/2024-12-25_14-30-00/LORA/YEARS/my_experiment/rank8-sc20-TinyLlama_TinyLlama-1.1B-Chat-v1.0-rc0.1-lr4e-07-epochs5/
```

**Note:** `{run_name}` is auto-generated as a timestamp (YYYY-MM-DD_HH-MM-SS) if not specified in config, or uses the configured `run_name` value. This top-level folder isolates different pipeline runs.

**Contents:**
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Model weights
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- Other HuggingFace model artifacts

**Condition:** Only saved if `unlearn.save_unlearn_model=True` (or `FORCE_SAVE_ALL_MODELS=True`)

---

#### Fine-tuned Models (RTT - Condition B)
**Location:** `models/{run_name}/fted/{unlearn_type}/{dataset}/{wandb_project_name}/{loss_type}/ft-skip_split{skip_split}/lr{lr}-epoch{epochs}/`

**Generated by:** `finetune_corpus.main()` (called during RTT phase)

**Code reference:** ```722:728:pipeline.py```

**Example path:**
```
models/2024-12-25_14-30-00/fted/LORA/YEARS/my_experiment/QUESTION_LETTER_ANSWER/ft-skip_split0/lr1e-06-epoch6/
```

**Contents:** Same as unlearned models (full HuggingFace checkpoint)

**Condition:** Only saved if `ft.save_models=True` (or `FORCE_SAVE_ALL_MODELS=True`)

---

#### Baseline RTT Models (Condition C)
**Location:** `models/{run_name}/baseline_rtt/{dataset}/{model_id}/{loss_type}/skip_split{skip_split}/lr{lr}-epoch{epochs}/`

**Generated by:** `finetune_corpus.main()` (baseline RTT runs)

**Code reference:** ```1693:1698:pipeline.py``` and ```1857:1862:pipeline.py```

**Example path:**
```
models/2024-12-25_14-30-00/baseline_rtt/YEARS/TinyLlama_TinyLlama-1.1B-Chat-v1.0/QUESTION_LETTER_ANSWER/skip_split0/lr1e-06-epoch6/
```

**Note:** `{model_id}` in the path has slashes replaced with underscores (e.g., `Qwen/Qwen2.5-3B-Instruct` becomes `Qwen_Qwen2.5-3B-Instruct`).

**Contents:** Same as fine-tuned models

**Condition:** Only saved if `ft.save_models=True` (or `FORCE_SAVE_ALL_MODELS=True`)

---

### 2. Error Logs

**Location:** `pipeline_error.log` (in the working directory)

**Generated by:** Exception handlers in `main()` and `run_pipeline()`

**Code references:**
- ```797:805:pipeline.py``` (error handling in `main()`)
- ```1857:1867:pipeline.py``` (error handling in `run_pipeline()`)
- ```1883:1892:pipeline.py``` (error handling for baseline RTT)

**Format:** Appended text file with:
- Timestamp of error
- Exception message
- Full traceback

**Note:** Errors are appended; previous errors remain in the file.

---

### 3. WandB Logs

**Location:** Remote (WandB cloud) and local cache at `~/.wandb/`

**Generated by:** `wandb.init()` and `wandb.log()` calls

**Code references:**
- ```1352:1363:pipeline.py``` (pipeline-level WandB init)
- WandB logging happens inside `unlearn_corpus.py` and `finetune_corpus.py`

**Contents:**
- Hyperparameters
- Training metrics (loss, accuracy)
- Model checkpoints (if configured)
- System metrics

**Project name:** Controlled by `wandb_project_name` config (default: from `conf/default.yaml`)

---

### 4. CSV Metrics Files

**Note:** The current `pipeline.py` does not contain `write_metrics_to_csv()` or `write_summary_csv()` functions. These may be implemented elsewhere or were removed in a refactor. The directory structure suggests these files are expected to exist.

**Expected locations (based on documentation and directory structure):**
- `evals/pipeline/unlearning/{timestamp}--num{i}.csv` - Unlearning metrics (Condition A)
- `evals/pipeline/ft/{timestamp}--num{i}.csv` - Fine-tuning metrics (Conditions B and C)
- `evals/pipeline/summary/{timestamp}.csv` - Summary CSV with A/B/C stats (if generated)
- `evals/pipeline/year_concept/{timestamp}--num{i}.csv` - Year concept evaluation metrics (generated by `analyze_year_concept.ipynb`)

**Config key:** `results_dir` (default: `"evals/pipeline"`)

**Year Concept CSV Format:**
- Columns: `model_path`, `base_model`, `lora_rank`, `dataset_name`, `ordering_acc`, `arithmetic_acc`, `classification_acc`, `overall_acc`, `ordering_count`, `arithmetic_count`, `classification_count`, `total_count`, `timestamp`, `start_time_sf`
- Generated by: `analyze_year_concept.ipynb` (calls `write_year_concept_csv()` from `pipeline.py:260-316`)
- Condition: Generated when running `analyze_year_concept.ipynb` evaluation cell

---

### 5. Hydra Output Directory

**Location:** `outputs/{date}/{time}/` (Hydra default)

**Generated by:** Hydra framework

**Contents:**
- Copy of the config file used
- Hydra logs
- Working directory for that run

**Code reference:** ```1236:1237:pipeline.py``` (`@hydra.main` decorator)

---

### Summary Table

| File Type | Location Pattern | Generated When | Config Control |
|-----------|-----------------|----------------|----------------|
| **Unlearned Models** | `models/{run_name}/{method}/{dataset}/{project}/rank{rank}-sc{sc}-{model_id}-rc{rc}-lr{lr}-epochs{epochs}/` | Unlearning phase completes | `unlearn.save_unlearn_model` |
| **Fine-tuned Models (B)** | `models/{run_name}/fted/{method}/{dataset}/{project}/{loss_type}/ft-skip_split{skip}/lr{lr}-epoch{epochs}/` | RTT phase completes | `ft.save_models` |
| **Baseline RTT Models (C)** | `models/{run_name}/baseline_rtt/{dataset}/{model_id}/{loss_type}/skip_split{skip}/lr{lr}-epoch{epochs}/` | Baseline RTT completes | `ft.save_models` |
| **Checkpoint Manifest** | `models/{run_name}/manifest.json` | When any checkpoint is saved | Automatic (always written) | (`pipeline.py:538-570`) |
| **Matched Forgetting JSON** | `models/{run_name}/matched_forgetting.json` | Matched forgetting selection completes | `matched_forgetting.enabled=true` | (`pipeline.py:573-597`) |
| **Year Concept CSV** | `evals/pipeline/year_concept/{timestamp}--num{i}.csv` | Year concept evaluation completes | Run `analyze_year_concept.ipynb` | (`analyze_year_concept.ipynb`, `pipeline.py:332-391`) |
| **Error Logs** | `pipeline_error.log` | Any exception occurs | N/A (always written) |
| **WandB Logs** | Remote + `~/.wandb/` | Training/evaluation runs | `wandb_project_name` |
| **Hydra Outputs** | `outputs/{date}/{time}/` | Pipeline starts | Hydra default |

---

### Notes

1. **Model saving**: Controlled by `unlearn.save_unlearn_model` and `ft.save_models`. If `FORCE_SAVE_ALL_MODELS=True` (line 26), all models are saved regardless of config.
2. **Directory structure**: All paths are relative to the working directory (Hydra runtime directory or current working directory).
3. **Run name isolation**: The `{run_name}` top-level folder isolates different pipeline runs. If `run_name` is `null` in config, it auto-generates a timestamp (YYYY-MM-DD_HH-MM-SS). For RTT-only runs (`only_ft=true`), `run_name` must be explicitly specified.
4. **Checkpoint Manifest**: A `manifest.json` file is automatically created in `models/{run_name}/` containing metadata for all saved checkpoints (A, B, and C). This manifest is used by `analyze_rtt.ipynb` for checkpoint discovery instead of parsing directory names. The manifest includes hyperparameters, paths, timestamps, and relationships between checkpoints (e.g., B checkpoints reference their parent A checkpoint via `a_path`). See (`pipeline.py:214-293`) for the manifest writer implementation.
5. **Metrics**: Results are returned as dictionaries from Ray remote functions. Users can process these results as needed (e.g., write to CSV, analyze in notebooks).
6. **Timestamps**: The `run_name` folder uses timestamps when auto-generated. Individual model paths within use hyperparameters, not timestamps. Error logs include timestamps.

---

## Configuration

For detailed configuration documentation, see [CONFIGURATION.md](CONFIGURATION.md).

### Config File Structure

All configs inherit from `conf/default.yaml` and use Hydra's override system.

**Key configuration sections:**

```yaml
# (conf/default.yaml)

# Execution modes
just_eval: false          # Only evaluate, no training
only_ft: false            # Only fine-tune, skip unlearning
dont_ft: false            # Skip RTT after unlearning

# Baseline validation
baseline_min_forget_acc: 0.3  # Minimum baseline accuracy to proceed (set to 0 to disable)

# Model
model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
num_layers: ${get_num_layers:${model_id}}  # Dynamic resolver

# Datasets (enum names from pipeline.py)
datasets: [YEARS]         # Can be list: [YEARS, MMLU, WMDP_MCQ_CORPUS]

# Unlearning config
unlearn:
  types: [LORA]           # [GD, WHP, FWF, CUT, LORA]
  lora_ranks: [1, 2, 4, 8, 16, 32]  # For LORA method
  save_unlearn_model: true
  types_config:
    LORA:
      loss_type: CORPUS   # CORPUS, LETTER, LETTER_ANSWER, etc.
      datasets_config:
        YEARS:
          epochs_lst: [5]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-3, 1e3, 10}
            add: [0, 0.002]

# Fine-tuning (RTT) config
ft:
  num_splits: 2
  loss_types: [QUESTION_LETTER_ANSWER]
  epochs_lst: [6]
  lrs: ${get_log_range:1e-7,5e-6,2}
  save_models: false

# Output
results_dir: "evals/pipeline"
wandb_project_name: "experiment_name"
data_root: "data"
acc_selection_rule: "final_epoch"  # "final_epoch" or "max_epoch" for A/B/C summary stats

# Run name for isolating different pipeline runs in models/ directory
# - null (default): Auto-generate timestamp-based name (YYYY-MM-DD_HH-MM-SS)
# - string: Use specified name (required for RTT-only runs with only_ft=true)
run_name: null
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file
**Evidence:** (`conf/default.yaml:1-320`)

### Custom OmegaConf Resolvers

| Resolver | Function | Example | Evidence |
|----------|----------|---------|----------|
| `get_log_range` | Generate logarithmic range | `${get_log_range:1e-3, 1e3, 10}` → `[0.001, 0.01, 0.1, 1, 10, 100]` | (`pipeline.py:499`) |
| `get_num_layers` | Get model layer count | `${get_num_layers:${model_id}}` | (`pipeline.py:512`) |
| `resolve_freeze_layers` | Convert fractions to layer indices | `${resolve_freeze_layers:[[0, "0.5"]], ${model_id}}` | (`pipeline.py:522`) |

### Available Config Presets

| Config File | Purpose | Key Settings | Evidence |
|-------------|---------|--------------|----------|
| `default.yaml` | Standard unlearn+RTT | `just_eval=false, only_ft=false` | (`conf/default.yaml`) |
| `just_eval.yaml` | Evaluation only | `just_eval=true` | (`conf/just_eval.yaml:6`) |
| `only_ft.yaml` | Fine-tune existing model | `only_ft=true` | (`conf/only_ft.yaml:8`) |
| `lora_smoketest.yaml` | Quick LoRA smoketest (minutes) | `testing=true, 1 epoch, 32 samples` | (`conf/lora_smoketest.yaml`) |
| `lora_rank_sweep.yaml` | LoRA rank sweep experiment | `lora_ranks: [1,2,4,8,16,32]` | (`conf/lora_rank_sweep.yaml`) |
| `many_cut_sc.yaml` | Grid search over CUT steering coefficients | `unlearn.cut_scs: [0.1, 1, 10]` | (`conf/many_cut_sc.yaml`) |
| `random_bd.yaml` | Random birthdays experiments | `datasets: [RANDOM_BD_SAME_RETAIN]` | (`conf/random_bd.yaml`) |

---

## Common Workflows

### (a) Install + Environment Setup

See [README.md](../README.md) for installation instructions.

### (b) Run Baseline Evaluation

Evaluate a pre-trained model on a dataset without any training:

```bash
# Using just_eval config
python pipeline.py --config-name=just_eval \
    model_id="meta-llama/Meta-Llama-3-8B" \
    datasets=[YEARS] \
    eval_model_paths=["meta-llama/Meta-Llama-3-8B"]
```
**Evidence:** (`conf/just_eval.yaml:6-7`)

### (b.1) Baseline Pre-flight Check

The pipeline automatically validates that the baseline model knows the information before running unlearning experiments:

```bash
# Enable baseline check (default threshold: 0.3)
python pipeline.py \
    datasets=[YEARS] \
    unlearn.types=[LORA] \
    baseline_min_forget_acc=0.3

# Disable baseline check (set to 0)
python pipeline.py \
    datasets=[YEARS] \
    unlearn.types=[LORA] \
    baseline_min_forget_acc=0

# Custom threshold
python pipeline.py \
    datasets=[YEARS] \
    unlearn.types=[LORA] \
    baseline_min_forget_acc=0.5  # Require 50% baseline accuracy
```

If a dataset fails the baseline check, the pipeline will skip unlearning experiments for that dataset and print a clear message. Baseline accuracies are stored in the summary CSV as `forget_acc_baseline`.
**Evidence:** (`pipeline.py:1882-1944`)

### (c) Run an Unlearning Method

```bash
# Gradient Difference (GD) on YEARS dataset
python pipeline.py \
    datasets=[YEARS] \
    unlearn.types=[GD] \
    model_id="meta-llama/Meta-Llama-3-8B" \
    wandb_project_name="my_experiment"

# Random Incorrect Answer (WHP/RIA) on MMLU
python pipeline.py \
    datasets=[MMLU] \
    unlearn.types=[WHP] \
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# CUT/RMU method (requires external rmu module)
python pipeline.py \
    datasets=[WMDP_MCQ_CORPUS] \
    unlearn.types=[CUT] \
    unlearn.cut_scs=[0.1,1,10]
```
**Evidence:** (`pipeline.py:28-34`) for method enums

### (d) Run Only Fine-tuning (RTT)

Fine-tune an already unlearned model:

```bash
python pipeline.py --config-name=only_ft \
    run_name="my_rtt_run" \
    ft_model_paths=[["path/to/unlearned/model", "YEARS"]] \
    ft.epochs_lst=[10] \
    ft.lrs=[1e-6,5e-6]
```

**Note:** `run_name` is **required** for RTT-only runs (`only_ft=true`) to ensure models are saved in the correct isolated folder. If not specified, the pipeline will raise an error.

**Evidence:** (`conf/only_ft.yaml:8-9`, `pipeline.py:1291-1295`)

### (e) Skip RTT (Unlearning Only)

```bash
python pipeline.py \
    datasets=[MMLU] \
    unlearn.types=[GD,WHP] \
    dont_ft=true
```
**Evidence:** (`pipeline.py:385`) `dont_ft: bool = False`

### (e.1) Run Matched Forgetting (LoRA)

Run matched-forgetting selection for LoRA unlearning. For each LoRA rank, performs a grid search over hyperparameters, selects the checkpoint that achieves forget accuracy closest to target (default: 0.60 ± 0.02), then minimizes retain damage.

```bash
python pipeline.py \
    datasets=[YEARS] \
    unlearn.types=[LORA] \
    unlearn.lora_ranks=[8,16,32] \
    model_id="Qwen/Qwen2.5-3B-Instruct" \
    matched_forgetting.enabled=true \
    matched_forgetting.target_forget_acc=0.60 \
    matched_forgetting.tolerance=0.02 \
    matched_forgetting.max_trials_per_rank=18 \
    wandb_project_name="matched_forgetting_experiment"
```

**Key Features:**
- Grid search over `rc_range`, `lr_range`, `epochs_range` (up to `max_trials_per_rank` candidates)
- Selects checkpoint with forget accuracy closest to target (minimizing retain damage as tie-breaker)
- Selected checkpoints are tagged with `["matched_forgetting"]` in manifest
- Selection results stored in `models/{run_name}/matched_forgetting.json`
- RTT phase automatically runs on selected checkpoints (if `dont_ft=false`)

**Configuration:**
```yaml
matched_forgetting:
  enabled: false
  target_forget_acc: 0.60
  tolerance: 0.02
  max_trials_per_rank: 18
  search_space:
    rc_range: ${get_log_range:0.001, 10.0, 3}
    rc_add: [0.01, 0.1, 1.0]
    lr_range: [2e-7, 4e-7, 8e-7]
    epochs_range: [3, 5, 6]
  selection_priority: ["retain_damage", "compute", "retain_coeff"]
  acc_selection_rule: final_epoch
  save_all_candidates: true
```

**Outputs:**
- `models/{run_name}/matched_forgetting.json` - Selection results with hyperparameters
- Selected checkpoints in `models/{run_name}/LORA/{dataset}/...` tagged in manifest
- RTT checkpoints (B) generated from selected matched forgetting checkpoints (A)

**Evidence:** (`pipeline.py:2050-2365`) for matched forgetting implementation

---

### (e.2) Run Year Concept Evaluation

Evaluate general year understanding (ordering, arithmetic, classification) on existing unlearned models.

**Workflow:**
1. Generate models using `pipeline.py` (year concept evaluation is NOT run during pipeline)
2. Evaluate and visualize using `analyze_year_concept.ipynb`

**Step 1: Generate Dataset (One-time)**
```bash
python scripts/generate_year_concept_eval.py
```
This creates `data/year_concept_eval/year_concept.jsonl` with ~300 questions.

**Step 2: Generate Models**
```bash
python pipeline.py \
    datasets=[YEARS] \
    unlearn.types=[LORA] \
    unlearn.lora_ranks=[8,16,32,64,128] \
    model_id="Qwen/Qwen2.5-3B-Instruct"
```

**Step 3: Evaluate and Visualize**
Open `analyze_year_concept.ipynb`:
1. Set configuration:
   ```python
   RUN_NAME = "2025-12-28_05-13-18"  # Your run name
   MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Your base model
   LORA_RANKS = None  # Evaluate all ranks, or [8, 16, 32]
   ```
2. Run all cells to evaluate models and generate visualizations

**Output:**
- CSV: `evals/pipeline/year_concept/{timestamp}--num{i}.csv`
- Plots: `figures/year_concept/*.png` (overall disruption, per-metric breakdown, correlation analysis)
- Summary: `figures/year_concept/summary_table.csv`

**Note:** The notebook automatically discovers models from `models/{run_name}/manifest.json`. It evaluates the baseline model (lora_rank=0) and all LoRA-unlearned models (type A).

**Evidence:** (`analyze_year_concept.ipynb`)

### (f) Multi-GPU Distributed Training

The pipeline automatically uses Ray for distributed training:

```bash
# Automatically uses up to 8 GPUs
python pipeline.py num_gpus=4  # Limit to 4 GPUs
```
**Evidence:** (`pipeline.py:1765`) `num_gpus = 8 if get_num_gpus() >= 8 else get_num_gpus()`

---

## Code Navigation by Task

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
2. **Router**: `pipeline.py:314` → Add dispatch in `unlearn()`
3. **Implementation**: `unlearn_corpus.py` → `main()` function
4. **Config**: `conf/default.yaml` → `unlearn.types_config.{METHOD}`

### Task: Modify Data Loading
1. **Dataset Dict**: `pipeline.py` → `datasets_dict[Datasets.{NAME}]` (around line 733)
2. **Path Resolution**: `pipeline.py` → `resolve_dataset_dict_paths()` (around line 99)
3. **Loading**: `unlearn_corpus.py` → `load_jsonl()`
4. **Validation**: `data/validate_data.py` → `validate_required_artifacts()`

### Task: Debug Metrics/Output
1. **Unlearn Metrics**: `pipeline.py:1105` → `write_metrics_to_csv()` (132)
2. **FT Metrics**: `finetune_corpus.py:520` → `write_metrics_to_csv()`
3. **Summary CSV**: `pipeline.py:2508` → `write_summary_csv()` (142)
4. **Output Dir**: `evals/pipeline/{unlearning,ft,summary}/`

### Task: Understand Checkpoint Discovery
1. **Manifest Creation**: `pipeline.py:214-293` → `write_checkpoint_manifest_entry()` (called after A/B/C checkpoint saves)
2. **Manifest Reading**: `analyze_rtt.ipynb` → `discover_checkpoints()` (reads from `models/{run_name}/manifest.json`)
3. **A Checkpoint Entry**: Written in `pipeline.py:792-797` after unlearning completes
4. **B Checkpoint Entry**: Written in `finetune_corpus.py:607-612` after RTT on unlearned model
5. **C Checkpoint Entry**: Written in `finetune_corpus.py:607-612` after RTT on baseline model

---

## Gotchas / Assumptions

### GPU Requirements

- **Minimum**: 1 GPU with ≥24GB VRAM (for 7B models with bfloat16)
- **Recommended**: 8× A100/H100 GPUs for full hyperparameter sweeps
- Flash Attention 2 is optional but recommended for faster training on compatible GPUs (Ampere or newer)
- By default, the codebase uses PyTorch's SDPA (scaled dot-product attention) if flash-attn is not available
- See `utils/attention_backend.py` for the attention backend selection logic
- Dataset paths are validated before pipeline run; missing artifacts produce a fail-fast error with a materialization command.
- Override the data root via `UNLEARN_DATA_ROOT` or `data_root` in Hydra config.

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `WANDB_API_KEY` | Yes | Weights & Biases logging |
| `OPENAI_API_KEY` | Only for data generation | GPT-3.5 API for corpus creation |

**Evidence:** (`data/dates-years-trimmed/to_corpus.py:12-13`)

### Data Location Assumptions

Data must be in `data/` directory with specific naming. See [DATA.md](DATA.md) for details.

### External Dependencies

- **RMU Module**: The CUT/RMU unlearning method requires an external `rmu` package not included in this repository:
  ```python
  # (pipeline.py:463)
  import rmu.unlearn_pipeline as rmu
  ```
  **TODO:** Locate or install the `rmu` package separately.

### Caching / Disk Usage

- Models saved to `models/` can be large (~14GB per 7B model checkpoint)
- Ray uses `/tmp` for object store (configurable via `RAY_TMPDIR`)
- WandB logs to `~/.wandb/` by default

### Seed Control

- Data shuffling seed: `data_seed` config key (`conf/default.yaml:34`)
- Model seeds are not explicitly set (PyTorch defaults)
- **Evidence:** (`unlearn_corpus.py:668`) `random.Random(data_seed).shuffle(train_dataset)`

### Expected Runtime Directory Layout

After running experiments, expect this structure:
```
.
├── models/
│   └── {run_name}/              # Top-level folder (timestamp or specified name)
│       ├── manifest.json        # Checkpoint manifest (A/B/C entries with metadata)
│       ├── {method}/            # e.g., LORA, GD
│       │   └── {dataset}/       # e.g., YEARS
│       │       └── {wandb_project}/
│       │           └── rank{rank}-sc{sc}-{model_id}-rc{rc}-lr{lr}-epochs{epochs}/
│       │               ├── config.json
│       │               ├── model.safetensors
│       │               └── tokenizer.json
│       ├── fted/
│       │   └── {method}/
│       │       └── {dataset}/
│       │           └── {project}/
│       │               └── {loss_type}/
│       │                   └── ft-skip_split{skip}/
│       │                       └── lr{lr}-epoch{epochs}/
│       └── baseline_rtt/
│           └── {dataset}/
│               └── {model_id}/
│                   └── {loss_type}/
│                       └── skip_split{skip}/
│                           └── lr{lr}-epoch{epochs}/
├── evals/
│   └── pipeline/
│       ├── unlearning/
│       │   └── {timestamp}--num{i}.csv      # A: Unlearn metrics
│       ├── ft/
│       │   └── {timestamp}--num{i}.csv     # B/C: RTT metrics
│       ├── year_concept/
│       │   └── {timestamp}--num{i}.csv     # Year concept evaluation metrics
│       └── summary/
│           └── {timestamp}.csv               # A/B/C summary with baseline
├── pipeline_error.log          # If errors occur
└── wandb/                      # WandB local cache
```

**Note:** The `{run_name}` folder isolates different pipeline runs. When `run_name=null` (default), it auto-generates a timestamp (e.g., `2024-12-25_14-30-00`). This prevents different runs with different settings from being grouped together in analysis.

**Manifest File:** The `manifest.json` file contains structured metadata for all checkpoints (A, B, C) in that run. This replaces fragile directory name parsing in `analyze_rtt.ipynb`. Each entry includes hyperparameters, paths, timestamps, and relationships (e.g., B entries include `a_path` pointing to their parent A checkpoint). See (`pipeline.py:214-293`) for the manifest writer implementation.

### Failure Modes

1. **Out of Memory**: Reduce `batch_size` or `val_batch_size`
2. **Ray Timeout**: Increase Ray object store memory
3. **Model Not Found**: Ensure HuggingFace login for gated models
4. **RMU Import Error**: CUT method requires external `rmu` module
5. **Baseline Check Failed**: If `baseline_min_forget_acc` threshold is not met, the pipeline will skip that dataset. Lower the threshold or set to `0` to disable the check.

---

## Development Rules

- Always run tests or a minimal validation after every code change.
- Whenever code is changed, check if there is now redundant code.
- Always choose the most efficient implementation path given available resources (e.g., prefer GPU-accelerated configs over CPU fallbacks when GPUs are available).
- Minimal validation: `python scripts/check_data.py datasets=[YEARS]`.
- Smoketest (completes in minutes): `python pipeline.py --config-name=full_pipeline_test`

---

## Method Reference

### Unlearning Methods (UnlearnType Enum)

| Method | Enum Value | Description | Implementation | Evidence |
|--------|------------|-------------|----------------|----------|
| **CUT** | `UnlearnType.CUT` | CUT/RMU (Li et al. 2024) - representation engineering | `rmu.unlearn_pipeline` | (`pipeline.py:29`) |
| **GD** | `UnlearnType.GD` | Gradient Difference - maximize loss on forget set | `unlearn_corpus.py` | (`pipeline.py:30`) |
| **WHP** | `UnlearnType.WHP` | Wrong Hypothesis Penalty (RIA) - train on wrong answers | `unlearn_corpus.py` | (`pipeline.py:31`) |
| **FWF** | `UnlearnType.FWF` | Fixed Wrong Fact - train on fixed incorrect facts | `unlearn_corpus.py` | (`pipeline.py:32`) |
| **LORA** | `UnlearnType.LORA` | LoRA-based unlearning - train only adapter weights | `unlearn_corpus.py` | (`pipeline.py:33`) |

### Loss Types (LossType Enum)

| Loss Type | Description | Evidence |
|-----------|-------------|----------|
| `LETTER` | Loss only on answer letter token | (`pipeline.py:37`) |
| `CORPUS` | Loss on all tokens | (`pipeline.py:38`) |
| `LETTER_ANSWER` | Loss on letter and answer text | (`pipeline.py:39`) |
| `QUESTION_LETTER_ANSWER` | Loss on question, letter, and answer | (`pipeline.py:40`) |
| `NUMBER` | Loss only on number tokens | (`pipeline.py:42`) |

### Data Formats (DataFormat Enum)

| Format | Description | Evidence |
|--------|-------------|----------|
| `CORPUS` | Plain text | (`pipeline.py:68`) |
| `MCQ` | Multiple choice questions | (`pipeline.py:69`) |
| `TF` | True/False format | (`pipeline.py:70`) |

---

## TODO / Open Questions

| Item | Files to Inspect | Reason |
|------|-----------------|--------|
| RMU/CUT Implementation | External `rmu` package | imports `rmu.unlearn_pipeline` but module not in repo |
| WMDP Original Data Source | `data/wmdp-deduped/dedup-bio.py`, `dedup-cyber.py` | Need to trace original WMDP data before deduplication |
| Fineweb Retain Data | Auto-materialized | `scripts/materialize_data.py` creates `fineweb_edu_seed-42/split_{i}` |
| Wikitext Retain Data | Auto-materialized | `scripts/materialize_data.py` creates `wikitext/wikitext_dataset` |
| BeaverTails Dataset | Auto-materialized | `scripts/materialize_data.py` creates beavertails category splits |
| Day of Month Dataset | Not in repo | `Datasets.DAY_OF_THE_MONTH` references missing data |
| Plotting/Aggregation Scripts | None found | No scripts for aggregating results or generating figures |
| Test Suite | None found | No unit tests or integration tests discovered |

