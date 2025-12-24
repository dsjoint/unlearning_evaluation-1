# Configuration Guide

**Purpose**: Guide for changing models, system resources, and experiment parameters.  
**Audience**: Users configuring experiments, switching models, adjusting resources.  
**Canonical for**: All configuration-related information.

This document lists all files that need to be modified when changing the base model or system resources (GPUs, memory, batch sizes, etc.).

---

## Table of Contents

1. [Changing the Base Model](#changing-the-base-model)
2. [Experiment Configuration](#experiment-configuration)
3. [Changing System Resources](#changing-system-resources)
4. [Quick Reference](#quick-reference)

---

## Changing the Base Model

When switching to a different base model, update the following files:

| File | Parameter(s) to Update | Description |
|------|------------------------|-------------|
| `conf/default.yaml` | `model_id`, `ft_model_paths` | Primary config; all other configs inherit from this |
| `conf/full_pipeline_test.yaml` | `model_id`, `ft_model_paths` | Full pipeline test configuration |

**Note:** You can create additional config files by inheriting from `default.yaml` using Hydra's config system. The current repository includes `default.yaml` and `full_pipeline_test.yaml` as examples.

### Key Parameters

```yaml
# Primary model identifier (HuggingFace model ID or local path)
model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Baseline validation threshold (set to 0 to disable)
baseline_min_forget_acc: 0.3

# Models for fine-tuning phase: [[model_path, dataset_name], ...]
ft_model_paths: [["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "MMLU"]]

# Models for evaluation-only mode
eval_model_paths: ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]

# Number of layers (auto-resolved from model config)
num_layers: ${get_num_layers:${model_id}}
```

**Note:** Python files automatically load models based on config - no code changes needed when switching models.

---

## Experiment Configuration

### Execution Modes

- `just_eval: true` - Evaluate existing models only (no training)
- `only_ft: true` - Skip unlearning, run fine-tuning only  
- `dont_ft: true` - Run unlearning, skip RTT phase

### Unlearning Configuration

Hyperparameters are configured per method and dataset:

```yaml
unlearn:
  types: [LORA]                    # Methods: [GD, WHP, FWF, CUT, LORA]
  lora_ranks: [16, 32]             # For LORA method only
  types_config:
    LORA:
      loss_type: CORPUS
      datasets_config:
        YEARS:
          epochs_lst: [3]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-2, 1e2, 10}  # Log range: [start, start*step, ...] until < end
            add: []                                  # Additional values to include
```

**OmegaConf Resolvers:**
- `${get_log_range:start, end, step}` - Generates logarithmic range: `[start, start*step, start*step², ...]` until < end
- `${get_num_layers:model_id}` - Auto-detects model layer count
- `${resolve_freeze_layers:[[0, "0.5"]], model_id}` - Converts fraction to layer indices

See `conf/default.yaml` for complete examples per method/dataset.

---

## Changing System Resources

### GPU and Memory Configuration

```yaml
# Number of GPUs for Ray distributed training
num_gpus: 1

# Training and validation batch sizes
batch_size: 4
val_batch_size: 8

# Attention backend: auto, flash_attention_2, sdpa, eager
# - auto: Use flash_attention_2 if available, otherwise sdpa (recommended)
# - flash_attention_2: Fastest on Ampere+ GPUs (requires flash-attn package)
# - sdpa: PyTorch scaled dot-product attention (good fallback)
# - eager: Standard PyTorch attention (slowest, most compatible)
attn_backend: auto

# Warmup steps for learning rate scheduler
warmup_steps: 24
```

**Note:** Precision settings (`bfloat16` for unlearning, `float16` for fine-tuning) are hardcoded in Python files and may need manual modification for different GPU architectures.

---

## Quick Reference

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_gpus` | 1 | Number of GPUs for training |
| `batch_size` | 4 | Training batch size |
| `val_batch_size` | 8 | Validation batch size |
| `warmup_steps` | 24 | LR scheduler warmup |
| `attn_backend` | `auto` | Attention implementation |
| `data_seed` | 4 | Random seed for data |
| `data_root` | `data` | Data root override (env: `UNLEARN_DATA_ROOT`) |
| `baseline_min_forget_acc` | 0.3 | Minimum baseline accuracy threshold (0 to disable) |
| `acc_selection_rule` | `final_epoch` | Accuracy selection for summary CSV (`final_epoch` or `max_epoch`) |
| `eval_every` | 2 | Evaluation frequency (every N epochs) |
| `ft.num_splits` | 2 | Number of evaluation splits to average over |
| `ft.eval_split_ids` | `null` | Explicit list of eval split IDs (e.g., `[0, 3]`). If `null`, splits are sampled deterministically |
| `ft.eval_seed` | 0 | Seed for sampling eval splits when `eval_split_ids` is `null` |

### RTT Evaluation Split Configuration

The same evaluation splits are used consistently across baseline, unlearning (A), unlearn+RTT (B), and baseline+RTT (C) for fair comparison.

### CLI Override Examples

```bash
# Change model via CLI
python pipeline.py model_id="meta-llama/Meta-Llama-3-8B"

# Change GPU count and batch size
python pipeline.py num_gpus=4 batch_size=2 val_batch_size=4

# Override evaluation splits
python pipeline.py ft.eval_split_ids=[0,2,4]
python pipeline.py ft.num_splits=3 ft.eval_seed=42

# Multiple overrides
python pipeline.py \
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    num_gpus=1 \
    batch_size=4 \
    attn_backend=sdpa \
    baseline_min_forget_acc=0.5
```

### Common Patterns

**Memory-Constrained Setup:**
```yaml
num_gpus: 1
batch_size: 2
val_batch_size: 4
attn_backend: sdpa
```

**Quick Smoke Test:**
```yaml
testing: true
dont_ft: true
unlearn:
  max_samples_lst: [32]
  types_config:
    LORA:
      datasets_config:
        YEARS:
          epochs_lst: [1]
          rcs:
            add: [1]
```

---

## Checklist: Changing Model

- [ ] Update `model_id` in `conf/default.yaml`
- [ ] Update `ft_model_paths` in `conf/default.yaml` (and any other config files you're using)
- [ ] Update `eval_model_paths` if using evaluation-only mode (`just_eval=true`)
- [ ] Consider adjusting `baseline_min_forget_acc` threshold (or set to 0 to disable)
- [ ] Consider adjusting `batch_size` for model size
- [ ] Consider adjusting learning rates in `types_config` sections
- [ ] Test with a small configuration first (e.g., `testing=true`, `dont_ft=true`, single epoch)

## Checklist: Changing Resources

- [ ] Update `num_gpus` in `conf/default.yaml` (and other configs if not inheriting)
- [ ] Adjust `batch_size` and `val_batch_size` based on GPU memory
- [ ] Set appropriate `attn_backend` for your GPU architecture

---

