# Configuration Guide: Model and Resource Settings

This document lists all files that need to be modified when changing the base model or system resources (GPUs, memory, batch sizes, etc.).

---

## Table of Contents

1. [Changing the Base Model](#changing-the-base-model)
2. [Changing System Resources](#changing-system-resources)
3. [Data Root and Validation](#data-root-and-validation)
4. [Quick Reference Tables](#quick-reference-tables)
5. [Common Configuration Patterns](#common-configuration-patterns)

---

## Changing the Base Model

When switching to a different base model (e.g., from `meta-llama/Meta-Llama-3-8B` to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`), update the following files:

### Configuration Files (`conf/`)

| File | Parameter(s) to Update | Description |
|------|------------------------|-------------|
| `conf/default.yaml` | `model_id`, `ft_model_paths` | Primary config; all other configs inherit from this |
| `conf/lora_smoketest.yaml` | `model_id` | LoRA smoketest configuration |
| `conf/lora_rank_sweep.yaml` | `model_id` | LoRA rank sweep experiments |
| `conf/their_corpus_with_ft.yaml` | `model_id`, `ft_model_paths` | Corpus-based training with fine-tuning |
| `conf/only_ft.yaml` | `model_id`, `ft_model_paths` | Fine-tuning only mode |
| `conf/many_cut_sc.yaml` | `model_id`, `ft_model_paths` | CUT steering coefficient sweeps |
| `conf/no_ft_many_cut_sc.yaml` | `model_id`, `ft_model_paths` | CUT without fine-tuning |
| `conf/letter_unlearn.yaml` | `model_id`, `ft_model_paths` | Letter-based unlearning |
| `conf/ft_on_all.yaml` | `model_id`, `ft_model_paths` | Fine-tuning on all splits |
| `conf/learn_random_bd.yaml` | `model_id`, `ft_model_paths` | Random birthday learning |
| `conf/mcq_format.yaml` | `model_id`, `ft_model_paths` | MCQ format experiments |
| `conf/just_eval.yaml` | `model_id`, `eval_model_paths`, `ft_model_paths` | Evaluation-only mode |
| `conf/random_bd.yaml` | `model_id`, `ft_model_paths` | Random birthday experiments |

### Key Parameters

```yaml
# Primary model identifier (HuggingFace model ID or local path)
model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Models for fine-tuning phase: [[model_path, dataset_name], ...]
ft_model_paths: [["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "MMLU"]]

# Models for evaluation-only mode
eval_model_paths: ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]

# Number of layers (auto-resolved from model config)
num_layers: ${get_num_layers:${model_id}}
```

### Python Files (No Changes Required)

The following Python files dynamically load the model based on config parameters:

| File | Relevant Code | Notes |
|------|---------------|-------|
| `pipeline.py` | `get_num_layers()` resolver | Auto-detects layer count from model config |
| `unlearn_corpus.py` | `AutoModelForCausalLM.from_pretrained()` | Uses `base_model` parameter |
| `finetune_corpus.py` | `AutoModelForCausalLM.from_pretrained()` | Uses `base_model` parameter |
| `conf/test_conf.py` | `get_num_layers()`, `resolve_freeze_layers()` | OmegaConf resolvers |

### Model-Specific Considerations

When changing models, also consider:

1. **Learning rates**: Smaller models may need different learning rate ranges
2. **Batch sizes**: Larger models may require smaller batch sizes
3. **Precision**: Currently uses `bfloat16` for unlearning, `float16` for fine-tuning
4. **Layer freezing**: Update `freeze_layers_coeffs` if using layer freezing

---

## Changing System Resources

### GPU Configuration

| File | Parameter | Description |
|------|-----------|-------------|
| `conf/default.yaml` | `num_gpus` | Number of GPUs for Ray distributed training |
| `conf/lora_smoketest.yaml` | (inherits from default) | |
| `conf/lora_rank_sweep.yaml` | (inherits from default) | |
| `conf/their_corpus_with_ft.yaml` | `num_gpus` | |
| `conf/only_ft.yaml` | `num_gpus` | |
| `conf/many_cut_sc.yaml` | `num_gpus` | |
| `conf/no_ft_many_cut_sc.yaml` | `num_gpus` | |
| `conf/letter_unlearn.yaml` | `num_gpus` | |
| `conf/ft_on_all.yaml` | `num_gpus` | |
| `conf/learn_random_bd.yaml` | `num_gpus` | |
| `conf/mcq_format.yaml` | `num_gpus` | |
| `conf/just_eval.yaml` | `num_gpus` | |
| `conf/random_bd.yaml` | `num_gpus` | |

### Python Files with GPU Settings

| File | Location | Code | Notes |
|------|----------|------|-------|
| `pipeline.py` | Line 137 | `@ray.remote(num_gpus=1)` | Ray remote decorator for unlearn task |
| `pipeline.py` | Line 1106-1107 | `num_gpus = 8 if get_num_gpus() >= 8 else get_num_gpus()` | Ray initialization |
| `unlearn_corpus.py` | Line 860 | `@ray.remote(num_gpus=1)` | Remote unlearn function |
| `unlearn_corpus.py` | Line 946 | `@ray.remote(num_gpus=1)` | Just eval function |
| `finetune_corpus.py` | Line 272 | `@ray.remote(num_gpus=1)` | Fine-tuning remote function |

### Memory & Batch Size Configuration

| File | Parameters | Description |
|------|------------|-------------|
| `conf/default.yaml` | `batch_size`, `val_batch_size` | Training and validation batch sizes |
| All other `conf/*.yaml` files | `batch_size`, `val_batch_size` | May override defaults |

```yaml
# Batch size for training
batch_size: 4

# Batch size for validation/evaluation
val_batch_size: 8

# Warmup steps for learning rate scheduler
warmup_steps: 24
```

### Attention Backend Configuration

| File | Parameter | Description |
|------|-----------|-------------|
| `conf/default.yaml` | `attn_backend` | Attention implementation: `auto`, `flash_attention_2`, `sdpa`, `eager` |

```yaml
# Attention backend options:
# - auto: Use flash_attention_2 if available, otherwise sdpa (default)
# - flash_attention_2: Requires flash-attn package, fastest on Ampere+ GPUs
# - sdpa: PyTorch scaled dot-product attention (good fallback)
# - eager: Standard PyTorch attention (slowest, most compatible)
attn_backend: auto
```

### Data Type / Precision Settings

These are **hardcoded** in Python files and may need manual modification for different GPU architectures:

| File | Line | Setting | Notes |
|------|------|---------|-------|
| `unlearn_corpus.py` | ~513 | `torch.bfloat16` | Model loading precision |
| `finetune_corpus.py` | ~321 | `torch.float16` | Model loading precision |

---

## Data Root and Validation

### Data Root
By default the pipeline reads from `data/`. Override via:
- Environment: `UNLEARN_DATA_ROOT=/path/to/data`
- Hydra: `data_root=/path/to/data`

### Materialize + Validate
Use these helpers before running the pipeline:
```
python scripts/materialize_data.py datasets=[YEARS]
python scripts/check_data.py datasets=[YEARS]
```

---

## Quick Reference Tables

### All Configuration Files Summary

| Config File | Primary Use Case | Inherits From |
|-------------|------------------|---------------|
| `default.yaml` | Base configuration | — |
| `lora_smoketest.yaml` | Quick LoRA testing | `default.yaml` |
| `lora_rank_sweep.yaml` | LoRA rank experiments | `default.yaml` |
| `their_corpus_with_ft.yaml` | External corpus experiments | — |
| `only_ft.yaml` | Fine-tuning only | — |
| `many_cut_sc.yaml` | CUT method experiments | — |
| `no_ft_many_cut_sc.yaml` | CUT without fine-tuning | — |
| `letter_unlearn.yaml` | Letter-based unlearning | — |
| `ft_on_all.yaml` | Full dataset fine-tuning | — |
| `learn_random_bd.yaml` | Random birthday learning | — |
| `mcq_format.yaml` | MCQ format experiments | — |
| `just_eval.yaml` | Evaluation only | — |
| `random_bd.yaml` | Random birthday unlearning | — |

### Resource Parameters Quick Reference

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `num_gpus` | `conf/*.yaml` | 1 | Number of GPUs for training |
| `batch_size` | `conf/*.yaml` | 4 | Training batch size |
| `val_batch_size` | `conf/*.yaml` | 8 | Validation batch size |
| `warmup_steps` | `conf/*.yaml` | 24 | LR scheduler warmup |
| `attn_backend` | `conf/*.yaml` | `auto` | Attention implementation |
| `data_seed` | `conf/*.yaml` | 4 | Random seed for data |
| `data_root` | `conf/*.yaml` | `data` | Data root override |

---

## Common Configuration Patterns

### Single GPU Setup (Recommended for Development)

```yaml
num_gpus: 1
batch_size: 4
val_batch_size: 8
attn_backend: auto
```

### Multi-GPU Setup (8 GPUs)

```yaml
num_gpus: 8
batch_size: 4  # Per GPU
val_batch_size: 8
attn_backend: flash_attention_2  # Recommended for A100/H100
```

### Memory-Constrained Setup

```yaml
num_gpus: 1
batch_size: 2
val_batch_size: 4
attn_backend: sdpa  # More memory-efficient than eager
```

### Quick Smoke Test

```yaml
# Use lora_smoketest.yaml as base
testing: true
dont_ft: true
batch_size: 4
val_batch_size: 8
unlearn:
  max_samples_lst: [32]  # Limit samples for fast testing
  types_config:
    LORA:
      datasets_config:
        MMLU:
          epochs_lst: [1]
          rcs:
            add: [1]
```

---

## CLI Override Examples

```bash
# Change model via CLI
python pipeline.py model_id="meta-llama/Meta-Llama-3-8B"

# Change GPU count
python pipeline.py num_gpus=4

# Change batch size
python pipeline.py batch_size=2 val_batch_size=4

# Multiple overrides
python pipeline.py \
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    num_gpus=1 \
    batch_size=4 \
    attn_backend=sdpa
```

---

## Checklist: Changing Model

- [ ] Update `model_id` in `conf/default.yaml`
- [ ] Update `ft_model_paths` in all relevant config files
- [ ] Update `eval_model_paths` if using evaluation-only mode
- [ ] Consider adjusting `batch_size` for model size
- [ ] Consider adjusting learning rates in `types_config` sections
- [ ] Test with `--config-name=lora_smoketest` first

## Checklist: Changing Resources

- [ ] Update `num_gpus` in `conf/default.yaml` (and other configs if not inheriting)
- [ ] Adjust `batch_size` and `val_batch_size` based on GPU memory
- [ ] Set appropriate `attn_backend` for your GPU architecture
- [ ] If modifying Python files, update `@ray.remote(num_gpus=N)` decorators
- [ ] For multi-GPU, ensure Ray is properly configured in `pipeline.py`

---
