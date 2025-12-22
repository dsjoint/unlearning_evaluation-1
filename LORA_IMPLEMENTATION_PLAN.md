# LoRA Unlearning Implementation Plan

This document provides a Codex-ready implementation plan for adding LoRA-based unlearning to the existing pipeline.

---

## 1. Repo Touchpoints (with Evidence)

### 1.1 Unlearn Type Enumeration
- **File**: `pipeline.py:26-31`
- **Code**: `UnlearnType` enum defines `CUT`, `GD`, `WHP`, `FWF`, `NOT_SPECIFIED`
- **Change needed**: Add `LORA = auto()`

### 1.2 Method Dispatch (Router)
- **File**: `pipeline.py:136-257`
- **Code**: `@ray.remote def unlearn(...)` routes based on `unlearn_type.value`
  - Lines 169-212: GD/WHP/FWF → `unlearn_corpus.main()`
  - Lines 214-245: CUT → `rmu.unlearn_pipeline.main()`
- **Change needed**: Add new branch for `UnlearnType.LORA` calling modified `unlearn_corpus.main()` with LoRA params

### 1.3 Training Loop
- **File**: `unlearn_corpus.py:729-783`
- **Code**: Training loop with Lion optimizer, `forget_loss + retain_coeff * retain_loss` objective
- **Change needed**: When LoRA is enabled, optimize only LoRA parameters

### 1.4 Model Loading
- **File**: `unlearn_corpus.py:507-509`
- **Code**:
```python
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
).to(device)
```
- **Change needed**: After loading, wrap with `get_peft_model()` when LoRA is enabled

### 1.5 Model Saving
- **File**: `unlearn_corpus.py:787-789`
- **Code**:
```python
if save_name is not None:
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
```
- **Change needed**: Before saving, merge LoRA weights into base model so RTT can load standard HF checkpoint

### 1.6 Hyperparameter Grid Expansion
- **File**: `pipeline.py:1159-1170`
- **Code**: Nested loops over `max_samples_lst`, `epochs_lst`, `lrs`, `rcs`, `scs` (for CUT only)
- **Pattern for LORA**: Add `lora_ranks` loop similar to `scs` loop (lines 1163-1170)

### 1.7 Config Structure
- **File**: `conf/default.yaml:29-246`
- **Pattern**: Each method has `unlearn.types_config.{METHOD}.loss_type` + `unlearn.types_config.{METHOD}.datasets_config.{DATASET}` with hyperparameter lists
- **Change needed**: Add `unlearn.types_config.LORA` section with `ranks` list + standard per-dataset config

### 1.8 Metrics Output (Unlearning)
- **File**: `pipeline.py:400-455`
- **Output path**: `evals/pipeline/unlearning/{timestamp}--num{i}.csv`
- **Fields include**: `unlearn_type`, `forget_accs`, `retain_accs` (dict with per-epoch values)
- **Change needed**: Add `lora_rank` to metrics dict

### 1.9 Metrics Output (RTT/Fine-tuning)
- **File**: `pipeline.py:556-600` and `finetune_corpus.py:480-502`
- **Output path**: `evals/pipeline/ft/{timestamp}--num{i}.csv`
- **Fields include**: `forget_accs_local` (dict with per-epoch accuracy values), `base_model`
- **No change needed** — existing RTT metrics suffice for recovery rate analysis

---

## 2. Minimal Design Decisions

### 2.1 Model Artifact Format: **Merge adapters then save**
**Rationale**: The RTT phase (`finetune_corpus.py:317-319`) loads models with:
```python
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
).to(device)
```
This expects a standard HuggingFace checkpoint. Saving adapter-only would require modifying RTT loading code. Merging LoRA weights before saving (`model.merge_and_unload()`) produces a standard checkpoint with **zero changes to downstream code**.

### 2.2 LoRA Objective: **Same as GD (gradient ascent on forget + descent on retain)**
**Evidence**: `unlearn_corpus.py:751-752`
```python
forget_loss = get_loss(..., unlearn_type=unlearn_type, ...)
retain_loss = get_loss(..., unlearn_type=UnlearnType.FWF, ...)  # normal descent
```
For GD, `get_loss_corpus()` (`unlearn_corpus.py:216-217`) returns `-original_loss` (gradient ascent).

LoRA unlearning will use the same objective (via `UnlearnType.GD` routing) but train only LoRA parameters.

### 2.3 LoRA Target Modules: **All linear projections**
Target `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` — standard for LLaMA/Mistral family. This matches typical LoRA fine-tuning.

### 2.4 Config Surface: **Only `ranks` list**
Following principle of minimal change:
- Add `unlearn.lora_ranks: [1, 2, 4, 8, 16, 32]`
- Add `unlearn.types_config.LORA` section mirroring other methods

### 2.5 Library: **peft**
PEFT is the standard library for LoRA. Add `peft>=0.12.0` to `requirements.txt`.

---

## 3. Step-by-Step Codex Task List

### Task 1: Add PEFT dependency
**Goal**: Enable LoRA via Hugging Face PEFT library.

**Files**: `requirements.txt`

**Exact edits**:
Add after the last line:
```
peft>=0.12.0
```

**Verification**: 
```bash
pip install -r requirements.txt && python -c "from peft import get_peft_model, LoraConfig; print('OK')"
```

---

### Task 2: Add `UnlearnType.LORA` enum value
**Goal**: Register LORA as a valid unlearning method.

**Files**: `pipeline.py`

**Exact edits** — modify the `UnlearnType` enum (lines 26-31):

**Before**:
```python
class UnlearnType(Enum):
    CUT = auto() # CUT/RMU Li et al 2024
    GD = auto()  # Gradiend Difference
    WHP = auto() # Random Incorrect Fact
    FWF = auto() # Fixed Incorrect Fact
    NOT_SPECIFIED = auto()
```

**After**:
```python
class UnlearnType(Enum):
    CUT = auto() # CUT/RMU Li et al 2024
    GD = auto()  # Gradiend Difference
    WHP = auto() # Random Incorrect Fact
    FWF = auto() # Fixed Incorrect Fact
    LORA = auto()  # LoRA-based unlearning
    NOT_SPECIFIED = auto()
```

**Verification**: 
```bash
python -c "from pipeline import UnlearnType; print(UnlearnType.LORA)"
```

---

### Task 3: Add `lora_rank` parameter to `unlearn()` function and add LORA dispatch branch
**Goal**: Route LORA method to `unlearn_corpus.main()` with LoRA parameters.

**Files**: `pipeline.py`

**Part A: Update function signature** — add `lora_rank` parameter after `max_samples` (around line 164):

**Before**:
```python
@ray.remote(num_gpus=1)
def unlearn(
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    unlearn_files: list[str] = [],
    wrong_unlearn_files: list[str] = [],
    fixed_wrong_unlearn_files: list[str] = [],
    val_files: list[str] = [],
    dev_file: str = "",
    retain_files: list[str] = [],
    val_retain_files: list[str] = [],
    retain_dev_file: str = "",
    base_model: str = "",
    lr: float = 1e-7,
    epochs: int = 3,
    batch_size: int = 4,
    val_batch_size: int = 8,
    retain_coeff: int = 1,
    warmup_steps: int = 24,
    data_seed: int = 0,
    eval_every: int = 1,
    save_name: Optional[str] = None,
    wandb_project_name: str = "unlearn",
    unlearn_freeze_layers: Optional[list[tuple[int, int]]] = None,
    mcq: bool = False,
    hydra_dict: dict = {},
    data_format: DataFormat = DataFormat.CORPUS,
    loss_type: LossType = LossType.CORPUS,
    steering_coeff: float = 20,
    max_samples: int = None,
):
```

**After**:
```python
@ray.remote(num_gpus=1)
def unlearn(
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    unlearn_files: list[str] = [],
    wrong_unlearn_files: list[str] = [],
    fixed_wrong_unlearn_files: list[str] = [],
    val_files: list[str] = [],
    dev_file: str = "",
    retain_files: list[str] = [],
    val_retain_files: list[str] = [],
    retain_dev_file: str = "",
    base_model: str = "",
    lr: float = 1e-7,
    epochs: int = 3,
    batch_size: int = 4,
    val_batch_size: int = 8,
    retain_coeff: int = 1,
    warmup_steps: int = 24,
    data_seed: int = 0,
    eval_every: int = 1,
    save_name: Optional[str] = None,
    wandb_project_name: str = "unlearn",
    unlearn_freeze_layers: Optional[list[tuple[int, int]]] = None,
    mcq: bool = False,
    hydra_dict: dict = {},
    data_format: DataFormat = DataFormat.CORPUS,
    loss_type: LossType = LossType.CORPUS,
    steering_coeff: float = 20,
    max_samples: int = None,
    lora_rank: int = 0,
):
```

**Part B: Add LORA dispatch branch** — insert after the GD/WHP/FWF block (after line 212) and before the CUT block (line 214):

Insert this code after line 212 (after the closing parenthesis of the GD/WHP/FWF return):
```python
    elif unlearn_type.value == UnlearnType.LORA.value:
        import unlearn_corpus
        (
            model_path,
            forget_accs, forget_accs_calibrated, forget_logits_dict,
            retain_accs, retain_accs_calibrated, retain_logits_dict,
            retain_accs_5_shot, retain_accs_5_shot_calibrated,
            retain_logits_5_shot_dict,
            samples
        ) = (
            unlearn_corpus.main(
                unlearn_type=UnlearnType.GD,  # Use GD objective (gradient ascent on forget)
                train_files=unlearn_files,
                wrong_unlearn_files=wrong_unlearn_files,
                fixed_wrong_unlearn_files=fixed_wrong_unlearn_files,
                val_files=val_files,
                dev_set=dev_file,
                retain_files=retain_files,
                val_retain_files=val_retain_files,
                retain_dev_file=retain_dev_file,
                base_model=base_model,
                lr=lr,
                name=save_name,
                epochs=epochs,
                batch_size=batch_size,
                val_batch_size=val_batch_size,
                retain_coeff=retain_coeff,
                warmup_steps=warmup_steps,
                data_seed=data_seed,
                eval_every=eval_every,
                save_name=save_name,
                project_name=wandb_project_name,
                freeze_layers=unlearn_freeze_layers,
                mcq=mcq,
                hydra_dict=hydra_dict,
                data_format=data_format,
                loss_type=loss_type,
                max_samples=max_samples,
                lora_rank=lora_rank,
            )
        )
```

**Verification**: `python -m py_compile pipeline.py`

---

### Task 4: Add `lora_rank` parameter to `main()` remote function in `pipeline.py`
**Goal**: Pass `lora_rank` through the pipeline orchestration layer.

**Files**: `pipeline.py`

**Part A: Update function signature** — add `lora_rank` to `main()` (around line 308):

Find the `main()` function signature (starts around line 261) and add `lora_rank: int = 0,` after `max_samples: int = 9999999999,`:

**Before** (line ~308):
```python
    max_samples: int = 9999999999, # limit number of datapoints for unlearning
):
```

**After**:
```python
    max_samples: int = 9999999999, # limit number of datapoints for unlearning
    lora_rank: int = 0,  # LoRA rank (0 = disabled)
):
```

**Part B: Update the call to `unlearn.remote()`** — add `lora_rank` parameter (around line 381):

Find the `unlearn.remote()` call inside `main()` and add `lora_rank=lora_rank,` after `max_samples=max_samples,`:

**Before** (around line 381):
```python
                ref = unlearn.remote(
                    unlearn_type=unlearn_type,
                    unlearn_files=unlearn_files,
                    wrong_unlearn_files=wrong_unlearn_files,
                    fixed_wrong_unlearn_files=fixed_wrong_unlearn_files,
                    val_files=val_files,
                    dev_file=dev_file,
                    retain_files=retain_files,
                    val_retain_files=val_retain_files,
                    retain_dev_file=retain_dev_file,
                    base_model=base_model,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    retain_coeff=retain_coeff,
                    warmup_steps=warmup_steps,
                    data_seed=data_seed,
                    eval_every=eval_every,
                    save_name=save_name,
                    wandb_project_name=wandb_project_name,
                    unlearn_freeze_layers=unlearn_freeze_layers,
                    mcq=unlearn_mcq,
                    hydra_dict=hydra_dict,
                    data_format=unlearn_data_format,
                    loss_type=unlearn_loss_type,
                    steering_coeff=steering_coeff,
                    max_samples=max_samples,
                )
```

**After**:
```python
                ref = unlearn.remote(
                    unlearn_type=unlearn_type,
                    unlearn_files=unlearn_files,
                    wrong_unlearn_files=wrong_unlearn_files,
                    fixed_wrong_unlearn_files=fixed_wrong_unlearn_files,
                    val_files=val_files,
                    dev_file=dev_file,
                    retain_files=retain_files,
                    val_retain_files=val_retain_files,
                    retain_dev_file=retain_dev_file,
                    base_model=base_model,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    retain_coeff=retain_coeff,
                    warmup_steps=warmup_steps,
                    data_seed=data_seed,
                    eval_every=eval_every,
                    save_name=save_name,
                    wandb_project_name=wandb_project_name,
                    unlearn_freeze_layers=unlearn_freeze_layers,
                    mcq=unlearn_mcq,
                    hydra_dict=hydra_dict,
                    data_format=unlearn_data_format,
                    loss_type=unlearn_loss_type,
                    steering_coeff=steering_coeff,
                    max_samples=max_samples,
                    lora_rank=lora_rank,
                )
```

---

### Task 5: Implement LoRA training in `unlearn_corpus.py`
**Goal**: Add LoRA adapter initialization, training, and merge-before-save logic.

**Files**: `unlearn_corpus.py`

**Part A: Add PEFT import** — add after line 14 (after `from pipeline import UnlearnType, LossType, DataFormat`):
```python
from peft import get_peft_model, LoraConfig, TaskType
```

**Part B: Add `lora_rank` parameter to `main()` function** — modify function signature (around line 482):

Find the `main()` function signature and add `lora_rank: int = 0,` after `loss_type: LossType = LossType.NOT_SPECIFIED,`:

**Before**:
```python
def main(
    train_files: list[str],
    wrong_unlearn_files: list[str],
    fixed_wrong_unlearn_files: list[str],
    val_files: list[str],
    dev_set: str,
    base_model: str,
    lr: float,
    name: str,
    k_shot: int = 0,
    epochs: int = 10,
    batch_size: int = 4,
    val_batch_size: int = 8,
    warmup_steps: int = 24,
    retain_files: list[str] = [],
    val_retain_files: list[str] = [],
    retain_dev_file: str = "",
    max_samples: Optional[int] = None,
    data_seed: int = 2,
    eval_every: int = 1,
    keep_set: Optional[int] = None,
    keep_set_weight: Optional[float] = None,
    train_on_wrong_answer: bool = False,
    train_set_size: Optional[int] = None,
    val_set_size: Optional[int] = None,
    kind: str = "base",
    save_name: Optional[str] = None,
    version: str = "v2.11",
    model = None,
    retain_coeff: int = 1,
    project_name: str = "unlearn",
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    results_file: str = None,
    just_eval: bool = False,
    disable_wandb: bool = False,
    freeze_layers: Optional[list[tuple[int, int]]] = None,
    mcq: bool = False,
    hydra_dict: dict = {},
    data_format: DataFormat = DataFormat.NOT_SPECIFIED,
    loss_type: LossType = LossType.NOT_SPECIFIED,
):
```

**After**:
```python
def main(
    train_files: list[str],
    wrong_unlearn_files: list[str],
    fixed_wrong_unlearn_files: list[str],
    val_files: list[str],
    dev_set: str,
    base_model: str,
    lr: float,
    name: str,
    k_shot: int = 0,
    epochs: int = 10,
    batch_size: int = 4,
    val_batch_size: int = 8,
    warmup_steps: int = 24,
    retain_files: list[str] = [],
    val_retain_files: list[str] = [],
    retain_dev_file: str = "",
    max_samples: Optional[int] = None,
    data_seed: int = 2,
    eval_every: int = 1,
    keep_set: Optional[int] = None,
    keep_set_weight: Optional[float] = None,
    train_on_wrong_answer: bool = False,
    train_set_size: Optional[int] = None,
    val_set_size: Optional[int] = None,
    kind: str = "base",
    save_name: Optional[str] = None,
    version: str = "v2.11",
    model = None,
    retain_coeff: int = 1,
    project_name: str = "unlearn",
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    results_file: str = None,
    just_eval: bool = False,
    disable_wandb: bool = False,
    freeze_layers: Optional[list[tuple[int, int]]] = None,
    mcq: bool = False,
    hydra_dict: dict = {},
    data_format: DataFormat = DataFormat.NOT_SPECIFIED,
    loss_type: LossType = LossType.NOT_SPECIFIED,
    lora_rank: int = 0,
):
```

**Part C: Add LoRA initialization** — insert after the `freeze_model_layers` call (after line 512):

Find this code block:
```python
    if freeze_layers is not None:
        freeze_model_layers(model, freeze_layers)

    optimizer = Lion(model.parameters(), lr=lr, use_triton=True)
```

Replace with:
```python
    if freeze_layers is not None:
        freeze_model_layers(model, freeze_layers)

    # LoRA setup
    if lora_rank > 0:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,  # Standard scaling
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Log trainable param count

    # Only optimize trainable parameters (LoRA params if lora_rank > 0, all params otherwise)
    optimizer = Lion(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        use_triton=True
    )
```

**Part D: Modify model saving to merge LoRA weights** — modify lines 787-789:

Find:
```python
    if save_name is not None:
        model.save_pretrained(save_name)
        tokenizer.save_pretrained(save_name)
```

Replace with:
```python
    if save_name is not None:
        # Merge LoRA weights into base model if applicable
        if lora_rank > 0:
            model = model.merge_and_unload()
        model.save_pretrained(save_name)
        tokenizer.save_pretrained(save_name)
```

---

### Task 6: Add LoRA hyperparameter grid in pipeline's `run_pipeline()`
**Goal**: Sweep over LoRA ranks similar to how `cut_scs` is swept for CUT.

**Files**: `pipeline.py`

**Part A: Add config reading** — insert after line 1111 (after `max_samples_lst` reading):

Find:
```python
        max_samples_lst = OmegaConf.select(
            cfg, "unlearn.max_samples_lst", default=[999999999]
        )
```

Add after:
```python
        lora_ranks = OmegaConf.select(
            cfg, "unlearn.lora_ranks", default=[0]
        )
```

**Part B: Update hyperparameter loop** — modify the nested loop structure (around lines 1159-1248):

Find the loop structure:
```python
                    for max_samples in max_samples_lst:
                        for epochs in epochs_lst:
                            for lr in lrs:
                                for rc in rcs:
                                    scs = (
                                        cut_scs
                                        if (
                                            unlearn_type.value
                                            == UnlearnType.CUT.value
                                        ) else [20]
                                    )
                                    for sc in scs: #!
                                        forget_model = (
                                            f"models/{unlearn_type.name}/"
                                            f"{dataset.name}/"
                                            f"{wandb_project_name}/{sc=}" #!
                                            f"{model_id}-rc{rc}-lr{lr}-"
                                            f"epochs{epochs}"
                                        )
```

Replace with:
```python
                    for max_samples in max_samples_lst:
                        for epochs in epochs_lst:
                            for lr in lrs:
                                for rc in rcs:
                                    scs = (
                                        cut_scs
                                        if (
                                            unlearn_type.value
                                            == UnlearnType.CUT.value
                                        ) else [20]
                                    )
                                    ranks = (
                                        lora_ranks
                                        if (
                                            unlearn_type.value
                                            == UnlearnType.LORA.value
                                        ) else [0]
                                    )
                                    for sc in scs:
                                        for lora_rank in ranks:
                                            forget_model = (
                                                f"models/{unlearn_type.name}/"
                                                f"{dataset.name}/"
                                                f"{wandb_project_name}/"
                                                f"rank{lora_rank}-sc{sc}-"
                                                f"{model_id}-rc{rc}-lr{lr}-"
                                                f"epochs{epochs}"
                                            )
```

**Part C: Add `lora_rank` to the `main.remote()` call** — add parameter (around line 1247):

Find the `main.remote()` call and add `lora_rank=lora_rank,` after `max_samples=max_samples,`:

The call currently ends with:
```python
                                            steering_coeff=sc, 
                                            max_samples=max_samples,
                                        )]
```

Change to:
```python
                                            steering_coeff=sc, 
                                            max_samples=max_samples,
                                            lora_rank=lora_rank,
                                        )]
```

---

### Task 7: Add `lora_rank` to metrics output
**Goal**: Include `lora_rank` in CSV metrics for analysis.

**Files**: `pipeline.py`

**Update metrics dict** — find the metrics dict (around line 400-441) and add `lora_rank`:

Find:
```python
            metrics = {
                "model_path": name,
                "dataset": dataset.name,
                # ... many fields ...
                "steering_coeff": steering_coeff,
                "max_samples": max_samples,
            }
```

Add `"lora_rank": lora_rank,` after `"max_samples": max_samples,`:

```python
            metrics = {
                "model_path": name,
                "dataset": dataset.name,
                # ... many fields ...
                "steering_coeff": steering_coeff,
                "max_samples": max_samples,
                "lora_rank": lora_rank,
            }
```

---

### Task 8: Create LORA config section in `conf/default.yaml`
**Goal**: Add configuration for LORA method with rank sweep.

**Files**: `conf/default.yaml`

**Part A: Add `lora_ranks` list** — insert after line 32 (after `cut_scs: [0.1, 1, 10]`):

Find:
```yaml
unlearn:
  types: [GD]
  many_cut_sc: true
  cut_scs: [0.1, 1, 10]
  save_unlearn_model: true
```

Change to:
```yaml
unlearn:
  types: [GD]
  many_cut_sc: true
  cut_scs: [0.1, 1, 10]
  lora_ranks: [1, 2, 4, 8, 16, 32]
  save_unlearn_model: true
```

**Part B: Add LORA types_config section** — insert after the FWF section (after line 246, before `ft:`):

Add:
```yaml
    LORA:
      loss_type: CORPUS
      datasets_config:
        YEARS:
          epochs_lst: [5]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-3, 1e3, 10}
            add: [0, 2, 4]
        YEARS_MMLU_RETAIN:
          epochs_lst: [5]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-3, 1e3, 10}
            add: [0, 2, 4]
        MMLU:
          epochs_lst: [5]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-3, 1e3, 10}
            add: [0, 2, 4]
        WMDP_MCQ_CORPUS:
          epochs_lst: [5]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-3, 1e3, 10}
            add: [0, 2, 4]
        WMDP_MCQ_CORPUS_FINEWEB:
          epochs_lst: [5]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-3, 1e3, 10}
            add: [0, 2, 4]
        RANDOM_BD:
          epochs_lst: [5]
          lrs: [4e-7]
          rcs:
            range: ${get_log_range:1e-3, 1e3, 10}
            add: [0, 2, 4]
```

---

### Task 9: Update `remote_main` and `just_eval` signatures in `unlearn_corpus.py`
**Goal**: Ensure LoRA parameter is passed through all entry points.

**Files**: `unlearn_corpus.py`

**Part A: Update `remote_main()` signature and call** (around lines 827-909):

Add `lora_rank: int = 0,` to the function signature after `loss_type`:
```python
@ray.remote(num_gpus=1)
def remote_main(
    # ... existing params ...
    loss_type: LossType = LossType.NOT_SPECIFIED,
    lora_rank: int = 0,
):
```

And add `lora_rank=lora_rank,` to the `main()` call at the end.

**Part B: Update `just_eval()` signature** (around lines 911-994):

Add `lora_rank: int = 0,` to keep the interface consistent (won't be used in eval).

---

### Task 10: Create new config files

**File**: `conf/lora_rank_sweep.yaml`
```yaml
defaults:
  - default
  - _self_

# LoRA rank sweep experiment
just_eval: false
dont_ft: false
testing: false

model_id: "HuggingFaceH4/zephyr-7b-beta"
datasets: [MMLU]
wandb_project_name: "lora_rank_sweep"

unlearn:
  types: [LORA]
  lora_ranks: [1, 2, 4, 8, 16, 32]
  save_unlearn_model: true

ft:
  num_splits: 2
  loss_types: [QUESTION_LETTER_ANSWER]
  epochs_lst: [6]
  lrs: ${get_log_range:1e-7,5e-6,2}
  save_models: false
```

**File**: `conf/lora_smoketest.yaml`
```yaml
defaults:
  - default
  - _self_

# Minimal smoketest config (runs in minutes)
just_eval: false
dont_ft: true
testing: true

model_id: "HuggingFaceH4/zephyr-7b-beta"
datasets: [MMLU]
wandb_project_name: "lora_smoketest"

unlearn:
  types: [LORA]
  lora_ranks: [4]
  max_samples_lst: [32]
  save_unlearn_model: true
  types_config:
    LORA:
      loss_type: CORPUS
      datasets_config:
        MMLU:
          epochs_lst: [1]
          lrs: [4e-7]
          rcs:
            range: []
            add: [1]

batch_size: 4
val_batch_size: 8
eval_every: 1
```

---

### Task 11: Create analysis script (Optional)

**File**: `analyze_recovery.py`
```python
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
    except:
        return -1

def main():
    import pandas as pd
    
    # Read all FT CSVs
    ft_files = glob.glob("evals/pipeline/ft/*.csv")
    if not ft_files:
        print("No FT CSV files found in evals/pipeline/ft/")
        sys.exit(1)
    
    rows = []
    for f in ft_files:
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                base_model = str(row.get('base_model', ''))
                if 'LORA' in base_model or 'rank' in base_model:
                    rank = parse_rank_from_path(base_model)
                    if rank > 0:
                        accs = row.get('forget_accs_local', '{}')
                        rec = recovery_epoch(accs, threshold=0.8)
                        if rec >= 0:
                            rows.append({'rank': rank, 'recovery_epoch': rec})
        except Exception as e:
            print(f"Warning: Error reading {f}: {e}")
    
    if not rows:
        print("No LORA experiments found in FT CSVs")
        sys.exit(1)
    
    result = pd.DataFrame(rows)
    summary = result.groupby('rank')['recovery_epoch'].agg(['mean', 'std', 'count'])
    print("\nRecovery Epoch by LoRA Rank:")
    print("=" * 50)
    print(summary.to_string())
    print()
    
    summary.to_csv("recovery_vs_rank.csv")
    print("Saved to recovery_vs_rank.csv")

if __name__ == "__main__":
    main()
```

---

## 4. Verification Commands

### 4.1 Syntax Check
```bash
python -m py_compile pipeline.py
python -m py_compile unlearn_corpus.py
```

### 4.2 Import Check
```bash
python -c "from pipeline import UnlearnType; print(UnlearnType.LORA)"
python -c "from peft import get_peft_model, LoraConfig; print('PEFT OK')"
```

### 4.3 Smoketest
```bash
python pipeline.py --config-name=lora_smoketest
```

Expected:
- Prints `trainable params: X || all params: Y || trainable%: Z%`
- Creates model in `models/LORA/MMLU/lora_smoketest/...`
- Creates CSV in `evals/pipeline/unlearning/...`

### 4.4 Verify Saved Model is Standard HF Format
```bash
python -c "
from transformers import AutoModelForCausalLM
import glob
paths = glob.glob('models/LORA/MMLU/lora_smoketest/*')
if paths:
    m = AutoModelForCausalLM.from_pretrained(paths[0])
    print(f'Loaded {paths[0]} successfully')
else:
    print('No model found')
"
```

---

## 5. Experiment Commands

### 5.1 Full Rank Sweep on MMLU
```bash
python pipeline.py --config-name=lora_rank_sweep \
    datasets=[MMLU] \
    unlearn.lora_ranks=[1,2,4,8,16,32] \
    wandb_project_name="lora_rank_sweep_mmlu"
```

### 5.2 Rank Sweep on YEARS Dataset
```bash
python pipeline.py --config-name=lora_rank_sweep \
    datasets=[YEARS] \
    unlearn.lora_ranks=[1,2,4,8,16,32,64] \
    wandb_project_name="lora_rank_sweep_years"
```

### 5.3 Single Rank with RTT
```bash
python pipeline.py \
    datasets=[MMLU] \
    unlearn.types=[LORA] \
    unlearn.lora_ranks=[8] \
    dont_ft=false \
    wandb_project_name="lora_r8_with_rtt"
```

---

## 6. Risks and Gotchas

### 6.1 Memory Overhead
LoRA adds minimal parameters (~0.4% of base for rank=32). RTX 5090 (32GB) handles this easily.

### 6.2 Target Modules
The target modules assume LLaMA/Mistral architecture. Zephyr-7B is LLaMA-based, so this is fine.

### 6.3 Merge and Unload
`model.merge_and_unload()` modifies weights in-place. Only call before final save.

### 6.4 LoRA Alpha Scaling
Using `lora_alpha = 2 * lora_rank`. If unstable at high ranks, consider fixing `lora_alpha=16`.

---

## 7. Recovery Rate Definition

### From Existing Outputs:
- **Unlearning CSV** (`evals/pipeline/unlearning/*.csv`): Contains `lora_rank`, `forget_accs`
- **RTT CSV** (`evals/pipeline/ft/*.csv`): Contains `forget_accs_local`, `base_model` (path includes `rank{X}`)

### Recovery Rate = First RTT epoch where `forget_accs_local[epoch] >= threshold`

Use `analyze_recovery.py` to extract this from existing CSVs.

---

## Summary of Files to Create/Modify

| File | Action |
|------|--------|
| `requirements.txt` | Add `peft>=0.12.0` |
| `pipeline.py` | Add LORA enum, dispatch, grid loop, metrics |
| `unlearn_corpus.py` | Add PEFT import, LoRA init, merge-before-save |
| `conf/default.yaml` | Add `lora_ranks` and `LORA` types_config |
| `conf/lora_rank_sweep.yaml` | Create (new file) |
| `conf/lora_smoketest.yaml` | Create (new file) |
| `analyze_recovery.py` | Create (new file, optional) |

