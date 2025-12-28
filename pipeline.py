import os
import sys
from typing import Optional, Dict, Any
from enum import Enum, auto
import logging
import ray
import datetime
from ray.experimental.tqdm_ray import tqdm
from ray.experimental import tqdm_ray
import requests
import traceback
import builtins
import io
import time
import threading
from zoneinfo import ZoneInfo
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from dotenv import load_dotenv
from data.requirements import resolve_dataset_path
from data.validate_data import validate_required_artifacts
import json
import tempfile
from filelock import FileLock, Timeout

# class definitions for hyperparameter configurations
# Force saving all A/B/C checkpoints regardless of config flags.
FORCE_SAVE_ALL_MODELS = True

class UnlearnType(Enum):
    CUT = auto() # CUT/RMU Li et al 2024
    GD = auto()  # Gradiend Difference
    WHP = auto() # Random Incorrect Fact
    FWF = auto() # Fixed Incorrect Fact
    LORA = auto()  # LoRA-based unlearning
    NOT_SPECIFIED = auto()

class LossType(Enum):
    LETTER = auto() # takes loss on the letter representing the answer for MCQ
    CORPUS = auto() # takes loss on all the tokens
    LETTER_ANSWER = auto() # takes loss on the letter and the answer for MCQ
    QUESTION_LETTER_ANSWER = auto()
    QUESTION_ANSWER = auto()
    NUMBER = auto() # only takes loss on tokens that contain numbers
    NOT_SPECIFIED = auto()

class Datasets(Enum):
    YEARS = auto()
    YEARS_TF = auto()
    MMLU = auto()
    WMDP_CORPUS = auto()
    WMDP_CORPUS_FINEWEB = auto()
    WMDP_CORPUS_MMLU = auto()
    WMDP_MCQ_CORPUS = auto()
    WMDP_MCQ_CORPUS_FINEWEB = auto()
    WMDP_MCQ_FINEWEB = auto()
    WMDP_MCQ_WIKITEXT = auto()
    WMDP_MCQ_LETTER_ANSWER = auto()
    BEAVERTAILS = auto()
    RANDOM_BD = auto()
    RANDOM_BD_SAME_RETAIN = auto()
    RANDOM_BD_ALL_SPLITS = auto()
    RANDOM_BD_WITH_MMLU = auto()
    RANDOM_BD_WITH_MMLU_CORPUS = auto()
    YEARS_MMLU_RETAIN = auto()
    DAY_OF_THE_MONTH = auto()
    NOT_SPECIFIED = auto()

class DataFormat(Enum):
    CORPUS = auto() # plain-text
    MCQ = auto()
    TF = auto() # true or false
    NOT_SPECIFIED = auto()


# helpers
raise_exceptions = False

def get_current_time(timezone="America/Los_Angeles"):
    return datetime.datetime.now(ZoneInfo(timezone))

def emit_terminal_notice(message: str) -> None:
    output = f"\n{message}"
    print(output, flush=True)
    try:
        with open("/dev/tty", "w") as tty:
            tty.write(output + "\n")
            tty.flush()
    except Exception:
        pass

def is_after_6pm():
    current_time = get_current_time().time()
    return current_time >= datetime.time(18, 0)

def get_runtime_cwd():
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.cwd
    except Exception:
        return os.getcwd()

def get_data_root(cfg: DictConfig) -> str:
    data_root = os.getenv("UNLEARN_DATA_ROOT")
    if not data_root:
        data_root = OmegaConf.select(cfg, "data_root") or "data"
    if not os.path.isabs(data_root):
        data_root = os.path.join(get_runtime_cwd(), data_root)
    return data_root

def resolve_dataset_dict_paths(dataset_dict: dict, data_root: str) -> dict:
    resolved = dict(dataset_dict)
    list_keys = [
        "unlearn_files",
        "wrong_unlearn_files",
        "fixed_wrong_unlearn_files",
        "val_files",
        "retain_files",
        "val_retain_files",
    ]
    for key in list_keys:
        if key in resolved:
            resolved[key] = [
                resolve_dataset_path(path, data_root)
                for path in resolved.get(key, [])
            ]
    str_keys = ["dev_file", "retain_dev_file"]
    for key in str_keys:
        if key in resolved and resolved[key]:
            resolved[key] = resolve_dataset_path(resolved[key], data_root)
    return resolved

# converts a nested dictionary into a flat one
def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
# returns a list `l` such that l[i+1] = l[i] * step
def get_log_range(start, end, step):
    curr = start
    its = []
    while curr < end:
        its += [curr]
        curr *= step
    return its

# allows for use of `get_log_range()` in hydra config files
# Guard against re-registration when module is re-imported
if not OmegaConf.has_resolver("get_log_range"):
    OmegaConf.register_new_resolver("get_log_range", get_log_range)

def get_num_layers(model_id: str):
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_id)
    return model_config.num_hidden_layers

if not OmegaConf.has_resolver("get_num_layers"):
    OmegaConf.register_new_resolver("get_num_layers", get_num_layers)

# resolves from list of layers to freeze from fractions to model layers 
# depending on the number of layers in the model
def resolve_freeze_layers(coeffs_tuple_list, model_id):
    if coeffs_tuple_list is None:
        return None
    nl = get_num_layers(model_id)
    lst = []
    for t in coeffs_tuple_list:
        lst.append((int(float(t[0])*nl), int(float(t[1])*nl)))
    return lst

if not OmegaConf.has_resolver("resolve_freeze_layers"):
    OmegaConf.register_new_resolver(
        "resolve_freeze_layers", resolve_freeze_layers
    )


def compute_rtt_signature(
    dataset_name: str,
    ft_loss_types: list,
    ft_lrs: list,
    ft_epochs_lst: list,
    num_ft_splits: int,
    ft_freeze_layers,
    ft_data_format_name: str,
    eval_split_ids: Optional[list[int]] = None,
) -> str:
    """Compute a stable signature for RTT configuration.
    
    Used to deduplicate baseline RTT runs - same signature means same RTT config.
    """
    import hashlib
    eval_split_ids_sig = (
        tuple(sorted(eval_split_ids)) if eval_split_ids is not None else None
    )
    sig_data = str((
        dataset_name,
        sorted([lt.name if hasattr(lt, 'name') else str(lt) for lt in ft_loss_types]),
        sorted(ft_lrs),
        sorted(ft_epochs_lst),
        num_ft_splits,
        str(ft_freeze_layers),
        ft_data_format_name,
        eval_split_ids_sig,
    ))
    return hashlib.md5(sig_data.encode()).hexdigest()[:12]


def build_parent_metadata(
    run_name: str,
    unlearn_type: UnlearnType,
    dataset: Datasets,
    wandb_project_name: str,
    base_model: str,
    unlearn_lr: float,
    unlearn_epochs: int,
    retain_coeff: float,
    lora_rank: int = 0,
    steering_coeff: Optional[float] = None,
) -> Dict[str, Any]:
    """Build parent metadata for B checkpoint from A checkpoint info.
    
    Args:
        run_name: Run name (top-level directory name)
        unlearn_type: Unlearning method type
        dataset: Dataset enum
        wandb_project_name: WandB project name
        base_model: Base model ID
        unlearn_lr: Learning rate used during unlearning
        unlearn_epochs: Number of epochs used during unlearning
        retain_coeff: Retain coefficient
        lora_rank: LoRA rank (0 if not used)
        steering_coeff: Steering coefficient (None if not used)
    
    Returns:
        Dictionary containing parent metadata for B checkpoint
    """
    parent_metadata = {
        "run_name": run_name,
        "method": unlearn_type.name,
        "dataset": dataset.name,
        "project": wandb_project_name,
        "model_id": base_model,
        "lr": unlearn_lr,  # A checkpoint's lr (unlearning phase)
        "epochs": unlearn_epochs,  # A checkpoint's epochs (unlearning phase)
        "retain_coeff": retain_coeff,
    }
    if lora_rank > 0:
        parent_metadata["lora_rank"] = lora_rank
    if steering_coeff is not None:
        parent_metadata["steering_coeff"] = steering_coeff
    return parent_metadata


def write_checkpoint_manifest_entry(
    run_name: str,
    checkpoint_type: str,
    checkpoint_path: str,
    metadata: Dict[str, Any],
) -> None:
    """Write a checkpoint entry to the manifest file with file locking.
    
    Args:
        run_name: Run name (top-level directory name)
        checkpoint_type: "A", "B", or "C"
        checkpoint_path: Full path to the checkpoint directory
        metadata: Dictionary containing all checkpoint metadata
    """
    manifest_path = os.path.join("models", run_name, "manifest.json")
    lock_path = manifest_path + ".lock"
    
    # Prepare the entry
    entry = {
        "type": checkpoint_type,
        "path": checkpoint_path,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **metadata
    }
    
    # File locking for concurrent writes (cross-platform using filelock library)
    max_retries = 10
    base_retry_delay = 0.1  # Base delay for exponential backoff
    
    for attempt in range(max_retries):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            
            # Use filelock for cross-platform file locking (non-blocking with timeout=0)
            lock = FileLock(lock_path, timeout=0)
            with lock:
                # Load existing entries or initialize empty list
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            entries = json.loads(content)
                        else:
                            entries = []
                else:
                    entries = []
                
                # Append new entry
                entries.append(entry)
                
                # Write to temporary file first (atomic write)
                temp_path = manifest_path + ".tmp"
                with open(temp_path, 'w') as tmp_file:
                    json.dump(entries, tmp_file, indent=2)
                
                # Atomic rename
                os.rename(temp_path, manifest_path)
                
                # Success - return
                return
                
        except Timeout:
            # Retryable: lock contention (FileLock raises Timeout when timeout=0 and lock is held)
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Could not acquire lock on {manifest_path} after {max_retries} attempts")
        except IOError as e:
            # Retryable: file system errors
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Could not acquire lock on {manifest_path} after {max_retries} attempts: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            # Non-retryable: corrupted file
            raise RuntimeError(f"Manifest file is corrupted: {e}")
        except Exception as e:
            # Other errors - retry for transient issues
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Failed to write manifest entry after {max_retries} attempts: {e}")


@ray.remote(num_gpus=1)
def evaluate_baseline_model(
    model_id: str,
    val_files: list[str],
    dev_file: str,
    data_root: str,
    val_batch_size: int = 8,
    attn_backend: Optional[str] = None,
) -> dict:
    """Evaluate baseline model accuracy on forget set (pre-flight check).
    
    Returns dict with accuracy per file and average accuracy.
    Used to verify the model actually knows the information before unlearning.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.attention_backend import get_attn_implementation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    attn_impl = get_attn_implementation(attn_backend)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Get label token IDs for A, B, C, D
    doc_to_choice = ["A", "B", "C", "D"]
    label_possibilities = [
        tokenizer.encode(c, add_special_tokens=False)[-1]
        for c in doc_to_choice
    ]
    
    def create_prompt(point):
        return "\n".join(
            [point["question"]]
            + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
            + ["Answer:"]
        )
    
    def _data_path(rel_path: str, ext: str = ".jsonl") -> str:
        """Helper to resolve data path with data_root prefix."""
        if os.path.isabs(rel_path):
            base = rel_path
        else:
            base = os.path.join(data_root, rel_path)
        if ext and not base.endswith(ext):
            base += ext
        return base
    
    results = {}
    total_correct = 0
    total_count = 0
    
    for val_file in val_files:
        # Load data
        file_path = _data_path(val_file)
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        correct = 0
        for point in data:
            prompt = create_prompt(point)
            tokens = tokenizer(
                prompt, return_tensors="pt", max_length=512,
                truncation=True, padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**tokens)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
            # Get prediction among A, B, C, D
            label_logits = logits[label_possibilities]
            pred = label_logits.argmax().item()
            
            if pred == point["answer"]:
                correct += 1
        
        acc = correct / len(data) if data else 0.0
        results[val_file] = acc
        total_correct += correct
        total_count += len(data)
    
    avg_acc = total_correct / total_count if total_count > 0 else 0.0
    results["average"] = avg_acc
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return results


# routes to appropriate unlearning function
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
    data_root: str = "data",
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
    use_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bf16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_double_quant: bool = True,
    max_seq_len: int = 512,
    grad_accum_steps: int = 1,
    gradient_checkpointing: bool = False,
    attn_backend: Optional[str] = None,
):
    if unlearn_type.value == UnlearnType.NOT_SPECIFIED.value:
        raise Exception("Must specify unlearning type")

    elif (
        unlearn_type.value == UnlearnType.GD.value
        or unlearn_type.value == UnlearnType.WHP.value
        or unlearn_type.value == UnlearnType.FWF.value
    ):
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
                unlearn_type=unlearn_type,
                train_files=unlearn_files,
                wrong_unlearn_files=wrong_unlearn_files,
                fixed_wrong_unlearn_files=fixed_wrong_unlearn_files,
                val_files=val_files,
                dev_set=dev_file,
                retain_files=retain_files,
                val_retain_files=val_retain_files,
                retain_dev_file=retain_dev_file,
                data_root=data_root,
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
                use_4bit=use_4bit,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_double_quant=bnb_4bit_double_quant,
                max_seq_len=max_seq_len,
                grad_accum_steps=grad_accum_steps,
                gradient_checkpointing=gradient_checkpointing,
                attn_backend=attn_backend,
            )
        )

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
                unlearn_type=UnlearnType.GD,
                train_files=unlearn_files,
                wrong_unlearn_files=wrong_unlearn_files,
                fixed_wrong_unlearn_files=fixed_wrong_unlearn_files,
                val_files=val_files,
                dev_set=dev_file,
                retain_files=retain_files,
                val_retain_files=val_retain_files,
                retain_dev_file=retain_dev_file,
                data_root=data_root,
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
                use_4bit=use_4bit,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_double_quant=bnb_4bit_double_quant,
                max_seq_len=max_seq_len,
                grad_accum_steps=grad_accum_steps,
                gradient_checkpointing=gradient_checkpointing,
                attn_backend=attn_backend,
            )
        )

    elif unlearn_type.value == UnlearnType.CUT.value:
        import rmu.unlearn_pipeline as rmu
        (
            model_path,
            forget_accs, forget_accs_calibrated, forget_logits_dict,
            retain_accs, retain_accs_calibrated, retain_logits_dict,
            retain_accs_5_shot, retain_accs_5_shot_calibrated,
            retain_logits_5_shot_dict,
            samples
        ) = rmu.main(
            unlearn_files=unlearn_files,
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
            hydra_dict=hydra_dict,
            data_format=data_format,
            steering_coeff=steering_coeff,
	    max_samples=max_samples,
        )
    
    else:
        raise Exception("Unlearn type not handled")
    
    return (
        model_path,
        forget_accs, forget_accs_calibrated, forget_logits_dict,
        retain_accs, retain_accs_calibrated, retain_logits_dict,
        retain_accs_5_shot, retain_accs_5_shot_calibrated,
        retain_logits_5_shot_dict,
        samples
    )
    
# calls unlearning for one configuration of unlearning then RTT
# can be used to perform either unlearning or RTT based on parameters
@ray.remote
def main(
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED, # unlearning method
    dataset: Datasets = Datasets.NOT_SPECIFIED,
    unlearn_files: list[str] = [], # path to jsonl files used for unlearning
    wrong_unlearn_files: list[str] = [], # used for RIA
    fixed_wrong_unlearn_files: list[str] = [], # used for UnlearnType.FWF
    val_files: list[str] = [], # for tracking accuracy
    dev_file: str = "", # used for creating few-shot prompts
    retain_files: list[str] = [], # jsonl files for retain dataset
    val_retain_files: list[str] = [], # for tracking accuracy on retain dataset
    retain_dev_file: str = "", # used to create few-shot prompts for retain set
    data_root: str = "data",
    base_model: str = "", # path to model to perform unlearning on
    lr: float = 1e-7,
    epochs: int = 3,
    batch_size: int = 4,
    val_batch_size: int = 8,
    retain_coeff: int = 1, # multiplied by retain loss
    warmup_steps: int = 24, # gradually increasing lr
    data_seed: int = 0,
    eval_every: int = 1, 
    save_name: Optional[str] = None,
    name: str = "forget_model",
    wandb_project_name: str = "unlearn",
    results_dir: str = "evals/pipline",
    only_ft: bool = False, # only perform RTT
    ft_model_path: str = "",  # If only performing RTT, use this model
    num_ft_splits: int = 5, # How many eval splits (V) to use for RTT
    eval_split_ids: Optional[list[int]] = None,
    num_total_splits: Optional[int] = None,
    ft_loss_types: list[LossType] = [LossType.NOT_SPECIFIED],
    ft_lrs: list[float] = [5e-7],
    ft_epochs_lst: list[int] = [3],
    save_ft_models: bool = False,
    start_time: str = "", 
    start_time_sf: str = "",
    dont_ft: bool = False, # true if only want unlearning and no RTT
    just_eval: bool = False, # only evaluate model on dataset
    diff_tokenizer: str = "", # use a custom tokenizer
    unlearn_freeze_layers: Optional[list[tuple[int, int]]] = None, 
    ft_freeze_layers: Optional[list[tuple[int, int]]] = None,
    ft_dont_eval: bool = False,
    ft_on_all: bool = False, # use all splits for training in RTT
    unlearn_mcq: bool = False, 
    hydra_dict: dict = {}, # for logging 
    unlearn_data_format: DataFormat = DataFormat.CORPUS,
    ft_data_format: DataFormat = DataFormat.MCQ,
    unlearn_loss_type: LossType = LossType.CORPUS,
    steering_coeff: float = 20, # for RMU
    max_samples: int = 9999999999, # limit number of datapoints for unlearning
    lora_rank: int = 0,  # LoRA rank (0 = disabled)
    use_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bf16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_double_quant: bool = True,
    max_seq_len: int = 512,
    grad_accum_steps: int = 1,
    gradient_checkpointing: bool = False,
    attn_backend: Optional[str] = None,  # Attention backend: auto, flash_attention_2, sdpa, eager
    ft_batch_size: Optional[int] = None,  # Separate batch size for RTT (defaults to batch_size)
    ft_val_batch_size: Optional[int] = None,  # Separate val batch size for RTT (defaults to val_batch_size)
    run_name: str = "",  # Run name for organizing model outputs
):
    try:
        if num_total_splits is None:
            num_total_splits = len(val_files)
        if eval_split_ids is None:
            eval_split_ids = list(
                range(min(num_ft_splits, num_total_splits))
            )
        else:
            eval_split_ids = [
                int(i) for i in eval_split_ids
                if 0 <= int(i) < num_total_splits
            ]
        if not only_ft:
            if just_eval:
                import unlearn_corpus
                ref =  unlearn_corpus.just_eval.remote(
                    unlearn_type=unlearn_corpus.UnlearnType.GD,
                    train_files=[],
                    wrong_unlearn_files=[],
                    fixed_wrong_unlearn_files=[],
                    val_files=val_files,
                    dev_set=dev_file,
                    retain_files=[],
                    val_retain_files=val_retain_files,
                    retain_dev_file=retain_dev_file,
                    data_root=data_root,
                    base_model=base_model,
                    lr=lr,
                    name=name,
                    epochs=epochs,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    retain_coeff=retain_coeff,
                    warmup_steps=warmup_steps,
                    data_seed=data_seed,
                    eval_every=eval_every,
                    save_name=None,
                    project_name=wandb_project_name,
                    just_eval=True,
                    disable_wandb=True,
                    freeze_layers=unlearn_freeze_layers,
                    hydra_dict=hydra_dict,
                    data_format=unlearn_data_format,
                    max_samples=max_samples,
                    use_4bit=use_4bit,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_double_quant=bnb_4bit_double_quant,
                    max_seq_len=max_seq_len,
                    grad_accum_steps=grad_accum_steps,
                    gradient_checkpointing=gradient_checkpointing,
                    attn_backend=attn_backend,
                )
                (
                    model_path,
                    forget_accs, forget_accs_calibrated,
                    forget_logits_dict,
                    retain_accs, retain_accs_calibrated,
                    retain_logits_dict,
                    retain_accs_5_shot, retain_accs_5_shot_calibrated,
                    retain_logits_5_shot_dict,
                    samples
                ) = ray.get(ref)
            else:
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
                    data_root=data_root,
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
                    use_4bit=use_4bit,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_double_quant=bnb_4bit_double_quant,
                    max_seq_len=max_seq_len,
                    grad_accum_steps=grad_accum_steps,
                    gradient_checkpointing=gradient_checkpointing,
                    attn_backend=attn_backend,
                )
                (
                    model_path,
                    forget_accs, forget_accs_calibrated,
                    forget_logits_dict,
                    retain_accs, retain_accs_calibrated,
                    retain_logits_dict,
                    retain_accs_5_shot, retain_accs_5_shot_calibrated,
                    retain_logits_5_shot_dict,
                    samples
                ) = ray.get(ref)
            
            # Write manifest entry for A checkpoint if it was saved
            if save_name is not None and not just_eval:
                # Extract metadata for A checkpoint
                a_metadata = {
                    "run_name": run_name,
                    "method": unlearn_type.name,
                    "dataset": dataset.name,
                    "project": wandb_project_name,
                    "model_id": base_model,
                    "lr": lr,
                    "epochs": epochs,
                    "retain_coeff": retain_coeff,
                }
                
                # Add optional fields
                if lora_rank > 0:
                    a_metadata["lora_rank"] = lora_rank
                if steering_coeff is not None:
                    a_metadata["steering_coeff"] = steering_coeff
                
                write_checkpoint_manifest_entry(
                    run_name=run_name,
                    checkpoint_type="A",
                    checkpoint_path=model_path,
                    metadata=a_metadata,
                )

            if just_eval:
                print(f"{base_model=}\n{forget_accs=}\n{retain_accs=}")  

        if only_ft:
            model_path = ft_model_path
        if dont_ft or just_eval:
            return
        # Use separate batch sizes for RTT if provided, otherwise use unlearning batch sizes
        rtt_batch_size = ft_batch_size if ft_batch_size is not None else batch_size
        rtt_val_batch_size = ft_val_batch_size if ft_val_batch_size is not None else val_batch_size
        ft_refs = []
        # Compute RTT signature for matching B and C results
        rtt_sig = compute_rtt_signature(
            dataset.name, ft_loss_types, ft_lrs, ft_epochs_lst,
            num_ft_splits, ft_freeze_layers, ft_data_format.name,
            eval_split_ids=eval_split_ids,
        )
        
        # Capture unlearning hyperparameters before FT loops (to avoid variable shadowing)
        unlearn_lr = lr
        unlearn_epochs = epochs
        
        # Helper function to extract remaining path after 'models/{run_name}/'
        def extract_remaining_path(model_path: str) -> str:
            """Extract path components after 'models/{run_name}/' from model path."""
            path_parts = model_path.split('/')
            if len(path_parts) > 1 and path_parts[0] == 'models':
                # Skip 'models' and first part (which will be run_name), keep the rest
                return '/'.join(path_parts[2:]) if len(path_parts) > 2 else '/'.join(path_parts[1:])
            else:
                return '/'.join(path_parts[1:]) if len(path_parts) > 1 else model_path
        
        for loss_type in ft_loss_types:
            for ft_lr in ft_lrs:
                for ft_epochs in ft_epochs_lst:
                    if not ft_on_all:
                        for skip_split in eval_split_ids:
                            import finetune_corpus
                            remaining_path = extract_remaining_path(model_path)
                            fted_model_path = (
                                f"models/{run_name}/fted/"
                                f"{remaining_path}/"
                                f"{wandb_project_name}/"
                                f"{loss_type}/ft-skip_split{skip_split}/"
                                f"lr{ft_lr}-epoch{ft_epochs}"
                            )
                            ft_files = [
                                file for i, file in enumerate(val_files)
                                if i != skip_split
                            ]
                            ft_val_files = (
                                [val_files[skip_split]]
                                if skip_split < len(val_files) else [""]
                            )
                            ft_val_retain_files = val_retain_files
                            train_split_ids = [
                                i for i in range(num_total_splits)
                                if i != skip_split
                            ]
                            
                            # Prepare parent metadata for B checkpoint (A checkpoint info)
                            parent_metadata = build_parent_metadata(
                                run_name=run_name,
                                unlearn_type=unlearn_type,
                                dataset=dataset,
                                wandb_project_name=wandb_project_name,
                                base_model=base_model,
                                unlearn_lr=unlearn_lr,
                                unlearn_epochs=unlearn_epochs,
                                retain_coeff=retain_coeff,
                                lora_rank=lora_rank,
                                steering_coeff=steering_coeff,
                            )
                            
                            ref = finetune_corpus.main.remote(
                                train_files=ft_files,
                                val_files=ft_val_files,
                                val_retain_files=ft_val_retain_files,
                                dev_set=ft_files[0],
                                data_root=data_root,
                                base_model=model_path,
                                lr=ft_lr,
                                epochs=ft_epochs,
                                name=fted_model_path,
                                batch_size=rtt_batch_size,
                                val_batch_size=rtt_val_batch_size,
                                save_name=(
                                    fted_model_path if save_ft_models
                                    else None
                                ),
                                loss_type=loss_type,
                                project_name=wandb_project_name,
                                diff_tokenizer=diff_tokenizer, 
                                freeze_layers=ft_freeze_layers,
                                dont_eval=ft_dont_eval,
                                hydra_dict=hydra_dict,
                                data_format=ft_data_format,
                                attn_backend=attn_backend,
                                run_name=run_name,
                                checkpoint_type="B",
                                parent_metadata=parent_metadata,
                                skip_split=skip_split,
                            )
                            ft_refs.append(ref)
                    else:
                        import finetune_corpus
                        remaining_path = extract_remaining_path(model_path)
                        fted_model_path = (
                            f"models/{run_name}/fted/"
                            f"{remaining_path}/"
                            f"{loss_type}/all_splits/lr{ft_lr}-epoch{ft_epochs}"
                        )
                        ft_files = val_files
                        ft_val_files = val_files
                        ft_val_retain_files = val_retain_files
                        all_split_ids = list(range(num_total_splits))
                        
                        # Prepare parent metadata for B checkpoint (A checkpoint info)
                        parent_metadata = build_parent_metadata(
                            run_name=run_name,
                            unlearn_type=unlearn_type,
                            dataset=dataset,
                            wandb_project_name=wandb_project_name,
                            base_model=base_model,
                            unlearn_lr=unlearn_lr,
                            unlearn_epochs=unlearn_epochs,
                            retain_coeff=retain_coeff,
                            lora_rank=lora_rank,
                            steering_coeff=steering_coeff,
                        )
                        
                        ref = finetune_corpus.main.remote(
                            train_files=ft_files,
                            val_files=ft_val_files,
                            val_retain_files=ft_val_retain_files,
                            dev_set=ft_files[0],
                            data_root=data_root,
                            base_model=model_path,
                            lr=ft_lr,
                            epochs=ft_epochs,
                            name=fted_model_path,
                            batch_size=ft_batch_size,
                            val_batch_size=ft_val_batch_size,
                            save_name=(
                                fted_model_path if save_ft_models
                                else None
                            ),
                            loss_type=loss_type,
                            project_name=wandb_project_name,
                            diff_tokenizer=diff_tokenizer, 
                            freeze_layers=ft_freeze_layers,
                            dont_eval=ft_dont_eval,
                            hydra_dict=hydra_dict,
                            data_format=ft_data_format,
                            attn_backend=attn_backend,
                            run_name=run_name,
                            checkpoint_type="B",
                            parent_metadata=parent_metadata,
                            skip_split=None,  # ft_on_all doesn't have skip_split
                        )
                        ft_refs.append(ref)
        while len(ft_refs) > 0:
            done_ft_refs, ft_refs = ray.wait(ft_refs)
            for done_ref in done_ft_refs:
                ray.get(done_ref)
    
    except ray.exceptions.RayTaskError as e:
        error_message = f"""\
            Exception in main:\n{str(e)}\n\n\
            Traceback:\n{traceback.format_exc()}\
        """
        print(error_message)
        
        # Write the error to a file
        error_file_path = "pipeline_error.log"
        with open(error_file_path, "a+") as error_file:
            error_file.seek(0)
            content = error_file.read()
            if content:
                error_file.write("\n\n")
            error_file.write(f"--- Error at {get_current_time()} ---\n")
            error_file.write(error_message)
        
        global raise_exceptions
        if raise_exceptions:
            raise e

# MMLU categories to use for forget loss
mmlu_cats_forget = ["STEM", "business", "chemistry", "culture", "geography"]

mmlu_cats_retain = [
"health", "history", "law", "philosophy", "social sciences"
]

# paths for different dataset
datasets_dict = {
    Datasets.YEARS: {
        "unlearn_files": [
            f"dates-years-trimmed/corpus_split_{i}" for i in range(5)
        ],
        "wrong_unlearn_files": [
            f"wrong-dates-years-trimmed/corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"fixed-wrong-dates-years-trimmed/corpus_split_{i}"
            for i in range(5)
        ],
        "val_files": [
            f"dates-years-trimmed/split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"fineweb_edu_seed-42/split_{i}" for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.YEARS_MMLU_RETAIN: {
        "unlearn_files": [
            f"dates-years-trimmed/corpus_split_{i}" for i in range(5)
        ],
        "wrong_unlearn_files": [
            f"dates-years-trimmed/whp_corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"dates-years-trimmed/fwf_corpus_split_{i}"
            for i in range(5)
        ],
        "val_files": [
            f"dates-years-trimmed/split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{mmlu_cats_forget[i]}"
            for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.YEARS_TF: {
        "unlearn_files": [
            *[f"dates-years-trimmed/corpus_split_{i}" for i in range(5)],
            *[f"dates-years-trimmed/tf_split_{i}" for i in range(5)],
        ],
        "wrong_unlearn_files": [
            f"wrong-dates-years-trimmed/corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            *[f"dates-years-trimmed/tf_split_{i}" for i in range(5)],
            *[
                f"fixed-wrong-dates-years-trimmed/corpus_split_{i}"
                for i in range(5)
            ]
        ],
        "val_files": [
            f"dates-years-trimmed/split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"fineweb_edu_seed-42/split_{i}" for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.MMLU: {
        "unlearn_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{mmlu_cats_forget[i]}"
            for i in range(5)
        ],
        "wrong_unlearn_files": [
            f"mmlu_cats_random_trimmed/whp_corpus_mmlu_{mmlu_cats_forget[i]}"
            for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"mmlu_cats_random_trimmed/"
            f"fwf_corpus_mmlu_{mmlu_cats_forget[i]}"
            for i in range(5)
        ],
        "val_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_forget[i]}"
            for i in range(5)
        ],
        "retain_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "dev_file": "mmlu_cats_random_trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_CORPUS: {
        "unlearn_files": [
            f"wmdp/bio-forget-corpus",
            f"wmdp/cyber-forget-corpus"
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "retain_files": [
            "wikitext/wikitext_dataset",
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_CORPUS_FINEWEB: {
        "unlearn_files": [
            f"wmdp/bio-forget-corpus",
            f"wmdp/cyber-forget-corpus"
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "retain_files": [
            f"fineweb_edu_seed-42/split_{i}" for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_CORPUS_MMLU: {
        "unlearn_files": [
            f"wmdp/bio-forget-corpus",
            f"wmdp/cyber-forget-corpus"
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "retain_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_CORPUS: {
        "unlearn_files": [
            f"wmdp-deduped/corpus_split_{i}" for i in range(5)
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [
            f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [
            "wikitext/wikitext_dataset",
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_CORPUS_FINEWEB: {
        "unlearn_files": [
            f"wmdp-deduped/corpus_split_{i}" for i in range(5)
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [
            f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"fineweb_edu_seed-42/split_{i}" for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_FINEWEB: {
        "unlearn_files": [
            f"wmdp-deduped/mcq_split_{i}" for i in range(5)
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [
            f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"fineweb_edu_seed-42/split_{i}" for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_WIKITEXT: {
        "unlearn_files": [
            f"wmdp-deduped/mcq_split_{i}" for i in range(5)
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [
            f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [
            "wikitext/wikitext_dataset",
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_LETTER_ANSWER: {
        "unlearn_files": [
            f"wmdp-deduped/mcq_split_{i}" for i in range(5)
        ],
        "val_files": [
            f"wmdp-deduped/split_{i}" for i in range(5)
        ],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [
            f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"fineweb_edu_seed-42/split_{i}" for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.BEAVERTAILS: {
        "unlearn_files": [
            "beavertails/criminal_activities_dataset",
            "beavertails/social_issues_dataset"
        ],
        "val_files": [
            "beavertails/criminal_activities_dataset",
            "beavertails/social_issues_dataset"
        ],
        "dev_file": "",
        "retain_files": [
            ""
        ],
        "val_retain_files": [
            ""
        ],
        "retain_dev_file" : "" 
    },
    Datasets.RANDOM_BD: {
        "unlearn_files": [
            f"random_bd/corpus_split_{i}" for i in range(5)
        ],
        "wrong_unlearn_files": [
            f"random_bd/whp_corpus_split_{i}" for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"random_bd/fwf_corpus_split_{i}" for i in range(5)
        ],
        "val_files": [
            f"random_bd/split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"fineweb_edu_seed-42/split_{i}" for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.RANDOM_BD_SAME_RETAIN: {
        "unlearn_files": [
            f"random_bd/corpus_split_{i}" for i in range(5)
        ],
        "val_files": [
            f"random_bd/split_{i}" for i in range(5)
        ],
        "retain_files": [
            f"random_bd/corpus_split_{i}" for i in range(5, 10)
        ],
        "val_retain_files": [
            f"random_bd/split_{i}" for i in range(5, 10)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.RANDOM_BD_ALL_SPLITS: {
        "unlearn_files": [
        ],
        "val_files": [
            f"random_bd/split_{i}" for i in range(10)
        ],
        "retain_files": [
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.RANDOM_BD_WITH_MMLU: {
        "unlearn_files": [
        ],
        "val_files": [
            *[f"random_bd/split_{i}" for i in range(10)],
            *[
                f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
                for i in range(5)
            ]
        ],
        "retain_files": [
        ],
        "val_retain_files": [
            *[f"random_bd/split_{i}" for i in range(10)],
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.RANDOM_BD_WITH_MMLU_CORPUS: {
        "unlearn_files": [
        ],
        "wrong_unlearn_files": [
            f"random_bd/corpus_split_{i}" for i in range(5)
        ],
        "val_files": [
            *[f"random_bd/split_{i}" for i in range(5)]
        ],
        "retain_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{mmlu_cats_retain[i]}"
            for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.DAY_OF_THE_MONTH: {
        "unlearn_files": [
        ],
        "val_files": [
            f"day_of_the_month/split_{i}" for i in range(1, 5)
        ],
        "retain_files": [
        ],
        "val_retain_files": [
            f"day_of_the_month/split_{i}" for i in range(1)
        ],
        "dev_file": "day_of_the_month/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    }
}


def get_num_gpus():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()

    else:
        return 0


config_file = "default"

# The main function that reads configurations from hydra config files and calls
# `main()` for each unlearning configuration
@hydra.main(config_path="conf", config_name=config_file, version_base=None)
def run_pipeline(cfg: DictConfig) -> None:
    try:
        data_root = get_data_root(cfg)
        try:
            validate_required_artifacts(cfg.datasets, data_root)
        except FileNotFoundError as exc:
            print(str(exc))
            return

        # Use num_gpus from config if specified, otherwise default to 8 or available GPUs
        num_gpus = OmegaConf.select(cfg, "num_gpus", default=None)
        if num_gpus is None:
            num_gpus = 8 if get_num_gpus() >= 8 else get_num_gpus()
        ray.init(num_gpus=num_gpus)
        refs = []
        
        # Baseline RTT tracking - deduplicate across hyperparameter points
        scheduled_baseline_rtt: set = set()  # (model_id, dataset.name, rtt_sig)
        baseline_rtt_refs: list = []

        curr_time = datetime.datetime.now()
        curr_time_str = curr_time.strftime("%Y-%m-%d-%H-%M-%S")
        start_time_sf_str = get_current_time().strftime("%Y-%m-%d-%H-%M-%S")
        
        # Load config flags early (needed for run_name validation)
        only_ft = cfg.only_ft
        just_eval = cfg.just_eval
        dont_ft = cfg.dont_ft
        
        # Validate run_name for RTT-only runs BEFORE auto-generation
        run_name = OmegaConf.select(cfg, "run_name", default=None)
        run_name_was_provided = run_name is not None and run_name != ""
        
        if only_ft and not run_name_was_provided:
            raise ValueError(
                "run_name must be explicitly specified when only_ft=true. "
                "Please set run_name in your config file (e.g., run_name: 'my_rtt_run')."
            )
        
        # Generate run_name if not specified (defaults to timestamp)
        if not run_name_was_provided:
            run_name = curr_time.strftime("%Y-%m-%d_%H-%M-%S")
        
        unlearn_types = [UnlearnType[ut] for ut in cfg.unlearn.types]
        datasets = [Datasets[d] for d in cfg.datasets]
        model_id = cfg.model_id
        unlearn_freeze_layers = cfg.unlearn.freeze_layers
        unlearn_types_config = cfg.unlearn.types_config
        eval_model_paths = cfg.eval_model_paths
        ft_model_paths = cfg.ft_model_paths
        wandb_project_name = cfg.wandb_project_name
        results_dir = cfg.results_dir
        
        batch_size = cfg.batch_size
        val_batch_size = cfg.val_batch_size
        warmup_steps = cfg.warmup_steps
        data_seed = cfg.data_seed
        eval_every = cfg.eval_every

        # fine-tuning hyper-parameters
        num_ft_splits = cfg.ft.num_splits
        ft_loss_types = [LossType[lt] for lt in cfg.ft.loss_types]
        ft_lrs = cfg.ft.lrs
        ft_epochs_lst = cfg.ft.epochs_lst
        save_ft_models = cfg.ft.save_models
        # Use separate batch sizes for RTT if specified, otherwise use global batch sizes
        ft_batch_size = OmegaConf.select(cfg, "ft.batch_size", default=batch_size)
        ft_val_batch_size = OmegaConf.select(cfg, "ft.val_batch_size", default=val_batch_size)
        eval_split_ids_cfg = OmegaConf.select(
            cfg, "ft.eval_split_ids", default=None
        )
        eval_seed = OmegaConf.select(cfg, "ft.eval_seed", default=None)

        # optional hyperparameters
        diff_tokenizer = OmegaConf.select(cfg, "diff_tokenizer", default="")
        unlearn_freeze_layers = OmegaConf.select(
            cfg, "unlearn.freeze_layers", default=None
        )
        ft_freeze_layers = OmegaConf.select(
            cfg, "ft.freeze_layers", default=None
        )
        ft_dont_eval = OmegaConf.select(cfg, "ft_dont_eval", default=False)
        ft_on_all = OmegaConf.select(cfg, "ft_on_all", default=False)
        unlearn_mcq = OmegaConf.select(cfg, "unlearn_mcq", default=False)
        unlearn_data_format = OmegaConf.select(
            cfg, "unlearn.data_format", default="CORPUS"
        )
        unlearn_data_format = DataFormat[unlearn_data_format]
        ft_data_format = OmegaConf.select(
            cfg, "ft.data_format", default="MCQ"
        )
        ft_data_format = DataFormat[ft_data_format]
        global raise_exceptions
        raise_exceptions = OmegaConf.select(
            cfg, "raise_exceptions", default=False
        )
        cut_scs = OmegaConf.select(
            cfg, "unlearn.cut_scs", default=[20]
        )
        max_samples_lst = OmegaConf.select(
            cfg, "unlearn.max_samples_lst", default=[999999999]
        )
        lora_ranks = OmegaConf.select(
            cfg, "unlearn.lora_ranks", default=[0]
        )
        save_unlearn_model = OmegaConf.select(
            cfg, "unlearn.save_unlearn_model", default=True
        )
        if FORCE_SAVE_ALL_MODELS:
            save_unlearn_model = True
            save_ft_models = True
        attn_backend = OmegaConf.select(
            cfg, "attn_backend", default="auto"
        )
        unlearn_use_4bit = OmegaConf.select(
            cfg, "unlearn.use_4bit", default=False
        )
        unlearn_bnb_4bit_compute_dtype = OmegaConf.select(
            cfg, "unlearn.bnb_4bit_compute_dtype", default="bf16"
        )
        unlearn_bnb_4bit_quant_type = OmegaConf.select(
            cfg, "unlearn.bnb_4bit_quant_type", default="nf4"
        )
        unlearn_bnb_4bit_double_quant = OmegaConf.select(
            cfg, "unlearn.bnb_4bit_double_quant", default=True
        )
        unlearn_max_seq_len = OmegaConf.select(
            cfg, "unlearn.max_seq_len", default=512
        )
        unlearn_grad_accum_steps = OmegaConf.select(
            cfg, "unlearn.grad_accum_steps", default=1
        )
        unlearn_gradient_checkpointing = OmegaConf.select(
            cfg, "unlearn.gradient_checkpointing", default=False
        )

        # Logs hyperparameters to wandb
        config_flat = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=wandb_project_name,
            config=config_flat,
            name="pipeline"
        )
        table_data = []
        for k, v in config_flat.items():
            table_data.append([k, str(v)])
        table = wandb.Table(columns=list(config_flat.keys()))
        table.add_data(*[str(v) for v in config_flat.values()])
        wandb.log({"Config Table": table})
        wandb.finish()

        # Pre-flight check: verify baseline model knows the information
        baseline_min_acc = OmegaConf.select(
            cfg, "baseline_min_forget_acc", default=0.3
        )
        validated_datasets = set()  # Datasets that pass baseline check
        
        if baseline_min_acc > 0 and not only_ft and not just_eval:
            print(f"\n{'='*60}")
            print(f"Pre-flight check: Evaluating baseline model on forget sets")
            print(f"Minimum required accuracy: {baseline_min_acc}")
            print(f"{'='*60}\n")
            
            for dataset in datasets:
                dataset_dict = resolve_dataset_dict_paths(
                    datasets_dict[dataset], data_root
                )
                
                print(f"Evaluating {model_id} on {dataset.name}...")
                
                try:
                    eval_ref = evaluate_baseline_model.remote(
                        model_id=model_id,
                        val_files=dataset_dict["val_files"],
                        dev_file=dataset_dict["dev_file"],
                        data_root=data_root,
                        val_batch_size=val_batch_size,
                        attn_backend=attn_backend,
                    )
                    eval_results = ray.get(eval_ref)
                    avg_acc = eval_results.get("average", 0.0)
                    
                    if avg_acc >= baseline_min_acc:
                        validated_datasets.add(dataset)
                        print(f"  ✓ {dataset.name}: {avg_acc:.4f} >= {baseline_min_acc} (PASS)")
                    else:
                        print(f"  ✗ {dataset.name}: {avg_acc:.4f} < {baseline_min_acc} (FAIL)")
                        print(f"    Skipping unlearning experiments for {dataset.name}")
                        print(f"    (Model doesn't know the information well enough)")
                        
                except Exception as e:
                    print(f"  ! {dataset.name}: Evaluation failed: {e}")
                    print(f"    Skipping unlearning experiments for {dataset.name}")
            
            print(f"\n{'='*60}")
            if validated_datasets:
                print(f"Proceeding with datasets: {[d.name for d in validated_datasets]}")
            else:
                print("No datasets passed baseline validation. Exiting.")
                ray.shutdown()
                return
            print(f"{'='*60}\n")
            
            # Filter datasets to only validated ones
            datasets = [d for d in datasets if d in validated_datasets]
        else:
            # Skip validation - mark all datasets as validated
            validated_datasets = set(datasets)

        if eval_seed is None:
            eval_seed = data_seed

        def resolve_eval_split_ids(
            dataset_name: str,
            num_total_splits: int,
        ) -> list[int]:
            num_eval_splits = min(num_ft_splits, num_total_splits)
            if eval_split_ids_cfg is not None:
                raw_ids = eval_split_ids_cfg
                if isinstance(raw_ids, str):
                    try:
                        import ast
                        raw_ids = ast.literal_eval(raw_ids)
                    except Exception:
                        raw_ids = [raw_ids]
                ids = [int(i) for i in list(raw_ids)]
                ids = [i for i in ids if 0 <= i < num_total_splits]
                return ids[:num_eval_splits]
            import hashlib
            import random
            seed_str = f"{eval_seed}-{dataset_name}"
            seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
            rng = random.Random(seed_int)
            return (
                sorted(rng.sample(range(num_total_splits), k=num_eval_splits))
                if num_eval_splits > 0 else []
            )

        dataset_dicts = {}
        eval_split_ids_by_dataset = {}
        num_total_splits_by_dataset = {}
        for dataset in datasets:
            dataset_dict = resolve_dataset_dict_paths(
                datasets_dict[dataset], data_root
            )
            dataset_dicts[dataset] = dataset_dict
            num_total_splits = len(dataset_dict["val_files"])
            num_total_splits_by_dataset[dataset] = num_total_splits
            eval_split_ids_by_dataset[dataset] = resolve_eval_split_ids(
                dataset.name, num_total_splits
            )

        if not only_ft and not just_eval:
            for unlearn_type in unlearn_types:
                unlearn_type_config = unlearn_types_config[
                    unlearn_type.name
                ] 
                unlearn_loss_type_str =  unlearn_type_config["loss_type"]
                unlearn_loss_type = LossType[unlearn_loss_type_str]
                for dataset in datasets:
                    dataset_config = (
                        unlearn_type_config["datasets_config"][dataset.name]
                    )
                    epochs_lst = dataset_config["epochs_lst"]
                    lrs = dataset_config["lrs"]
                    rcs = (
                        dataset_config["rcs"]["range"]
                        + [float(rc) for rc in dataset_config["rcs"]["add"]]
                    )
                    dataset_dict = dataset_dicts[dataset]
                    eval_split_ids = eval_split_ids_by_dataset[dataset]
                    num_total_splits = num_total_splits_by_dataset[dataset]
                    print(f"""
                        {unlearn_type=}
                        {unlearn_loss_type=}
                        {dataset=}
                        {epochs_lst=}
                        {lrs=}
                        {rcs=}
                        {max_samples_lst=}
                    """)
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
                                                f"models/{run_name}/"
                                                f"{unlearn_type.name}/"
                                                f"{dataset.name}/"
                                                f"{wandb_project_name}/"
                                                f"rank{lora_rank}-sc{sc}-"
                                                f"{model_id}-rc{rc}-lr{lr}-"
                                                f"epochs{epochs}"
                                            )
                                            save_name = (
                                                forget_model if save_unlearn_model
                                                else None
                                            )
                                            refs += [main.remote(
                                                unlearn_type=unlearn_type,
                                                dataset=dataset,
                                                unlearn_files=(
                                                    dataset_dict["unlearn_files"]
                                                ),
                                                wrong_unlearn_files=(
                                                    dataset_dict.get(
                                                        "wrong_unlearn_files",
                                                        []
                                                )),
                                                fixed_wrong_unlearn_files = (
                                                    dataset_dict.get(
                                                        "fixed_wrong_unlearn_files",
                                                        []
                                                    )
                                                ),
                                                val_files=dataset_dict["val_files"],
                                                dev_file=dataset_dict["dev_file"],
                                                retain_files=(
                                                    dataset_dict["retain_files"]
                                                ),
                                                val_retain_files=dataset_dict[
                                                    "val_retain_files"
                                                ],
                                                retain_dev_file=dataset_dict[
                                                    "retain_dev_file"
                                                ],
                                                data_root=data_root,
                                                base_model=model_id,
                                                lr=lr,
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                val_batch_size=val_batch_size,
                                                retain_coeff=rc,
                                                warmup_steps=warmup_steps,
                                                data_seed=data_seed,
                                                eval_every=eval_every,
                                                save_name=save_name,
                                                name=forget_model,
                                                wandb_project_name=(
                                                    wandb_project_name
                                                ),
                                                results_dir=results_dir,
                                                only_ft=only_ft,
                                                ft_model_path="",
                                                num_ft_splits=num_ft_splits,
                                                eval_split_ids=eval_split_ids,
                                                num_total_splits=num_total_splits,
                                                ft_loss_types=ft_loss_types,
                                                ft_lrs=ft_lrs,
                                                ft_epochs_lst=ft_epochs_lst,
                                                save_ft_models=save_ft_models,
                                                start_time=curr_time_str,
                                                start_time_sf=start_time_sf_str,
                                                dont_ft=dont_ft,
                                                unlearn_freeze_layers=(
                                                    unlearn_freeze_layers
                                                ),
                                                ft_freeze_layers=ft_freeze_layers,
                                                ft_dont_eval=ft_dont_eval,
                                                unlearn_mcq=unlearn_mcq,
                                                hydra_dict=config_flat,
                                                unlearn_data_format=(
                                                    unlearn_data_format
                                                ),
                                                ft_data_format=ft_data_format,
                                                unlearn_loss_type=unlearn_loss_type,
                                                steering_coeff=sc,
                                                max_samples=max_samples,
                                                lora_rank=lora_rank,
                                                use_4bit=unlearn_use_4bit,
                                                bnb_4bit_compute_dtype=unlearn_bnb_4bit_compute_dtype,
                                                bnb_4bit_quant_type=unlearn_bnb_4bit_quant_type,
                                                bnb_4bit_double_quant=unlearn_bnb_4bit_double_quant,
                                                max_seq_len=unlearn_max_seq_len,
                                                grad_accum_steps=unlearn_grad_accum_steps,
                                                gradient_checkpointing=unlearn_gradient_checkpointing,
                                                attn_backend=attn_backend,
                                                ft_batch_size=ft_batch_size,
                                                ft_val_batch_size=ft_val_batch_size,
                                                run_name=run_name,
                                            )]

            # Schedule baseline RTT (C) - once per unique (model_id, dataset, RTT-config)
            # Only schedule if RTT is enabled
            if not dont_ft and not ft_on_all:
                import finetune_corpus
                for dataset in datasets:
                    dataset_dict = dataset_dicts[dataset]
                    eval_split_ids = eval_split_ids_by_dataset[dataset]
                    num_total_splits = num_total_splits_by_dataset[dataset]
                    rtt_sig = compute_rtt_signature(
                        dataset.name, ft_loss_types, ft_lrs, ft_epochs_lst,
                        num_ft_splits, ft_freeze_layers, ft_data_format.name,
                        eval_split_ids=eval_split_ids,
                    )
                    baseline_key = (model_id, dataset.name, rtt_sig)
                    if baseline_key in scheduled_baseline_rtt:
                        continue
                    scheduled_baseline_rtt.add(baseline_key)
                    for loss_type in ft_loss_types:
                        for ft_lr in ft_lrs:
                            for ft_epochs in ft_epochs_lst:
                                for skip_split in eval_split_ids:
                                    ft_files = [
                                        file for i, file in enumerate(
                                            dataset_dict["val_files"]
                                        )
                                        if i != skip_split
                                    ]
                                    ft_val_files = (
                                        [dataset_dict["val_files"][skip_split]]
                                        if skip_split < len(
                                            dataset_dict["val_files"]
                                        ) else [""]
                                    )
                                    train_split_ids = [
                                        i for i in range(num_total_splits)
                                        if i != skip_split
                                    ]
                                    baseline_rtt_path = (
                                        f"models/{run_name}/baseline_rtt/"
                                        f"{dataset.name}/{model_id.replace('/', '_')}/"
                                        f"{loss_type.name}/skip_split{skip_split}/"
                                        f"lr{ft_lr}-epoch{ft_epochs}"
                                    )
                                    # Prepare minimal metadata for C checkpoint
                                    c_metadata = {
                                        "run_name": run_name,
                                        "dataset": dataset.name,
                                        "model_id": model_id,
                                    }
                                    
                                    ref = finetune_corpus.main.remote(
                                        train_files=ft_files,
                                        val_files=ft_val_files,
                                        val_retain_files=dataset_dict.get(
                                            "val_retain_files", ft_files.copy()
                                        ),
                                        dev_set=ft_files[0] if ft_files else "",
                                        data_root=data_root,
                                        base_model=model_id,  # Original base model
                                        lr=ft_lr,
                                        epochs=ft_epochs,
                                        name=baseline_rtt_path,
                                        batch_size=ft_batch_size,
                                        val_batch_size=ft_val_batch_size,
                                        save_name=baseline_rtt_path,
                                        loss_type=loss_type,
                                        project_name=f"{wandb_project_name}_baseline_rtt",
                                        diff_tokenizer=diff_tokenizer,
                                        freeze_layers=ft_freeze_layers,
                                        dont_eval=ft_dont_eval,
                                        hydra_dict=config_flat,
                                        data_format=ft_data_format,
                                        attn_backend=attn_backend,
                                        run_name=run_name,
                                        checkpoint_type="C",
                                        parent_metadata=c_metadata,
                                        skip_split=skip_split,
                                    )
                                    baseline_rtt_refs.append(ref)

        elif only_ft:
            for ft_model_path, dataset in ft_model_paths:
                dataset = Datasets[dataset]
                unlearn_type = UnlearnType.GD
                unlearn_type_config = unlearn_types_config[
                    unlearn_type.name
                ] 
                unlearn_loss_type = unlearn_type_config["loss_type"]
                dataset_config = (
                    unlearn_type_config["datasets_config"][Datasets.YEARS.name]
                )
                epochs_lst = dataset_config["epochs_lst"]
                lrs = dataset_config["lrs"]
                rcs = (
                    dataset_config["rcs"]["range"]
                    + dataset_config["rcs"]["add"]
                )
                dataset_dict = dataset_dicts.get(dataset)
                if dataset_dict is None:
                    dataset_dict = resolve_dataset_dict_paths(
                        datasets_dict[dataset], data_root
                    )
                    dataset_dicts[dataset] = dataset_dict
                eval_split_ids = eval_split_ids_by_dataset.get(dataset)
                if eval_split_ids is None:
                    num_total_splits = len(dataset_dict["val_files"])
                    num_total_splits_by_dataset[dataset] = num_total_splits
                    eval_split_ids = resolve_eval_split_ids(
                        dataset.name, num_total_splits
                    )
                    eval_split_ids_by_dataset[dataset] = eval_split_ids
                else:
                    num_total_splits = num_total_splits_by_dataset[dataset]
                refs += [main.remote(
                    unlearn_type=unlearn_types[0],
                    dataset=dataset,
                    unlearn_files=dataset_dict["unlearn_files"],
                    wrong_unlearn_files=dataset_dict.get(
                        "wrong_unlearn_files", []
                    ),
                    fixed_wrong_unlearn_files = dataset_dict.get(
                        "fixed_wrong_unlearn_files", []
                    ),
                    val_files=dataset_dict["val_files"],
                    dev_file=dataset_dict["dev_file"],
                    retain_files=dataset_dict["retain_files"],
                    val_retain_files=dataset_dict["val_retain_files"],
                    retain_dev_file=dataset_dict["retain_dev_file"],
                    data_root=data_root,
                    base_model=model_id,
                    lr=lrs[0],
                    epochs=2,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    retain_coeff=rcs[0],
                    warmup_steps=warmup_steps,
                    data_seed=data_seed,
                    eval_every=eval_every,
                    save_name=ft_model_path,
                    wandb_project_name=wandb_project_name,
                    results_dir=results_dir,
                    only_ft=only_ft,
                    ft_model_path=ft_model_path,
                    num_ft_splits=num_ft_splits,
                    eval_split_ids=eval_split_ids,
                    num_total_splits=num_total_splits,
                    ft_loss_types=ft_loss_types,
                    ft_lrs=ft_lrs,
                    ft_epochs_lst=ft_epochs_lst,
                    save_ft_models=save_ft_models,
                    start_time=curr_time_str,
                    start_time_sf=start_time_sf_str,
                    dont_ft=dont_ft,
                    diff_tokenizer=diff_tokenizer,
                    unlearn_freeze_layers=unlearn_freeze_layers,
                    ft_freeze_layers=ft_freeze_layers,
                    ft_dont_eval=ft_dont_eval,
                    ft_on_all=ft_on_all,
                    hydra_dict=config_flat,
                    unlearn_data_format=unlearn_data_format,
                    ft_data_format=ft_data_format,
                    unlearn_loss_type=unlearn_loss_type,
                    attn_backend=attn_backend,
                    run_name=run_name,
                )]

            # Schedule baseline RTT (C) for only_ft mode - same as normal pipeline
            if not dont_ft and not ft_on_all:
                import finetune_corpus
                for ft_model_path, dataset_str in ft_model_paths:
                    dataset = Datasets[dataset_str]
                    dataset_dict = dataset_dicts.get(dataset)
                    if dataset_dict is None:
                        dataset_dict = resolve_dataset_dict_paths(
                            datasets_dict[dataset], data_root
                        )
                        dataset_dicts[dataset] = dataset_dict
                    eval_split_ids = eval_split_ids_by_dataset.get(dataset)
                    if eval_split_ids is None:
                        num_total_splits = len(dataset_dict["val_files"])
                        num_total_splits_by_dataset[dataset] = num_total_splits
                        eval_split_ids = resolve_eval_split_ids(
                            dataset.name, num_total_splits
                        )
                        eval_split_ids_by_dataset[dataset] = eval_split_ids
                    else:
                        num_total_splits = num_total_splits_by_dataset[dataset]
                    rtt_sig = compute_rtt_signature(
                        dataset.name, ft_loss_types, ft_lrs, ft_epochs_lst,
                        num_ft_splits, ft_freeze_layers, ft_data_format.name,
                        eval_split_ids=eval_split_ids,
                    )
                    baseline_key = (model_id, dataset.name, rtt_sig)
                    if baseline_key in scheduled_baseline_rtt:
                        continue
                    scheduled_baseline_rtt.add(baseline_key)
                    for loss_type in ft_loss_types:
                        for ft_lr in ft_lrs:
                            for ft_epochs in ft_epochs_lst:
                                for skip_split in eval_split_ids:
                                    ft_files = [
                                        file for i, file in enumerate(
                                            dataset_dict["val_files"]
                                        )
                                        if i != skip_split
                                    ]
                                    ft_val_files = (
                                        [dataset_dict["val_files"][skip_split]]
                                        if skip_split < len(
                                            dataset_dict["val_files"]
                                        ) else [""]
                                    )
                                    baseline_rtt_path = (
                                        f"models/{run_name}/baseline_rtt/"
                                        f"{dataset.name}/{model_id.replace('/', '_')}/"
                                        f"{loss_type.name}/skip_split{skip_split}/"
                                        f"lr{ft_lr}-epoch{ft_epochs}"
                                    )
                                    # Prepare minimal metadata for C checkpoint
                                    c_metadata = {
                                        "run_name": run_name,
                                        "dataset": dataset.name,
                                        "model_id": model_id,
                                    }
                                    
                                    ref = finetune_corpus.main.remote(
                                        train_files=ft_files,
                                        val_files=ft_val_files,
                                        val_retain_files=dataset_dict.get(
                                            "val_retain_files", ft_files.copy()
                                        ),
                                        dev_set=ft_files[0] if ft_files else "",
                                        data_root=data_root,
                                        base_model=model_id,  # Original base model
                                        lr=ft_lr,
                                        epochs=ft_epochs,
                                        name=baseline_rtt_path,
                                        batch_size=ft_batch_size,
                                        val_batch_size=ft_val_batch_size,
                                        save_name=baseline_rtt_path,
                                        loss_type=loss_type,
                                        project_name=f"{wandb_project_name}_baseline_rtt",
                                        diff_tokenizer=diff_tokenizer,
                                        freeze_layers=ft_freeze_layers,
                                        dont_eval=ft_dont_eval,
                                        hydra_dict=config_flat,
                                        data_format=ft_data_format,
                                        attn_backend=attn_backend,
                                        run_name=run_name,
                                        checkpoint_type="C",
                                        parent_metadata=c_metadata,
                                        skip_split=skip_split,
                                    )
                                    baseline_rtt_refs.append(ref)


        elif just_eval:
            for dataset in datasets:
                for model_id in eval_model_paths:
                    unlearn_type = UnlearnType.GD
                    unlearn_type_config = unlearn_types_config[
                        unlearn_type.name
                    ] 
                    unlearn_loss_type = unlearn_type_config["loss_type"]
                    datasets_config = unlearn_type_config["datasets_config"]
                    dataset_config = (
                        datasets_config[Datasets.YEARS.name]
                    )
                    epochs_lst = dataset_config["epochs_lst"]
                    lrs = dataset_config["lrs"]
                    rcs = (
                        dataset_config["rcs"]["range"]
                        + dataset_config["rcs"]["add"]
                    )
                    dataset_dict = dataset_dicts.get(dataset)
                    if dataset_dict is None:
                        dataset_dict = resolve_dataset_dict_paths(
                            datasets_dict[dataset], data_root
                        )
                        dataset_dicts[dataset] = dataset_dict
                    eval_split_ids = eval_split_ids_by_dataset.get(dataset)
                    if eval_split_ids is None:
                        num_total_splits = len(dataset_dict["val_files"])
                        num_total_splits_by_dataset[dataset] = num_total_splits
                        eval_split_ids = resolve_eval_split_ids(
                            dataset.name, num_total_splits
                        )
                        eval_split_ids_by_dataset[dataset] = eval_split_ids
                    else:
                        num_total_splits = num_total_splits_by_dataset[dataset]
                    refs += [main.remote(
                        unlearn_type=unlearn_types[0],
                        dataset=dataset,
                        unlearn_files=dataset_dict["unlearn_files"],
                        wrong_unlearn_files=dataset_dict.get(
                            "wrong_unlearn_files", []
                        ),
                        fixed_wrong_unlearn_files=dataset_dict.get(
                            "fixed_wrong_unlearn_files", []
                        ),
                        val_files=dataset_dict["val_files"],
                        dev_file=dataset_dict["dev_file"],
                        retain_files=dataset_dict["retain_files"],
                        val_retain_files=dataset_dict["val_retain_files"],
                        retain_dev_file=dataset_dict["retain_dev_file"],
                        data_root=data_root,
                        base_model=model_id,
                        lr=lrs[0],
                        epochs=2,
                        batch_size=batch_size,
                        val_batch_size=val_batch_size,
                        retain_coeff=rcs[0],
                        warmup_steps=warmup_steps,
                        data_seed=data_seed,
                        eval_every=eval_every,
                        save_name=model_id,
                        wandb_project_name=wandb_project_name,
                        results_dir=results_dir,
                        only_ft=only_ft,
                        ft_model_path="",
                        num_ft_splits=num_ft_splits,
                        eval_split_ids=eval_split_ids,
                        num_total_splits=num_total_splits,
                        ft_loss_types=ft_loss_types,
                        ft_lrs=ft_lrs,
                        ft_epochs_lst=ft_epochs_lst,
                        save_ft_models=save_ft_models,
                        start_time=curr_time_str,
                        start_time_sf=start_time_sf_str,
                        dont_ft=dont_ft,
                        just_eval=True,
                        unlearn_freeze_layers=unlearn_freeze_layers,
                        ft_freeze_layers=ft_freeze_layers,
                        ft_dont_eval=ft_dont_eval,
                        hydra_dict=config_flat,
                        unlearn_data_format=unlearn_data_format,
                        ft_data_format=ft_data_format,
                        unlearn_loss_type=unlearn_loss_type,
                        attn_backend=attn_backend,
                        run_name=run_name,
                    )]

        for ref in refs:
            try:
                ray.get(ref)
            except ray.exceptions.RayTaskError as e:
                error_message = f"""
                Exception in main:\n{str(e)}\n\n\
                Traceback:\n{traceback.format_exc()}\
                """
                print(error_message)
                
                # Write the error to a file
                error_file_path = "pipeline_error.log"
                with open(error_file_path, "a+") as error_file:
                    error_file.seek(0)
                    content = error_file.read()
                    if content:
                        error_file.write("\n\n")
                    error_file.write(
                        f"--- Error at {get_current_time()} ---\n"
                    )
                    error_file.write(error_message)
                if raise_exceptions:
                    raise(e)

        # Process baseline RTT results (C condition)
        while len(baseline_rtt_refs) > 0:
            done_refs, baseline_rtt_refs = ray.wait(baseline_rtt_refs)
            for done_ref in done_refs:
                try:
                    ray.get(done_ref)
                except ray.exceptions.RayTaskError as e:
                    error_message = f"""
                    Exception in baseline RTT:\n{str(e)}\n\n\
                    Traceback:\n{traceback.format_exc()}\
                    """
                    print(error_message)
                    error_file_path = "pipeline_error.log"
                    with open(error_file_path, "a+") as error_file:
                        error_file.seek(0)
                        content = error_file.read()
                        if content:
                            error_file.write("\n\n")
                        error_file.write(
                            f"--- Error at {get_current_time()} ---\n"
                        )
                        error_file.write(error_message)
                    if raise_exceptions:
                        raise(e)

        emit_terminal_notice(f"[PIPELINE COMPLETE] {get_current_time()}")
        ray.shutdown()
    except Exception as e:
        emit_terminal_notice(f"[PIPELINE FAILED] {get_current_time()}")
        err_str = f"""\
        Training Run failed with error: {e}\n\n\n{traceback.format_exc()}\
        """
        raise Exception(err_str)

if __name__ == "__main__":
    run_pipeline()
