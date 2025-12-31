import os
import sys
from typing import Optional, Dict, Any, Callable
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


# Default values for matched forgetting fallback
DEFAULT_UNLEARN_LR = 4e-7
DEFAULT_UNLEARN_EPOCHS = 6
DEFAULT_RETAIN_COEFF = 0.1


def parse_and_validate_list_config(
    config_dict: dict,
    key: str,
    default: list,
    allow_empty: bool = False,
    param_name: str = None,
) -> list:
    """Parse and validate a list configuration value from OmegaConf dict.
    
    Args:
        config_dict: Configuration dictionary
        key: Key to extract
        default: Default value if key not present
        allow_empty: Whether empty lists are allowed (default: False)
        param_name: Name for error messages (default: uses key)
    
    Returns:
        Validated list
    
    Raises:
        ValueError: If value is invalid
    """
    import ast
    from omegaconf import ListConfig
    
    param_name = param_name or key
    value = config_dict.get(key, default)
    
    # Handle string representation
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception as e:
            raise ValueError(f"Invalid {param_name} format: {value}") from e
    
    # Convert OmegaConf ListConfig to Python list
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=True)
    
    # Validate type
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{param_name} must be a list, got: {type(value).__name__}")
    
    value = list(value)
    
    # Validate non-empty if required
    if not allow_empty and len(value) == 0:
        raise ValueError(f"{param_name} must be non-empty list, got: {value}")
    
    return value


def extract_model_path_suffix(model_path: str, run_name: str) -> str:
    """Extract path suffix after 'models/{run_name}/' with validation.
    
    Args:
        model_path: Full model path
        run_name: Expected run name for validation
    
    Returns:
        Path suffix after 'models/{run_name}/'
    
    Raises:
        ValueError: If path format is invalid
    """
    path_parts = model_path.split('/')
    if len(path_parts) < 3 or path_parts[0] != 'models' or path_parts[1] != run_name:
        raise ValueError(f"Unexpected model path format: {model_path}")
    return '/'.join(path_parts[2:])


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


def write_year_concept_csv(
    results: list[Dict[str, Any]],
    results_dir: str,
    timestamp: str,
) -> str:
    """Write year concept evaluation results to CSV file.
    
    Args:
        results: List of result dictionaries from year concept evaluation
        results_dir: Root directory for results (e.g., "evals/pipeline")
        timestamp: Timestamp string for filename
    """
    import csv
    
    # Create output directory
    output_dir = os.path.join(results_dir, "year_concept")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find next available file number
    num = 0
    while True:
        csv_path = os.path.join(output_dir, f"{timestamp}--num{num}.csv")
        if not os.path.exists(csv_path):
            break
        num += 1
    
    # Write CSV
    if not results:
        return ""
    
    fieldnames = [
        "model_path",
        "base_model",
        "lora_rank",
        "checkpoint_type",
        "dataset_name",
        "ordering_acc",
        "arithmetic_acc",
        "classification_acc",
        "boundary_acc",
        "overall_acc",
        "ordering_count",
        "arithmetic_count",
        "classification_count",
        "boundary_count",
        "total_count",
        "timestamp",
        "start_time_sf",
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Only write the fields we care about
            row = {k: result.get(k, "") for k in fieldnames}
            writer.writerow(row)
    
    print(f"Year concept evaluation results written to: {csv_path}")
    return csv_path


def _write_json_with_lock(
    file_path: str,
    update_func: Callable[[Any], Any],
    initial_value: Any = None,
) -> None:
    """Write JSON file with file locking and atomic writes.
    
    Args:
        file_path: Path to JSON file
        update_func: Function that takes current data and returns updated data
        initial_value: Initial value if file doesn't exist (default: None means infer from update_func result)
    
    Raises:
        RuntimeError: If file locking or write fails after retries
    """
    lock_path = file_path + ".lock"
    max_retries = 10
    base_retry_delay = 0.1  # Base delay for exponential backoff
    
    for attempt in range(max_retries):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Use filelock for cross-platform file locking (non-blocking with timeout=0)
            lock = FileLock(lock_path, timeout=0)
            with lock:
                # Load existing or initialize
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            current_data = json.loads(content)
                        else:
                            current_data = initial_value if initial_value is not None else {}
                else:
                    current_data = initial_value if initial_value is not None else {}
                
                # Update data using provided function
                updated_data = update_func(current_data)
                
                # Write to temporary file first (atomic write)
                temp_path = file_path + ".tmp"
                with open(temp_path, 'w') as tmp_file:
                    json.dump(updated_data, tmp_file, indent=2)
                
                # Atomic rename
                os.rename(temp_path, file_path)
                
                # Success - return
                return
                
        except Timeout:
            # Retryable: lock contention (FileLock raises Timeout when timeout=0 and lock is held)
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Could not acquire lock on {file_path} after {max_retries} attempts")
        except IOError as e:
            # Retryable: file system errors
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Could not acquire lock on {file_path} after {max_retries} attempts: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            # Non-retryable: corrupted file
            raise RuntimeError(f"JSON file is corrupted: {e}")
        except Exception as e:
            # Other errors - retry for transient issues
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Failed to write JSON file after {max_retries} attempts: {e}")


def _read_json_with_lock(file_path: str, default: Any = None) -> Any:
    """Read JSON file with file locking to prevent race conditions.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist
    
    Returns:
        Parsed JSON data or default
    
    Raises:
        RuntimeError: If file read fails after retries
    """
    lock_path = file_path + ".lock"
    max_retries = 10
    base_retry_delay = 0.1  # Base delay for exponential backoff
    
    for attempt in range(max_retries):
        try:
            if not os.path.exists(file_path):
                return default
            
            # Use filelock for cross-platform file locking (non-blocking with timeout=0)
            lock = FileLock(lock_path, timeout=0)
            with lock:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        return default
                    return json.loads(content)
                    
        except Timeout:
            # Retryable: lock contention
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                # If we can't get lock after retries, read without lock as fallback
                # (better than failing entirely)
                try:
                    with open(file_path, 'r') as f:
                        return json.loads(f.read())
                except Exception as e:
                    raise RuntimeError(f"Failed to read JSON file {file_path} after lock timeout: {e}")
        except (json.JSONDecodeError, IOError) as e:
            # Retryable: transient read errors
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Failed to read JSON file {file_path}: {e}")
        except Exception as e:
            # Other errors - retry for transient issues
            if attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Failed to read JSON file {file_path} after {max_retries} attempts: {e}")


def write_checkpoint_manifest_entry(
    run_name: str,
    checkpoint_type: str,
    checkpoint_path: str,
    metadata: Dict[str, Any],
    tags: Optional[list[str]] = None,
) -> None:
    """Write a checkpoint entry to the manifest file with file locking.
    
    Args:
        run_name: Run name (top-level directory name)
        checkpoint_type: "A", "B", or "C"
        checkpoint_path: Full path to the checkpoint directory
        metadata: Dictionary containing all checkpoint metadata
        tags: Optional list of tags (e.g., ["matched_forgetting"])
    """
    manifest_path = os.path.join("models", run_name, "manifest.json")
    
    def update_manifest(current_entries):
        if current_entries is None:
            current_entries = []
        entry = {
            "type": checkpoint_type,
            "path": checkpoint_path,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **metadata
        }
        if tags:
            entry["tags"] = tags
        current_entries.append(entry)
        return current_entries
    
    _write_json_with_lock(manifest_path, update_manifest, initial_value=[])


def write_matched_forgetting_json(
    run_name: str,
    dataset_name: str,
    lora_rank: int,
    selection_data: dict,
    k_budget: Optional[int] = None,
) -> None:
    """Write matched forgetting selection data with file locking.
    
    Args:
        run_name: Run name (top-level directory name)
        dataset_name: Dataset name
        lora_rank: LoRA rank
        selection_data: Dictionary containing selection metadata
        k_budget: Optional K budget (for learned top-K mode)
    """
    matched_forgetting_path = os.path.join("models", run_name, "matched_forgetting.json")
    
    def update_matched_data(current_data):
        if current_data is None:
            current_data = {}
        if dataset_name not in current_data:
            current_data[dataset_name] = {}
        # Use rank_K key format if k_budget is provided, otherwise use rank-only key
        if k_budget is not None:
            key = f"rank{lora_rank}_k{k_budget}"
        else:
            key = str(lora_rank)
        current_data[dataset_name][key] = selection_data
        return current_data
    
    _write_json_with_lock(matched_forgetting_path, update_matched_data, initial_value={})


def extract_avg_acc(
    acc_dict: dict,
    acc_selection_rule: str = "final_epoch"
) -> float:
    """Extract average accuracy from accuracy dict structure.
    
    Generic function that works for both forget and retain accuracy dicts.
    
    Args:
        acc_dict: Dict mapping {file: {epoch: accuracy}}
        acc_selection_rule: "final_epoch" or "max_epoch"
    
    Returns:
        Average accuracy across all files at selected epoch
    """
    if not acc_dict:
        return 0.0
    
    file_accs = []
    for file, epoch_accs in acc_dict.items():
        if not epoch_accs:
            continue
        if acc_selection_rule == "final_epoch":
            selected_epoch = max(epoch_accs.keys())
        elif acc_selection_rule == "max_epoch":
            selected_epoch = max(epoch_accs.keys(), key=lambda e: epoch_accs[e])
        else:
            raise ValueError(f"Unknown acc_selection_rule: {acc_selection_rule}")
        file_accs.append(epoch_accs[selected_epoch])
    
    return sum(file_accs) / len(file_accs) if file_accs else 0.0


def select_matched_checkpoint(
    candidates: list[dict],
    target_acc: float,
    tolerance: float,
    baseline_retain_acc: float,
    selection_priority: list[str],
) -> Optional[dict]:
    """Select best checkpoint from candidates using matched-forgetting rules.
    
    Args:
        candidates: List of dicts with keys: forget_acc, retain_acc, epochs, rc, lr, model_path, ...
        target_acc: Target forget accuracy (A*)
        tolerance: Tolerance band around target
        baseline_retain_acc: Baseline retain accuracy for computing retain damage
        selection_priority: Order of tie-breaking: ["retain_damage", "compute", "retain_coeff"]
    
    Returns:
        Selected candidate dict or None if no candidates
    """
    if not candidates:
        return None
    
    # Filter candidates within tolerance
    in_tolerance = [
        c for c in candidates
        if target_acc - tolerance <= c["forget_acc"] <= target_acc + tolerance
    ]
    
    # If none in tolerance, pick closest to target
    if not in_tolerance:
        in_tolerance = [
            min(candidates, key=lambda c: abs(c["forget_acc"] - target_acc))
        ]
    
    # Apply selection rule
    def score(candidate):
        """Compute score tuple for lexicographic comparison.
        
        Returns tuple (retain_damage, compute, retain_coeff) where:
        - Smaller values are preferred (min() selects best candidate)
        - Order matches selection_priority for tie-breaking
        - Used with min() for lexicographic comparison
        """
        retain_damage = baseline_retain_acc - candidate["retain_acc"]
        # Approximate compute as epochs * steps_per_epoch
        # Using 1000 as typical steps per epoch approximation
        STEPS_PER_EPOCH_APPROX = 1000
        compute = candidate.get("epochs", 0) * STEPS_PER_EPOCH_APPROX
        rc = candidate.get("rc", 0)
        
        scores = []
        for priority in selection_priority:
            if priority == "retain_damage":
                scores.append(retain_damage)
            elif priority == "compute":
                scores.append(compute)
            elif priority == "retain_coeff":
                scores.append(rc)
        return tuple(scores)
    
    return min(in_tolerance, key=score)


@ray.remote
def evaluate_baseline_retain_acc(
    model_id: str,
    val_retain_files: list[str],
    retain_dev_file: str,
    data_root: str,
    val_batch_size: int = 8,
    attn_backend: Optional[str] = None,
) -> float:
    """Evaluate baseline model accuracy on retain set.
    
    Returns average retain accuracy across all retain validation files.
    Used for computing retain damage in matched-forgetting selection.
    
    Note: This function doesn't require GPU directly - just_eval.remote() handles GPU allocation.
    """
    import unlearn_corpus
    
    # Use just_eval to evaluate baseline model on retain set
    # Pass empty train_files to skip training, just evaluate
    # just_eval.remote() will handle GPU allocation, so this function doesn't need num_gpus=1
    (
        model_path,
        forget_accs, forget_accs_calibrated, forget_logits_dict,
        retain_accs, retain_accs_calibrated, retain_logits_dict,
        retain_accs_5_shot, retain_accs_5_shot_calibrated,
        retain_logits_5_shot_dict,
        samples,
        gate_metadata
    ) = ray.get(unlearn_corpus.just_eval.remote(
        train_files=[],
        wrong_unlearn_files=[],
        fixed_wrong_unlearn_files=[],
        val_files=[],  # Not used for retain evaluation
        dev_set=retain_dev_file,
        retain_files=[],
        val_retain_files=val_retain_files,
        retain_dev_file=retain_dev_file,
        data_root=data_root,
        base_model=model_id,
        lr=1e-7,  # Not used for evaluation
        name="baseline_retain_eval",
        epochs=1,  # Not used for evaluation
        batch_size=4,
        val_batch_size=val_batch_size,
        retain_coeff=1,
        warmup_steps=0,
        data_seed=0,
        eval_every=1,
        save_name=None,
        project_name="baseline_retain_eval",
        just_eval=True,
        disable_wandb=True,
        freeze_layers=None,
        mcq=False,
        hydra_dict={},
        data_format=DataFormat.MCQ,  # Assume MCQ format for retain evaluation
        loss_type=LossType.LETTER,
        unlearn_type=UnlearnType.GD,  # Required parameter, but not used when just_eval=True and train_files=[]
        lora_rank=0,
        use_4bit=False,
        bnb_4bit_compute_dtype="bf16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_double_quant=True,
        max_seq_len=512,
        grad_accum_steps=1,
        gradient_checkpointing=False,
        attn_backend=attn_backend,
    ))
    
    # Extract average retain accuracy
    # Structure: {file: {epoch: accuracy}}
    if not retain_accs:
        return 0.0
    
    file_accs = []
    for file, epoch_accs in retain_accs.items():
        if not epoch_accs:
            continue
        # Use final epoch
        selected_epoch = max(epoch_accs.keys())
        file_accs.append(epoch_accs[selected_epoch])
    
    return sum(file_accs) / len(file_accs) if file_accs else 0.0


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
    tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
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
            samples,
            gate_metadata
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
            samples,
            gate_metadata
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
        gate_metadata = None  # CUT doesn't use gate metadata
    
    else:
        raise Exception("Unlearn type not handled")
    
    return (
        model_path,
        forget_accs, forget_accs_calibrated, forget_logits_dict,
        retain_accs, retain_accs_calibrated, retain_logits_dict,
        retain_accs_5_shot, retain_accs_5_shot_calibrated,
        retain_logits_5_shot_dict,
        samples,
        gate_metadata if 'gate_metadata' in locals() else None
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
    collect_results: bool = False,  # Return lightweight A/B metrics for aggregation
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
                    samples,
                    gate_metadata
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
                    samples,
                    gate_metadata
                ) = ray.get(ref)
            
            # Write manifest entry for A checkpoint if it was saved
            # Check if model was actually saved (model_path exists and is not empty/base model)
            model_was_saved = (
                model_path is not None 
                and model_path != "" 
                and model_path != base_model
                and os.path.exists(model_path)
                and os.path.exists(os.path.join(model_path, "config.json"))
            )
            
            if model_was_saved and not just_eval:
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
                
                # Add gate metadata if present
                if gate_metadata:
                    a_metadata["lora_layer_budget_k"] = gate_metadata.get("lora_layer_budget_k")
                    a_metadata["selected_blocks"] = gate_metadata.get("selected_blocks")
                    a_metadata["final_gate_scores"] = gate_metadata.get("final_gate_scores")
                    a_metadata["gate_tau_start"] = gate_metadata.get("gate_tau_start")
                    a_metadata["gate_tau_end"] = gate_metadata.get("gate_tau_end")
                    a_metadata["gate_seed"] = gate_metadata.get("gate_seed")
                
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
            # Read A checkpoint's manifest entry to get correct metadata
            manifest_path = os.path.join("models", run_name, "manifest.json")
            if os.path.exists(manifest_path):
                manifest_data = _read_json_with_lock(manifest_path, default=[])
                if isinstance(manifest_data, list):
                    # Find the Type A checkpoint entry matching model_path
                    for entry in manifest_data:
                        if entry.get("type") == "A" and entry.get("path") == model_path:
                            # Override wandb_project_name with the project from A checkpoint
                            if "project" in entry:
                                wandb_project_name = entry["project"]
                            # Override epochs with the epochs from A checkpoint
                            if "epochs" in entry:
                                epochs = entry["epochs"]
                            # Override lr with the lr from A checkpoint
                            if "lr" in entry:
                                lr = entry["lr"]
                            # Override retain_coeff with the retain_coeff from A checkpoint
                            if "retain_coeff" in entry:
                                retain_coeff = entry["retain_coeff"]
                            # Override base_model with the model_id from A checkpoint
                            if "model_id" in entry:
                                base_model = entry["model_id"]
                            # Override unlearn_type if method is specified
                            if "method" in entry:
                                try:
                                    unlearn_type = UnlearnType[entry["method"]]
                                except KeyError:
                                    pass
                            break
        if dont_ft or just_eval:
            # Return the tuple even when skipping RTT
            return (
                model_path,
                forget_accs, forget_accs_calibrated, forget_logits_dict,
                retain_accs, retain_accs_calibrated, retain_logits_dict,
                retain_accs_5_shot, retain_accs_5_shot_calibrated,
                retain_logits_5_shot_dict,
                samples,
                gate_metadata if 'gate_metadata' in locals() else None
            )
        # Use separate batch sizes for RTT if provided, otherwise use unlearning batch sizes
        rtt_batch_size = ft_batch_size if ft_batch_size is not None else batch_size
        rtt_val_batch_size = ft_val_batch_size if ft_val_batch_size is not None else val_batch_size
        ft_refs = []
        collected_b_results: list[dict] = []
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
                            
                            # Read gate metadata from Type A checkpoint's manifest entry if available
                            manifest_path = os.path.join("models", run_name, "manifest.json")
                            if os.path.exists(manifest_path):
                                manifest_data = _read_json_with_lock(manifest_path, default=[])
                                if isinstance(manifest_data, list):
                                    # Find the Type A checkpoint entry matching model_path
                                    for entry in manifest_data:
                                        if entry.get("type") == "A" and entry.get("path") == model_path:
                                            # Extract gate metadata if present
                                            if "lora_layer_budget_k" in entry:
                                                parent_metadata["lora_layer_budget_k"] = entry.get("lora_layer_budget_k")
                                            if "selected_blocks" in entry:
                                                parent_metadata["selected_blocks"] = entry.get("selected_blocks")
                                            if "final_gate_scores" in entry:
                                                parent_metadata["final_gate_scores"] = entry.get("final_gate_scores")
                                            if "gate_tau_start" in entry:
                                                parent_metadata["gate_tau_start"] = entry.get("gate_tau_start")
                                            if "gate_tau_end" in entry:
                                                parent_metadata["gate_tau_end"] = entry.get("gate_tau_end")
                                            if "gate_seed" in entry:
                                                parent_metadata["gate_seed"] = entry.get("gate_seed")
                                            break
                            
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
                ft_result = ray.get(done_ref)
                if collect_results and isinstance(ft_result, dict) and ft_result:
                    collected_b_results.append(ft_result)

        if not collect_results:
            return None

        # Build a lightweight, JSON-serializable payload for downstream aggregation.
        import datetime

        def _strip_large_fields(d: dict) -> dict:
            if not isinstance(d, dict):
                return {}
            return {
                k: v
                for k, v in d.items()
                if k
                not in [
                    "forget_logits_dict",
                    "retain_logits_dict",
                    "retain_logits_5_shot_dict",
                    "samples",
                ]
            }

        # Prefer the saved checkpoint path when available; otherwise use the configured `name`
        # as a stable identifier for this run (useful when checkpoints are not saved).
        a_path = model_path if model_path else name

        a_payload = {
            "type": "A",
            "path": a_path,
            "saved": bool(model_was_saved) if "model_was_saved" in locals() else False,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_name": run_name,
            "method": unlearn_type.name if hasattr(unlearn_type, "name") else str(unlearn_type),
            "dataset": dataset.name if hasattr(dataset, "name") else str(dataset),
            "project": wandb_project_name,
            "model_id": base_model,
            "lr": lr,
            "epochs": epochs,
            "retain_coeff": retain_coeff,
            "data_seed": data_seed,
            "eval_split_ids": list(eval_split_ids) if eval_split_ids is not None else None,
            "num_total_splits": num_total_splits,
            "lora_rank": lora_rank,
            "steering_coeff": steering_coeff,
            "forget_accs": forget_accs,
            "forget_accs_calibrated": forget_accs_calibrated,
            "retain_accs": retain_accs,
            "retain_accs_calibrated": retain_accs_calibrated,
            "retain_accs_5_shot": retain_accs_5_shot,
            "retain_accs_5_shot_calibrated": retain_accs_5_shot_calibrated,
        }
        if gate_metadata:
            a_payload.update(_strip_large_fields(gate_metadata))
            if "lora_layer_budget_k" in gate_metadata:
                a_payload["lora_layer_budget_k"] = gate_metadata.get("lora_layer_budget_k")
            if "selected_blocks" in gate_metadata:
                a_payload["selected_blocks"] = gate_metadata.get("selected_blocks")
            if "final_gate_scores" in gate_metadata:
                a_payload["final_gate_scores"] = gate_metadata.get("final_gate_scores")
            if "gate_tau_start" in gate_metadata:
                a_payload["gate_tau_start"] = gate_metadata.get("gate_tau_start")
            if "gate_tau_end" in gate_metadata:
                a_payload["gate_tau_end"] = gate_metadata.get("gate_tau_end")
            if "gate_seed" in gate_metadata:
                a_payload["gate_seed"] = gate_metadata.get("gate_seed")

        # Normalize finetune results to match the notebook-style schema.
        normalized_b: list[dict] = []
        for b in collected_b_results:
            b_clean = _strip_large_fields(b)
            b_payload = {
                "type": "B",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "run_name": run_name,
                "method": unlearn_type.name if hasattr(unlearn_type, "name") else str(unlearn_type),
                "dataset": dataset.name if hasattr(dataset, "name") else str(dataset),
                "project": wandb_project_name,
                "model_id": base_model,
                "a_path": a_path,
                "a_lr": lr,
                "a_epochs": epochs,
                "retain_coeff": retain_coeff,
                "lora_rank": lora_rank,
            }
            # Copy over the FT-specific fields we care about (and normalize key names).
            if "loss_type" in b_clean:
                b_payload["loss_type"] = b_clean.get("loss_type")
            if "lr" in b_clean:
                b_payload["lr"] = b_clean.get("lr")
            if "epochs" in b_clean:
                b_payload["epochs"] = b_clean.get("epochs")
            if "skip_split" in b_clean:
                b_payload["skip_split"] = b_clean.get("skip_split")
            if "checkpoint_type" in b_clean:
                b_payload["checkpoint_type"] = b_clean.get("checkpoint_type")
            if "name" in b_clean:
                b_payload["path"] = b_clean.get("name")
            elif "save_name" in b_clean and b_clean.get("save_name"):
                b_payload["path"] = b_clean.get("save_name")
            else:
                b_payload["path"] = None

            # Prefer standardized keys if available; fall back to finetune's *_local names.
            b_payload["forget_accs"] = b_clean.get("forget_accs") or b_clean.get("forget_accs_local")
            b_payload["forget_accs_calibrated"] = (
                b_clean.get("forget_accs_calibrated")
                or b_clean.get("forget_accs_calibrated_local")
            )
            b_payload["retain_accs"] = b_clean.get("retain_accs") or b_clean.get("retain_accs_local")
            b_payload["retain_accs_calibrated"] = (
                b_clean.get("retain_accs_calibrated")
                or b_clean.get("retain_accs_calibrated_local")
            )
            normalized_b.append(b_payload)

        return {"A": a_payload, "B": normalized_b}

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
        write_raw_results = bool(OmegaConf.select(cfg, "write_raw_results", default=False))
        raw_results: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "Baseline": []} if write_raw_results else {}
        
        # Baseline RTT tracking - deduplicate across hyperparameter points
        scheduled_baseline_rtt: set = set()  # (model_id, dataset.name, rtt_sig)
        baseline_rtt_refs: list = []
        baseline_rtt_ref_metadata: dict = {}

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
            # Check if matched forgetting is enabled for LORA
            matched_forgetting_enabled = OmegaConf.select(
                cfg, "matched_forgetting.enabled", default=False
            )
            
            # Dictionary to store selected checkpoints for matched forgetting
            # Defined outside loop so it's accessible for RTT scheduling
            selected_checkpoints = {}  # (dataset, lora_rank) -> selected_model_path
            
            for unlearn_type in unlearn_types:
                unlearn_type_config = unlearn_types_config[
                    unlearn_type.name
                ] 
                unlearn_loss_type_str =  unlearn_type_config["loss_type"]
                unlearn_loss_type = LossType[unlearn_loss_type_str]
                
                # Extract gate config keys and add to config_flat for hydra_dict
                layer_selection_mode = unlearn_type_config.get("layer_selection_mode", "none")
                lora_layer_budget_k = unlearn_type_config.get("lora_layer_budget_k", None)
                gate_tau_start = unlearn_type_config.get("gate_tau_start", 10.0)
                gate_tau_end = unlearn_type_config.get("gate_tau_end", 0.1)
                gate_warmup_steps = unlearn_type_config.get("gate_warmup_steps", 0)
                gate_seed = unlearn_type_config.get("gate_seed", None)
                gate_reg_coeff = unlearn_type_config.get("gate_reg_coeff", 0.0)
                # Add to config_flat so they're available in hydra_dict
                config_flat["layer_selection_mode"] = layer_selection_mode
                if lora_layer_budget_k is not None:
                    config_flat["lora_layer_budget_k"] = lora_layer_budget_k
                config_flat["gate_tau_start"] = gate_tau_start
                config_flat["gate_tau_end"] = gate_tau_end
                config_flat["gate_warmup_steps"] = gate_warmup_steps
                if gate_seed is not None:
                    config_flat["gate_seed"] = gate_seed
                config_flat["gate_reg_coeff"] = gate_reg_coeff
                
                # Matched forgetting path for LORA
                if matched_forgetting_enabled and unlearn_type == UnlearnType.LORA:
                    # Load matched forgetting config
                    target_acc = OmegaConf.select(
                        cfg, "matched_forgetting.target_forget_acc", default=0.60
                    )
                    tolerance = OmegaConf.select(
                        cfg, "matched_forgetting.tolerance", default=0.02
                    )
                    max_trials = OmegaConf.select(
                        cfg, "matched_forgetting.max_trials_per_rank", default=18
                    )
                    acc_selection_rule = OmegaConf.select(
                        cfg, "matched_forgetting.acc_selection_rule", default="final_epoch"
                    )
                    selection_priority = OmegaConf.select(
                        cfg, "matched_forgetting.selection_priority",
                        default=["retain_damage", "compute", "retain_coeff"]
                    )
                    save_all_candidates = OmegaConf.select(
                        cfg, "matched_forgetting.save_all_candidates", default=True
                    )
                    
                    # Get search space (can override per-dataset)
                    search_space = OmegaConf.select(
                        cfg, "matched_forgetting.search_space", default={}
                    )
                    # Parse and validate search space config
                    rc_range = parse_and_validate_list_config(
                        search_space, "rc_range", [0.001, 0.01, 0.1, 1.0], param_name="rc_range"
                    )
                    rc_add = parse_and_validate_list_config(
                        search_space, "rc_add", [], allow_empty=True, param_name="rc_add"
                    )
                    rcs = rc_range + rc_add
                    
                    lrs = parse_and_validate_list_config(
                        search_space, "lr_range", [4e-7], param_name="lr_range"
                    )
                    epochs_lst = parse_and_validate_list_config(
                        search_space, "epochs_range", [6], param_name="epochs_range"
                    )
                    
                    # Compute baseline retain accuracy (once per dataset)
                    baseline_retain_accs = {}
                    valid_datasets_for_mf = []  # Datasets that passed baseline evaluation
                    print(f"\n{'='*60}")
                    print(f"Matched Forgetting: Computing baseline retain accuracies")
                    print(f"{'='*60}\n")
                    for dataset in datasets:
                        if dataset not in baseline_retain_accs:
                            dataset_dict = dataset_dicts[dataset]
                            print(f"Evaluating baseline retain accuracy for {dataset.name}...")
                            try:
                                baseline_ref = evaluate_baseline_retain_acc.remote(
                                    model_id=model_id,
                                    val_retain_files=dataset_dict["val_retain_files"],
                                    retain_dev_file=dataset_dict["retain_dev_file"],
                                    data_root=data_root,
                                    val_batch_size=val_batch_size,
                                    attn_backend=attn_backend,
                                )
                                baseline_retain_accs[dataset] = ray.get(baseline_ref)
                                print(f"  Baseline retain accuracy: {baseline_retain_accs[dataset]:.4f}")
                                valid_datasets_for_mf.append(dataset)
                            except Exception as e:
                                print(f"  ERROR: Failed to compute baseline retain acc for {dataset.name}: {e}")
                                print(f"  Skipping matched forgetting for {dataset.name}")
                                # Skip this dataset for matched forgetting
                    
                    if not valid_datasets_for_mf:
                        print(f"\n{'='*60}")
                        print(f"ERROR: No datasets passed baseline retain accuracy evaluation")
                        print(f"Skipping matched forgetting entirely")
                        print(f"{'='*60}\n")
                        continue  # Skip matched forgetting for this unlearn_type
                    
                    # Update datasets list to only valid ones
                    datasets = valid_datasets_for_mf
                    
                    # Check if learned top-K is enabled
                    if layer_selection_mode == "learned_topk_hard":
                        # Sweep over K values (with fixed rank)
                        # Use first lora_rank from config, or require single rank
                        if len(lora_ranks) > 1:
                            raise ValueError("Matched forgetting with learned_topk_hard requires single lora_rank")
                        fixed_lora_rank = lora_ranks[0]
                        
                        # Parse K budget list (similar to how rc_range is parsed)
                        # Convert OmegaConf ListConfig to Python list if needed
                        from omegaconf import ListConfig, DictConfig
                        # Always convert OmegaConf objects to native Python types first
                        # Try multiple methods to ensure conversion
                        try:
                            if OmegaConf.is_config(lora_layer_budget_k):
                                lora_layer_budget_k = OmegaConf.to_container(lora_layer_budget_k, resolve=True)
                            elif isinstance(lora_layer_budget_k, ListConfig):
                                lora_layer_budget_k = list(lora_layer_budget_k)
                        except Exception:
                            pass  # If conversion fails, try isinstance checks below
                        
                        # Now check the native Python type
                        if isinstance(lora_layer_budget_k, (list, tuple)):
                            k_values = list(lora_layer_budget_k) if isinstance(lora_layer_budget_k, tuple) else lora_layer_budget_k
                        elif isinstance(lora_layer_budget_k, dict):
                            # Support range syntax like rc_range
                            k_range = parse_and_validate_list_config(
                                {"k_range": lora_layer_budget_k.get("range", [])},
                                "k_range", [2, 4, 8], param_name="k_range"
                            )
                            k_add = parse_and_validate_list_config(
                                {"k_add": lora_layer_budget_k.get("add", [])},
                                "k_add", [], allow_empty=True, param_name="k_add"
                            )
                            k_values = k_range + k_add
                        else:
                            # Last resort: try to convert to list if it's iterable
                            try:
                                k_values = list(lora_layer_budget_k)
                            except (TypeError, ValueError):
                                raise ValueError(f"Invalid lora_layer_budget_k config: {lora_layer_budget_k} (type: {type(lora_layer_budget_k)})")
                        
                        # Per-K matched forgetting search
                        print(f"\n{'='*60}")
                        print(f"Matched Forgetting: Starting K-sweep grid search")
                        print(f"Target forget accuracy: {target_acc} ± {tolerance}")
                        print(f"Max trials per K: {max_trials}")
                        print(f"Fixed rank: {fixed_lora_rank}, K values: {k_values}")
                        print(f"{'='*60}\n")
                        
                        for dataset in datasets:
                            dataset_dict = dataset_dicts[dataset]
                            eval_split_ids = eval_split_ids_by_dataset[dataset]
                            num_total_splits = num_total_splits_by_dataset[dataset]
                            
                            for k_budget in k_values:
                                print(f"\nMatched Forgetting: {dataset.name}, rank {fixed_lora_rank}, K={k_budget}")
                                candidates = []
                                candidate_refs = []
                                
                                # Generate candidate hyperparameter sets
                                candidate_count = 0
                                for epochs in epochs_lst:
                                    for lr in lrs:
                                        for rc in rcs:
                                            if candidate_count >= max_trials:
                                                break
                                            
                                            model_id_safe = model_id.replace('/', '_')
                                            # Include K in save name for K-sweep
                                            candidate_path = (
                                                f"models/{run_name}/"
                                                f"{unlearn_type.name}/{dataset.name}/"
                                                f"{wandb_project_name}/"
                                                f"rank{fixed_lora_rank}-k{k_budget}-sc20-{model_id_safe}-rc{rc}-lr{lr}-epochs{epochs}"
                                            )
                                            save_name_candidate = candidate_path if (save_unlearn_model and save_all_candidates) else None
                                            
                                            # Update config_flat with current k_budget for this candidate
                                            config_flat["lora_layer_budget_k"] = k_budget
                                            
                                            # Launch unlearn run
                                            ref = main.remote(
                                                unlearn_type=unlearn_type,
                                                dataset=dataset,
                                                unlearn_files=dataset_dict["unlearn_files"],
                                                wrong_unlearn_files=dataset_dict.get("wrong_unlearn_files", []),
                                                fixed_wrong_unlearn_files=dataset_dict.get("fixed_wrong_unlearn_files", []),
                                                val_files=dataset_dict["val_files"],
                                                dev_file=dataset_dict["dev_file"],
                                                retain_files=dataset_dict["retain_files"],
                                                val_retain_files=dataset_dict["val_retain_files"],
                                                retain_dev_file=dataset_dict["retain_dev_file"],
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
                                                save_name=save_name_candidate,
                                                name=candidate_path,
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
                                                dont_ft=True,  # Don't run RTT during search
                                                unlearn_freeze_layers=unlearn_freeze_layers,
                                                ft_freeze_layers=ft_freeze_layers,
                                                ft_dont_eval=ft_dont_eval,
                                                unlearn_mcq=unlearn_mcq,
                                                hydra_dict=config_flat,
                                                unlearn_data_format=unlearn_data_format,
                                                ft_data_format=ft_data_format,
                                                unlearn_loss_type=unlearn_loss_type,
                                                steering_coeff=20,
                                                max_samples=max_samples_lst[0] if max_samples_lst else 999999999,
                                                lora_rank=fixed_lora_rank,
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
                                            )
                                            candidate_refs.append((ref, {
                                                "epochs": epochs,
                                                "lr": lr,
                                                "rc": rc,
                                                "lora_rank": fixed_lora_rank,
                                                "k_budget": k_budget,
                                                "model_path": candidate_path,
                                                "save_name": save_name_candidate,
                                            }))
                                            candidate_count += 1
                                
                                # Collect results and select best candidate for this K
                                print(f"  Waiting for {len(candidate_refs)} candidates to complete...")
                                for ref, candidate_meta in candidate_refs:
                                    try:
                                        (
                                            model_path,
                                            forget_accs, forget_accs_calibrated, forget_logits_dict,
                                            retain_accs, retain_accs_calibrated, retain_logits_dict,
                                            retain_accs_5_shot, retain_accs_5_shot_calibrated,
                                            retain_logits_5_shot_dict,
                                            samples,
                                            gate_metadata
                                        ) = ray.get(ref)
                                        
                                        forget_acc = extract_avg_acc(forget_accs, acc_selection_rule)
                                        retain_acc = extract_avg_acc(retain_accs, acc_selection_rule)
                                        
                                        candidate_meta.update({
                                            "forget_acc": forget_acc,
                                            "retain_acc": retain_acc,
                                            "model_path": model_path,
                                            "gate_metadata": gate_metadata,
                                        })
                                        candidates.append(candidate_meta)
                                        print(f"    Candidate (rc={candidate_meta['rc']}, lr={candidate_meta['lr']}, epochs={candidate_meta['epochs']}): "
                                              f"forget_acc={forget_acc:.4f}, retain_acc={retain_acc:.4f}")
                                    except Exception as e:
                                        print(f"    ERROR: Failed to process candidate {candidate_meta}: {e}")
                                        continue
                                
                                # Select best candidate for this K
                                selected = select_matched_forgetting_candidate(
                                    candidates=candidates,
                                    target_acc=target_acc,
                                    tolerance=tolerance,
                                    baseline_retain_acc=baseline_retain_accs[dataset],
                                    selection_priority=selection_priority,
                                )
                                
                                if selected:
                                    print(f"  Selected: rc={selected['rc']}, lr={selected['lr']}, epochs={selected['epochs']}")
                                    print(f"    forget_acc={selected['forget_acc']:.4f}, retain_acc={selected['retain_acc']:.4f}")
                                    
                                    # Write matched_forgetting.json entry with file locking
                                    selection_data = {
                                        "selected_hparams": {
                                            "epochs": selected["epochs"],
                                            "lr": selected["lr"],
                                            "rc": selected["rc"],
                                        },
                                        "achieved_forget_acc": selected["forget_acc"],
                                        "achieved_retain_acc": selected["retain_acc"],
                                        "model_path": selected["model_path"],
                                        "baseline_retain_acc": baseline_retain_accs[dataset],
                                        "retain_damage": baseline_retain_accs[dataset] - selected["retain_acc"],
                                    }
                                    # Store full gate_metadata if present
                                    if selected.get("gate_metadata"):
                                        selection_data["gate_metadata"] = selected["gate_metadata"]
                                    
                                    write_matched_forgetting_json(
                                        run_name=run_name,
                                        dataset_name=dataset.name,
                                        lora_rank=fixed_lora_rank,
                                        k_budget=k_budget,
                                        selection_data=selection_data,
                                    )
                                    
                                    # Write manifest entry with matched_forgetting tag
                                    a_metadata = {
                                        "run_name": run_name,
                                        "method": unlearn_type.name,
                                        "dataset": dataset.name,
                                        "project": wandb_project_name,
                                        "model_id": model_id,
                                        "lr": selected["lr"],
                                        "epochs": selected["epochs"],
                                        "retain_coeff": selected["rc"],
                                        "lora_rank": fixed_lora_rank,
                                    }
                                    # Add gate metadata if present
                                    if selected.get("gate_metadata"):
                                        gate_meta = selected["gate_metadata"]
                                        a_metadata["lora_layer_budget_k"] = gate_meta.get("lora_layer_budget_k")
                                        a_metadata["selected_blocks"] = gate_meta.get("selected_blocks")
                                        a_metadata["final_gate_scores"] = gate_meta.get("final_gate_scores")
                                        a_metadata["gate_tau_start"] = gate_meta.get("gate_tau_start")
                                        a_metadata["gate_tau_end"] = gate_meta.get("gate_tau_end")
                                        a_metadata["gate_seed"] = gate_meta.get("gate_seed")
                                    
                                    write_checkpoint_manifest_entry(
                                        run_name=run_name,
                                        checkpoint_type="A",
                                        checkpoint_path=selected["model_path"],
                                        metadata=a_metadata,
                                        tags=["matched_forgetting"],
                                    )
                                    
                                    # Store selected checkpoint for RTT phase (key includes k_budget)
                                    selected_checkpoints[(dataset, fixed_lora_rank, k_budget)] = selected["model_path"]
                                else:
                                    print(f"  WARNING: No candidate selected for {dataset.name} rank {fixed_lora_rank} K={k_budget}")
                    else:
                        # Original matched forgetting (sweep over ranks)
                        print(f"\n{'='*60}")
                        print(f"Matched Forgetting: Starting grid search")
                        print(f"Target forget accuracy: {target_acc} ± {tolerance}")
                        print(f"Max trials per rank: {max_trials}")
                        print(f"{'='*60}\n")
                        
                        for dataset in datasets:
                            dataset_dict = dataset_dicts[dataset]
                            eval_split_ids = eval_split_ids_by_dataset[dataset]
                            num_total_splits = num_total_splits_by_dataset[dataset]
                            
                            for lora_rank in lora_ranks:
                                print(f"\nMatched Forgetting: {dataset.name}, rank {lora_rank}")
                                candidates = []
                                candidate_refs = []
                                
                                # Generate candidate hyperparameter sets
                                candidate_count = 0
                                for epochs in epochs_lst:
                                    for lr in lrs:
                                        for rc in rcs:
                                            if candidate_count >= max_trials:
                                                break
                                            
                                            model_id_safe = model_id.replace('/', '_')
                                            candidate_path = (
                                                f"models/{run_name}/"
                                                f"{unlearn_type.name}/{dataset.name}/"
                                                f"{wandb_project_name}/"
                                                f"rank{lora_rank}-sc20-{model_id_safe}-rc{rc}-lr{lr}-epochs{epochs}"
                                            )
                                        save_name_candidate = candidate_path if (save_unlearn_model and save_all_candidates) else None
                                        
                                        # Launch unlearn run
                                        ref = main.remote(
                                            unlearn_type=unlearn_type,
                                            dataset=dataset,
                                            unlearn_files=dataset_dict["unlearn_files"],
                                            wrong_unlearn_files=dataset_dict.get("wrong_unlearn_files", []),
                                            fixed_wrong_unlearn_files=dataset_dict.get("fixed_wrong_unlearn_files", []),
                                            val_files=dataset_dict["val_files"],
                                            dev_file=dataset_dict["dev_file"],
                                            retain_files=dataset_dict["retain_files"],
                                            val_retain_files=dataset_dict["val_retain_files"],
                                            retain_dev_file=dataset_dict["retain_dev_file"],
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
                                            save_name=save_name_candidate,
                                            name=candidate_path,
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
                                            dont_ft=True,  # Don't run RTT during search
                                            unlearn_freeze_layers=unlearn_freeze_layers,
                                            ft_freeze_layers=ft_freeze_layers,
                                            ft_dont_eval=ft_dont_eval,
                                            unlearn_mcq=unlearn_mcq,
                                            hydra_dict=config_flat,
                                            unlearn_data_format=unlearn_data_format,
                                            ft_data_format=ft_data_format,
                                            unlearn_loss_type=unlearn_loss_type,
                                            steering_coeff=20,
                                            max_samples=max_samples_lst[0] if max_samples_lst else 999999999,
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
                                        )
                                        candidate_refs.append((ref, {
                                            "epochs": epochs,
                                            "lr": lr,
                                            "rc": rc,
                                            "lora_rank": lora_rank,
                                            "model_path": candidate_path,
                                            "save_name": save_name_candidate,
                                        }))
                                        candidate_count += 1
                            
                            # Collect results and select best candidate
                            print(f"  Waiting for {len(candidate_refs)} candidates to complete...")
                            for ref, candidate_meta in candidate_refs:
                                try:
                                    (
                                        model_path,
                                        forget_accs, forget_accs_calibrated, forget_logits_dict,
                                        retain_accs, retain_accs_calibrated, retain_logits_dict,
                                        retain_accs_5_shot, retain_accs_5_shot_calibrated,
                                        retain_logits_5_shot_dict,
                                        samples,
                                        gate_metadata
                                    ) = ray.get(ref)
                                    
                                    forget_acc = extract_avg_acc(forget_accs, acc_selection_rule)
                                    retain_acc = extract_avg_acc(retain_accs, acc_selection_rule)
                                    
                                    candidate_meta.update({
                                        "forget_acc": forget_acc,
                                        "retain_acc": retain_acc,
                                        "model_path": model_path,
                                        "gate_metadata": gate_metadata,  # Store gate metadata for manifest
                                    })
                                    candidates.append(candidate_meta)
                                    print(f"    Candidate (rc={candidate_meta['rc']}, lr={candidate_meta['lr']}, epochs={candidate_meta['epochs']}): "
                                          f"forget_acc={forget_acc:.4f}, retain_acc={retain_acc:.4f}")
                                except Exception as e:
                                    print(f"    ERROR: Failed to process candidate {candidate_meta}: {e}")
                                    continue
                            
                            # Check if any candidates succeeded
                            if not candidates:
                                print(f"  ERROR: All candidates failed for {dataset.name} rank {lora_rank}")
                                print(f"  Skipping matched forgetting for this configuration")
                                continue  # Skip to next rank
                            
                            # Select best candidate
                            selected = select_matched_checkpoint(
                                candidates,
                                target_acc,
                                tolerance,
                                baseline_retain_accs[dataset],
                                selection_priority,
                            )
                            
                            if selected:
                                print(f"  Selected: rc={selected['rc']}, lr={selected['lr']}, epochs={selected['epochs']}")
                                print(f"    forget_acc={selected['forget_acc']:.4f}, retain_acc={selected['retain_acc']:.4f}")
                                
                                # Ensure selected checkpoint is saved if it wasn't saved during search
                                if not selected.get("save_name") and save_unlearn_model:
                                    # Re-save selected model
                                    selected["save_name"] = selected["model_path"]
                                    # Note: We'd need to re-run unlearning to save, but for now we'll just mark it
                                    # In practice, if save_all_candidates=False, we should save the selected one
                                
                                # Write matched_forgetting.json entry with file locking
                                selection_data = {
                                    "selected_hparams": {
                                        "epochs": selected["epochs"],
                                        "lr": selected["lr"],
                                        "rc": selected["rc"],
                                    },
                                    "achieved_forget_acc": selected["forget_acc"],
                                    "achieved_retain_acc": selected["retain_acc"],
                                    "model_path": selected["model_path"],
                                    "baseline_retain_acc": baseline_retain_accs[dataset],
                                    "retain_damage": baseline_retain_accs[dataset] - selected["retain_acc"],
                                }
                                # Store full gate_metadata if present
                                if selected.get("gate_metadata"):
                                    selection_data["gate_metadata"] = selected["gate_metadata"]
                                
                                write_matched_forgetting_json(
                                    run_name=run_name,
                                    dataset_name=dataset.name,
                                    lora_rank=lora_rank,
                                    selection_data=selection_data,
                                )
                                
                                # Write manifest entry with matched_forgetting tag
                                a_metadata = {
                                    "run_name": run_name,
                                    "method": unlearn_type.name,
                                    "dataset": dataset.name,
                                    "project": wandb_project_name,
                                    "model_id": model_id,
                                    "lr": selected["lr"],
                                    "epochs": selected["epochs"],
                                    "retain_coeff": selected["rc"],
                                    "lora_rank": lora_rank,
                                }
                                # Add gate metadata if present
                                if selected.get("gate_metadata"):
                                    gate_meta = selected["gate_metadata"]
                                    a_metadata["lora_layer_budget_k"] = gate_meta.get("lora_layer_budget_k")
                                    a_metadata["selected_blocks"] = gate_meta.get("selected_blocks")
                                    a_metadata["final_gate_scores"] = gate_meta.get("final_gate_scores")
                                    a_metadata["gate_tau_start"] = gate_meta.get("gate_tau_start")
                                    a_metadata["gate_tau_end"] = gate_meta.get("gate_tau_end")
                                    a_metadata["gate_seed"] = gate_meta.get("gate_seed")
                                
                                write_checkpoint_manifest_entry(
                                    run_name=run_name,
                                    checkpoint_type="A",
                                    checkpoint_path=selected["model_path"],
                                    metadata=a_metadata,
                                    tags=["matched_forgetting"],
                                )
                                
                                # Store selected checkpoint for RTT phase
                                selected_checkpoints[(dataset, lora_rank)] = selected["model_path"]
                            else:
                                print(f"  WARNING: No candidate selected for {dataset.name} rank {lora_rank}")
                    
                    print(f"\n{'='*60}")
                    print(f"Matched Forgetting: Search complete")
                    print(f"Selected {len(selected_checkpoints)} checkpoints")
                    print(f"{'='*60}\n")
                    
                    # Skip to RTT phase (which will be handled separately below)
                    continue
                
                # Original unlearning loop (existing code)
                # Extract gate config keys and add to config_flat for hydra_dict (if not already done)
                if "layer_selection_mode" not in config_flat:
                    layer_selection_mode = unlearn_type_config.get("layer_selection_mode", "none")
                    lora_layer_budget_k = unlearn_type_config.get("lora_layer_budget_k", None)
                    gate_tau_start = unlearn_type_config.get("gate_tau_start", 10.0)
                    gate_tau_end = unlearn_type_config.get("gate_tau_end", 0.1)
                    gate_warmup_steps = unlearn_type_config.get("gate_warmup_steps", 0)
                    gate_seed = unlearn_type_config.get("gate_seed", None)
                    gate_reg_coeff = unlearn_type_config.get("gate_reg_coeff", 0.0)
                    # Add to config_flat so they're available in hydra_dict
                    config_flat["layer_selection_mode"] = layer_selection_mode
                    # Handle K value sweep: if lora_layer_budget_k is a list, iterate over it
                    if lora_layer_budget_k is not None:
                        if isinstance(lora_layer_budget_k, list):
                            # Will iterate over K values in the loop below
                            k_values_to_sweep = lora_layer_budget_k
                        else:
                            # Single K value - wrap in list for consistent handling
                            k_values_to_sweep = [lora_layer_budget_k]
                    else:
                        k_values_to_sweep = [None]  # No K sweep
                    config_flat["gate_tau_start"] = gate_tau_start
                    config_flat["gate_tau_end"] = gate_tau_end
                    config_flat["gate_warmup_steps"] = gate_warmup_steps
                    if gate_seed is not None:
                        config_flat["gate_seed"] = gate_seed
                    config_flat["gate_reg_coeff"] = gate_reg_coeff
                else:
                    layer_selection_mode = config_flat.get("layer_selection_mode", "none")
                    lora_layer_budget_k = config_flat.get("lora_layer_budget_k", None)
                    # Handle K value sweep: if lora_layer_budget_k is a list, iterate over it
                    # Convert OmegaConf ListConfig to Python list if needed
                    from omegaconf import ListConfig
                    if lora_layer_budget_k is not None:
                        # Convert OmegaConf objects to native Python types first
                        if OmegaConf.is_config(lora_layer_budget_k):
                            lora_layer_budget_k = OmegaConf.to_container(lora_layer_budget_k, resolve=True)
                        # Now check the native Python type
                        if isinstance(lora_layer_budget_k, (list, tuple)):
                            # Will iterate over K values in the loop below
                            k_values_to_sweep = list(lora_layer_budget_k) if isinstance(lora_layer_budget_k, tuple) else lora_layer_budget_k
                        else:
                            # Single K value - wrap in list for consistent handling
                            k_values_to_sweep = [lora_layer_budget_k]
                    else:
                        k_values_to_sweep = [None]  # No K sweep
                
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
                                            # Iterate over K values if learned top-K is enabled
                                            for k_budget in k_values_to_sweep:
                                                model_id_safe = model_id.replace('/', '_')
                                                # Include K in save name if learned top-K is enabled
                                                if layer_selection_mode == "learned_topk_hard" and k_budget is not None:
                                                    forget_model = (
                                                        f"models/{run_name}/"
                                                        f"{unlearn_type.name}/"
                                                        f"{dataset.name}/"
                                                        f"{wandb_project_name}/"
                                                        f"rank{lora_rank}-k{k_budget}-sc{sc}-"
                                                        f"{model_id_safe}-rc{rc}-lr{lr}-"
                                                        f"epochs{epochs}"
                                                    )
                                                else:
                                                    forget_model = (
                                                        f"models/{run_name}/"
                                                        f"{unlearn_type.name}/"
                                                        f"{dataset.name}/"
                                                        f"{wandb_project_name}/"
                                                        f"rank{lora_rank}-sc{sc}-"
                                                        f"{model_id_safe}-rc{rc}-lr{lr}-"
                                                        f"epochs{epochs}"
                                                    )
                                                save_name = (
                                                    forget_model if save_unlearn_model
                                                    else None
                                                )
                                                # Create a copy of config_flat for this run to avoid mutating shared dict
                                                # This is critical when iterating over K values - each remote call needs its own copy
                                                config_flat_copy = config_flat.copy()
                                                if layer_selection_mode == "learned_topk_hard" and k_budget is not None:
                                                    config_flat_copy["lora_layer_budget_k"] = k_budget
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
                                                hydra_dict=config_flat_copy,
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
                                                collect_results=write_raw_results,
                                            )]

            # Schedule RTT for matched forgetting checkpoints (B)
            # Only schedule if RTT is enabled and matched forgetting was used
            if not dont_ft and not ft_on_all and matched_forgetting_enabled and selected_checkpoints:
                import finetune_corpus
                print(f"\n{'='*60}")
                print(f"Matched Forgetting: Scheduling RTT on selected checkpoints")
                print(f"{'='*60}\n")
                
                for key, selected_model_path in selected_checkpoints.items():
                    # Handle both key formats: (dataset, lora_rank) or (dataset, lora_rank, k_budget)
                    if len(key) == 2:
                        dataset, lora_rank = key
                        k_budget = None
                    elif len(key) == 3:
                        dataset, lora_rank, k_budget = key
                    else:
                        raise ValueError(f"Unexpected selected_checkpoints key format: {key}")
                    dataset_dict = dataset_dicts[dataset]
                    eval_split_ids = eval_split_ids_by_dataset[dataset]
                    num_total_splits = num_total_splits_by_dataset[dataset]
                    
                    # Read matched_forgetting.json to get hyperparameters
                    # Initialize with defaults
                    unlearn_lr = DEFAULT_UNLEARN_LR
                    unlearn_epochs = DEFAULT_UNLEARN_EPOCHS
                    retain_coeff = DEFAULT_RETAIN_COEFF
                    
                    matched_forgetting_path = os.path.join("models", run_name, "matched_forgetting.json")
                    matched_data = _read_json_with_lock(matched_forgetting_path, default={})
                    # Handle both key formats: rank-only or rank_K
                    if k_budget is not None:
                        key = f"rank{lora_rank}_k{k_budget}"
                    else:
                        key = str(lora_rank)
                    if matched_data and dataset.name in matched_data and key in matched_data[dataset.name]:
                        selected_hparams = matched_data[dataset.name][key].get("selected_hparams", {})
                        unlearn_lr = selected_hparams.get("lr", DEFAULT_UNLEARN_LR)
                        unlearn_epochs = selected_hparams.get("epochs", DEFAULT_UNLEARN_EPOCHS)
                        retain_coeff = selected_hparams.get("rc", DEFAULT_RETAIN_COEFF)
                    
                    for loss_type in ft_loss_types:
                        for ft_lr in ft_lrs:
                            for ft_epochs in ft_epochs_lst:
                                if not ft_on_all:
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
                                        ft_val_retain_files = dataset_dict.get(
                                            "val_retain_files", ft_files.copy()
                                        )
                                        
                                        # Extract remaining path after 'models/{run_name}/'
                                        remaining_path = extract_model_path_suffix(selected_model_path, run_name)
                                        fted_model_path = (
                                            f"models/{run_name}/fted/"
                                            f"{remaining_path}/"
                                            f"{wandb_project_name}/"
                                            f"{loss_type.name}/ft-skip_split{skip_split}/"
                                            f"lr{ft_lr}-epoch{ft_epochs}"
                                        )
                                        
                                        # Prepare parent metadata for B checkpoint (A checkpoint info)
                                        parent_metadata = build_parent_metadata(
                                            run_name=run_name,
                                            unlearn_type=UnlearnType.LORA,
                                            dataset=dataset,
                                            wandb_project_name=wandb_project_name,
                                            base_model=model_id,
                                            unlearn_lr=unlearn_lr,
                                            unlearn_epochs=unlearn_epochs,
                                            retain_coeff=retain_coeff,
                                            lora_rank=lora_rank,
                                            steering_coeff=None,
                                        )
                                        # Add gate metadata from matched_forgetting.json if present
                                        if matched_data and dataset.name in matched_data and key in matched_data[dataset.name]:
                                            mf_entry = matched_data[dataset.name][key]
                                            if "gate_metadata" in mf_entry:
                                                # Full gate metadata is available - add to parent_metadata for inheritance
                                                gate_meta = mf_entry["gate_metadata"]
                                                parent_metadata.update({
                                                    "lora_layer_budget_k": gate_meta.get("lora_layer_budget_k"),
                                                    "selected_blocks": gate_meta.get("selected_blocks"),
                                                    "final_gate_scores": gate_meta.get("final_gate_scores"),
                                                    "gate_tau_start": gate_meta.get("gate_tau_start"),
                                                    "gate_tau_end": gate_meta.get("gate_tau_end"),
                                                    "gate_seed": gate_meta.get("gate_seed"),
                                                })
                                        
                                        ref = finetune_corpus.main.remote(
                                            train_files=ft_files,
                                            val_files=ft_val_files,
                                            val_retain_files=ft_val_retain_files,
                                            dev_set=ft_files[0] if ft_files else "",
                                            data_root=data_root,
                                            base_model=selected_model_path,  # Use selected matched checkpoint
                                            lr=ft_lr,
                                            epochs=ft_epochs,
                                            name=fted_model_path,
                                            batch_size=ft_batch_size,
                                            val_batch_size=ft_val_batch_size,
                                            save_name=fted_model_path if save_ft_models else None,
                                            loss_type=loss_type,
                                            project_name=wandb_project_name,
                                            diff_tokenizer=diff_tokenizer,
                                            freeze_layers=ft_freeze_layers,
                                            dont_eval=ft_dont_eval,
                                            hydra_dict=config_flat,
                                            data_format=ft_data_format,
                                            attn_backend=attn_backend,
                                            run_name=run_name,
                                            checkpoint_type="B",
                                            parent_metadata=parent_metadata,
                                            skip_split=skip_split,
                                        )
                                        refs.append(ref)
                                else:
                                    # ft_on_all case
                                    ft_files = dataset_dict["val_files"]
                                    ft_val_files = dataset_dict["val_files"]
                                    ft_val_retain_files = dataset_dict.get(
                                        "val_retain_files", ft_files.copy()
                                    )
                                    
                                    # Extract remaining path after 'models/{run_name}/'
                                    remaining_path = extract_model_path_suffix(selected_model_path, run_name)
                                    fted_model_path = (
                                        f"models/{run_name}/fted/"
                                        f"{remaining_path}/"
                                        f"{loss_type.name}/all_splits/lr{ft_lr}-epoch{ft_epochs}"
                                    )
                                    
                                    parent_metadata = build_parent_metadata(
                                        run_name=run_name,
                                        unlearn_type=UnlearnType.LORA,
                                        dataset=dataset,
                                        wandb_project_name=wandb_project_name,
                                        base_model=model_id,
                                        unlearn_lr=unlearn_lr,
                                        unlearn_epochs=unlearn_epochs,
                                        retain_coeff=retain_coeff,
                                        lora_rank=lora_rank,
                                        steering_coeff=None,
                                    )
                                    
                                    ref = finetune_corpus.main.remote(
                                        train_files=ft_files,
                                        val_files=ft_val_files,
                                        val_retain_files=ft_val_retain_files,
                                        dev_set=ft_files[0] if ft_files else "",
                                        data_root=data_root,
                                        base_model=selected_model_path,
                                        lr=ft_lr,
                                        epochs=ft_epochs,
                                        name=fted_model_path,
                                        batch_size=ft_batch_size,
                                        val_batch_size=ft_val_batch_size,
                                        save_name=fted_model_path if save_ft_models else None,
                                        loss_type=loss_type,
                                        project_name=wandb_project_name,
                                        diff_tokenizer=diff_tokenizer,
                                        freeze_layers=ft_freeze_layers,
                                        dont_eval=ft_dont_eval,
                                        hydra_dict=config_flat,
                                        data_format=ft_data_format,
                                        attn_backend=attn_backend,
                                        run_name=run_name,
                                        checkpoint_type="B",
                                        parent_metadata=parent_metadata,
                                        skip_split=None,
                                    )
                                    refs.append(ref)
                
                print(f"Matched Forgetting: Scheduled {len([r for r in refs if r])} RTT runs\n")

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
                                    # Skip if baseline RTT model already exists
                                    if os.path.exists(baseline_rtt_path) and os.path.exists(
                                        os.path.join(baseline_rtt_path, "config.json")
                                    ):
                                        print(
                                            f"Skipping baseline RTT - model already exists: {baseline_rtt_path}"
                                        )
                                        continue
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
                                    baseline_rtt_ref_metadata[ref] = c_metadata

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
                    collect_results=write_raw_results,
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
                                    # Skip if baseline RTT model already exists
                                    if os.path.exists(baseline_rtt_path) and os.path.exists(
                                        os.path.join(baseline_rtt_path, "config.json")
                                    ):
                                        print(
                                            f"Skipping baseline RTT - model already exists: {baseline_rtt_path}"
                                        )
                                        continue
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
                                    baseline_rtt_ref_metadata[ref] = c_metadata


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

        # Always wait for scheduled main() tasks to finish; optionally collect results.
        if refs:
            print(f"\nWaiting for {len(refs)} main() task(s) to complete...")
        for ref in refs:
            try:
                result = ray.get(ref)
                if write_raw_results and isinstance(result, dict) and result:
                    a = result.get("A")
                    b = result.get("B")
                    if isinstance(a, dict) and a:
                        raw_results["A"].append(a)
                    if isinstance(b, list) and b:
                        raw_results["B"].extend([x for x in b if isinstance(x, dict) and x])
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
                    c_result = ray.get(done_ref)
                    if write_raw_results and isinstance(c_result, dict) and c_result:
                        # Tag and store; finetune_corpus returns checkpoint_type when invoked here.
                        c_result = {k: v for k, v in c_result.items() if k not in ["forget_logits_dict", "retain_logits_dict", "samples"]}
                        c_meta = baseline_rtt_ref_metadata.get(done_ref, {}) if isinstance(baseline_rtt_ref_metadata, dict) else {}
                        # Normalize keys to align with other outputs.
                        c_payload = {
                            "type": "C",
                            "timestamp": c_result.get("timestamp"),
                            "run_name": run_name,
                            "dataset": c_meta.get("dataset") or c_result.get("dataset"),
                            "model_id": c_meta.get("model_id") or c_result.get("model_id") or c_result.get("base_model"),
                            "loss_type": c_result.get("loss_type"),
                            "lr": c_result.get("lr"),
                            "epochs": c_result.get("epochs"),
                            "skip_split": c_result.get("skip_split"),
                            "path": c_result.get("name") or c_result.get("save_name"),
                            "forget_accs": c_result.get("forget_accs") or c_result.get("forget_accs_local"),
                            "forget_accs_calibrated": c_result.get("forget_accs_calibrated") or c_result.get("forget_accs_calibrated_local"),
                            "retain_accs": c_result.get("retain_accs") or c_result.get("retain_accs_local"),
                            "retain_accs_calibrated": c_result.get("retain_accs_calibrated") or c_result.get("retain_accs_calibrated_local"),
                        }
                        raw_results["C"].append(c_payload)
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

        if write_raw_results:
            import re
            os.makedirs(os.path.join(results_dir, "summary"), exist_ok=True)
            safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name or "run")
            out_path = os.path.join(results_dir, "summary", f"raw_results_{safe_run_name}.json")
            with open(out_path, "w") as f:
                json.dump(raw_results, f, indent=2, default=str)
            print(f"Wrote raw metrics to {out_path}")

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
