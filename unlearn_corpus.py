import os
import site
from pathlib import Path
import unittest
import json
import math
import random
from typing import Optional, TypedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ray.experimental.tqdm_ray import tqdm
import wandb
import ray
from lion_pytorch import Lion
from filelock import FileLock
from enum import Enum, auto
import logging
from pipeline import UnlearnType, LossType, DataFormat
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from utils.attention_backend import get_attn_implementation

MAX_SEQ_LEN = 512


random.seed(42)
class Point(TypedDict):
    question: str
    choices: list[str]
    answer: int
    is_false: bool


test_prompts = ["Hi, my name is", "Once upon a time,", "The capital of France"]


def resolve_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def maybe_set_bnb_cuda_version() -> None:
    if os.environ.get("BNB_CUDA_VERSION"):
        return
    cuda_version = torch.version.cuda
    if not cuda_version:
        return
    parts = cuda_version.split(".")
    if len(parts) < 2:
        return
    target = f"{parts[0]}{parts[1]}"
    bb_dir = None
    for sp in site.getsitepackages():
        candidate = Path(sp) / "bitsandbytes"
        if candidate.exists():
            bb_dir = candidate
            break
    if not bb_dir:
        return
    target_lib = bb_dir / f"libbitsandbytes_cuda{target}.so"
    if target_lib.exists():
        return
    versions = []
    for lib in bb_dir.glob("libbitsandbytes_cuda*.so"):
        name = lib.name.replace("libbitsandbytes_cuda", "").replace(".so", "")
        if name.isdigit():
            versions.append(name)
    if not versions:
        return
    best = sorted(versions)[-1]
    os.environ["BNB_CUDA_VERSION"] = best
    print(
        f"BNB_CUDA_VERSION not set; falling back to {best} to match "
        f"available bitsandbytes CUDA binaries."
    )


def sample_tokens(
    model, tokenizer, device, prompts=test_prompts, max_length=15
):
    model.eval()
    generated_texts = []
    
    for prompt_text in prompts:
        input_ids = tokenizer.encode(
            prompt_text, return_tensors="pt"
        ).to(device)
        outputs = model.generate(input_ids, max_length=max_length)
        texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        generated_texts.extend(texts)
    
    return generated_texts

def create_prompt(point: Point) -> str:
    try:
        return "\n".join(
            [point["question"]]
            + [
                f"{doc_to_choice[i]}. {c}"
                for i, c in enumerate(point["choices"])
            ]
            + ["Answer:"]
        )
    except Exception as e:
        print(f"{point=}")
        raise Exception(e)

def make_k_shot(data: list[Point], dev_set: list[Point], k: int) -> list[Point]:
    try:
        if k == 0:
            return data
        preprompt = "\n\n".join(
            [
                f"{create_prompt(point)}{doc_to_choice[point['answer']]}."
                for point in dev_set[:k]
            ]
        )
        return [
            {
                "question": preprompt + "\n\n" + create_prompt(point),
                "choices": point["choices"],
                "answer": point["answer"]
            }
            for point in data
        ]
    except: 
        print(f"{locals()=}")
        raise Exception("stop")

def process_batch(
    batch: list[Point],
    device: torch.device,
    tokenizer: AutoTokenizer,
    label_possibilities: list[int],
    train_on_wrong_answer: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    print_a_prompt: bool = False,
    print_prefix: str = "prompts"
):
    prompts = [create_prompt(point) for point in batch]
    if print_a_prompt:
        print(f"{print_prefix}: {prompts}")
    tokens = tokenizer(
        prompts, return_tensors="pt", max_length=max_seq_len,
        truncation=True, padding=True
    ).to(device)

    def get_answer(point):
        if train_on_wrong_answer:
            return random.Random(point["question"]).choice(
                [i for i in range(len(doc_to_choice)) if i != point["answer"]]
            )
        else:
            return point["answer"]

    last_pos_label_ids = torch.tensor(
        [label_possibilities[get_answer(point)] for point in batch],
        device=device
    )
    return tokens, last_pos_label_ids

def get_loss_and_acc(
    model, tokens, last_pos_label_ids, label_possibilities,
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
) -> tuple[torch.Tensor, float]:
    logits = model(**model.prepare_inputs_for_generation(**tokens)).logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, last_pos_label_ids)
    label_impossibilities = list(set(range(logits.shape[1])) - set(label_possibilities))
    logits[:, label_impossibilities] = -float("inf")
    acc = (logits.argmax(dim=-1) == last_pos_label_ids).float().sum().item()
    if unlearn_type.value == UnlearnType.GD.value:
        loss = -loss
    logits_labels = logits[:, label_possibilities].detach().float().cpu().numpy()
    return loss, acc, logits_labels

doc_to_choice = ["A", "B", "C", "D"]

def create_prompt_tf(point: Point, unlearn_type: UnlearnType) -> str:
    ans = point["is_false"] if unlearn_type.value == UnlearnType.FWF.value else not point["is_false"]
    return " ".join(
        ["The statement:", point["text"], "is", "true or", "false", "Answer:"]
        + ["true" if ans else "false"]
    )

def create_prompt_text(point: Point, max_len: int = 2000,) -> str:
    return point["text"] if isinstance(point, dict) and len(point["text"]) < max_len else point["text"][:max_len]  if isinstance(point, dict) else point

def create_prompt_question_letter_answer(point: Point, unlearn_type=UnlearnType.NOT_SPECIFIED) -> str:
    if (
        unlearn_type.value != UnlearnType.FWF.value
        and unlearn_type.value != UnlearnType.WHP.value
    ):
        ans = point["answer"] 
    else:
        ans = random.randint(0, 3)
        while ans == point["answer"]:
            ans = random.randint(0, 3)

    return "\n".join(
        [point["question"]]
        + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
        + [f"Answer: {doc_to_choice[i]}. {c}"
            for i, c in enumerate(point["choices"]) if i == ans 
        ]
    )

def create_prompt_unlearn(
    point: Point, unlearn_type: UnlearnType,
    data_format: DataFormat = DataFormat.NOT_SPECIFIED,
) -> str:
    try:
        if data_format.value == DataFormat.TF.value:
            return create_prompt_tf(point, unlearn_type)
        elif data_format.value == DataFormat.MCQ:
            return create_prompt_question_letter_answer(
                point, unlearn_type=unlearn_type
            )
        elif data_format.value == DataFormat.CORPUS.value:
            return create_prompt_text(point)
        else:
            raise Exception("Non-handled data format")
    except Exception as e:
        print(f"{point=}\n\n")
        raise Exception(e)

def get_log_probs(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, : -1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probs_for_tokens 

def get_loss_corpus(
    model,
    batch: list[Point],
    device: torch.device,
    tokenizer: AutoTokenizer,
    label_possibilities: list[int],
    train_on_wrong_answer: bool = False,
    max_len: int = 2000,
    max_seq_len: int = MAX_SEQ_LEN,
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    data_format: DataFormat = DataFormat.NOT_SPECIFIED,
    print_prompts: bool = False,
    prompts_prefix: str = "prompts",
):
    prompts = [
        create_prompt_unlearn(
            row, unlearn_type=unlearn_type, data_format=data_format
        )
        for row in batch
    ]
    if print_prompts:
        print(f"{prompts_prefix}: {prompts}")

    tokens = tokenizer(
        prompts, return_tensors="pt", max_length=max_seq_len,
        truncation=True, padding=True
    ).to(device)
    logits = model(**model.prepare_inputs_for_generation(**tokens)).logits
    original_loss = -get_log_probs(logits, tokens["input_ids"]).mean()
    if unlearn_type.value == UnlearnType.GD.value:
        loss = -original_loss
    else:
        loss = original_loss

    return loss


def create_prompt_letter_answer(point: Point) -> str:
    return "\n".join(
        [point["question"]]
        + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
        + [f"Answer: {doc_to_choice[i]}. {c}"
            for i, c in enumerate(point["choices"]) if i == point["answer"]   
        ]
    )

def find_last_occur_of_pattern(tokens, patterns_lst, tokenizer):
    flipped_tokens = tokens.flip(-1)
    for i, c in enumerate(flipped_tokens):
        if i == 0:
            continue
        text = tokenizer.decode(c) + tokenizer.decode(flipped_tokens[i - 1])
        found = False
        for k in patterns_lst:
            if k in text:
                return i

def get_loss_letter_answer(
    model,
    batch,
    device,
    tokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    print_prompts: bool = False,
    prompts_prefix: str = "prompts",
):
    prompts = [create_prompt_letter_answer(point) for point in batch]
    if print_prompts:
        print(f"printing from letter_answer")
        print(f"{prompts_prefix}: {prompts}")

    tokens = tokenizer(
        prompts, return_tensors="pt", max_length=max_seq_len,
        truncation=True, padding=True
    ).to(device)
    logits = model(**model.prepare_inputs_for_generation(**tokens)).logits
    neg_log_probs = -get_log_probs(logits, tokens["input_ids"])
    loss = 0
    for i in range(len(batch)):
        patterns_lst = [c+"." for c in doc_to_choice]
        ans_token_ind = find_last_occur_of_pattern(
            tokens.input_ids[i], patterns_lst, tokenizer
        )
        loss += neg_log_probs[i, -ans_token_ind - 1:].mean(dim=-1)

    loss = loss / len(batch)
    if unlearn_type.value == UnlearnType.GD.value:
        loss = -loss

    return loss

def has_number(string):
    import re
    return bool(re.search(r'\d', string))

def find_number(tokens, tokenizer):
    flipped_tokens = tokens.flip(-1)
    for i, c in enumerate(flipped_tokens):
        if i == 0:
            continue
        j = i
        while has_number(tokenizer.decode(flipped_tokens[j])):
            j +=1
        if j != i:
            return (j, i)
    return (None, None)

def get_loss_number(
    model,
    batch,
    device,
    tokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    data_format: DataFormat = DataFormat.NOT_SPECIFIED,
    print_prompts: bool = False,
    prompts_prefix: str = "prompts",
):
    prompts = [
        create_prompt_unlearn(
            row, unlearn_type=unlearn_type, data_format=data_format
        )
        for row in batch
    ]
    if print_prompts:
        print(f"{prompts_prefix}: {prompts}")

    tokens = tokenizer(
        prompts, return_tensors="pt", max_length=max_seq_len,
        truncation=True, padding=True
    ).to(device)
    logits = model(**model.prepare_inputs_for_generation(**tokens)).logits
    neg_log_probs = -get_log_probs(logits, tokens["input_ids"])
    loss = torch.tensor([0], dtype=torch.float).to(device)
    for i in range(len(batch)):
        start, end = find_number(tokens.input_ids[i], tokenizer)
        if start is not None and end is not None:
            loss += neg_log_probs[i, -start:-end].mean(dim=-1)

    loss = loss / len(batch)
    if unlearn_type.value == UnlearnType.GD.value:
        loss = -loss

    return loss.mean()
def get_loss(
    model,
    batch: list[Point],
    device: torch.device,
    tokenizer: AutoTokenizer,
    label_possibilities: list[int],
    train_on_wrong_answer: bool = False,
    max_len: int = 2000,
    max_seq_len: int = MAX_SEQ_LEN,
    unlearn_type: UnlearnType = UnlearnType.NOT_SPECIFIED,
    mcq: bool = False,
    print_prompts: bool = False,
    prompts_prefix: str = "prompts",
    data_format: DataFormat = DataFormat.NOT_SPECIFIED,
    loss_type: LossType = LossType.NOT_SPECIFIED,
):
    if (
        (
            data_format.value == DataFormat.CORPUS.value
            and loss_type.value != LossType.NUMBER.value
        )
        or loss_type.value in [
            LossType.CORPUS.value, LossType.QUESTION_LETTER_ANSWER.value
        ]
    ):
        return get_loss_corpus(
            model,
            batch,
            device,
            tokenizer,
            label_possibilities,
            train_on_wrong_answer=train_on_wrong_answer,
            max_len=max_len,
            max_seq_len=max_seq_len,
            unlearn_type=unlearn_type,
            data_format=data_format,
            print_prompts=print_prompts,
            prompts_prefix=prompts_prefix,
        )
    elif (
        data_format.value == DataFormat.MCQ.value
        and loss_type.value == LossType.LETTER_ANSWER.value
    ):
        return get_loss_letter_answer(
            model,
            batch,
            device,
            tokenizer,
            max_seq_len=max_seq_len,
            unlearn_type=unlearn_type,
            print_prompts=print_prompts,
            prompts_prefix=prompts_prefix,
        )
    elif (
        data_format.value == DataFormat.MCQ.value
        and loss_type.value == LossType.LETTER.value
    ):
        tokens, last_pos_label_ids = process_batch(
            batch,
            device,
            tokenizer,
            label_possibilities,
            max_seq_len=max_seq_len,
        )
        loss, _, _ = get_loss_and_acc(
            model,
            tokens,
            last_pos_label_ids,
            label_possibilities,
            unlearn_type=unlearn_type,
        )
        return loss
    elif (
        data_format.value == DataFormat.CORPUS.value
        and loss_type.value == LossType.NUMBER.value
    ):
        return get_loss_number(
            model,
            batch,
            device,
            tokenizer,
            max_seq_len=max_seq_len,
            unlearn_type=unlearn_type,
            data_format=data_format,
            print_prompts=print_prompts,
            prompts_prefix=prompts_prefix,
        )
        
    else:
        raise Exception(
            f"Unhandled loss. {loss_type=} {data_format=} {unlearn_type=}"
        )
 
def load_jsonl(files):
    dataset = []
    for file in files:
        for line in open(file, "r"):
            dataset += [json.loads(line)]
        
    return dataset

def freeze_model_layers(model, tuples):
    frozen = []
    not_frozen = []
    for name, param in model.named_parameters():
        not_frozen.append(name)
        if 'layers' in name:
            layer_num = int(name.split('.')[2])
            for f, l in tuples:
                if layer_num >= f and layer_num < l:
                    param.requires_grad = False
                    frozen.append(layer_num)
                    not_frozen.remove(name)
        else:
            param.requires_grad = False
            frozen.append(name)
            not_frozen.remove(name)

def _data_path(data_root: str, rel_path: str, ext: str = ".jsonl") -> str:
    if os.path.isabs(rel_path):
        base = rel_path
    else:
        base = os.path.join(data_root, rel_path)
    if ext and not base.endswith(ext):
        base += ext
    return base


def get_model_layers(model):
    """Extract transformer layers from PEFT-wrapped or base model.
    
    Args:
        model: PEFT-wrapped model or base model
    
    Returns:
        layers: ModuleList of transformer layers
        actual_model: The underlying model (not PEFT wrapper)
    
    Raises:
        AttributeError: If layers cannot be found in model structure
    """
    # PEFT wraps the model: model.base_model is LoraModel, need to go deeper
    if hasattr(model, 'base_model'):
        # PEFT-wrapped: get the actual base model
        if hasattr(model.base_model, 'base_model'):
            # Nested: model.base_model.base_model is the actual model
            actual_model = model.base_model.base_model
        else:
            # Single level: model.base_model is the actual model
            actual_model = model.base_model
    else:
        # Not PEFT-wrapped
        actual_model = model
    
    # Handle different model structures: Llama has model.layers, some wrap another model
    if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
        layers = actual_model.model.layers
    elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'model') and hasattr(actual_model.model.model, 'layers'):
        layers = actual_model.model.model.layers
    elif hasattr(actual_model, 'layers'):
        layers = actual_model.layers
    elif hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
        layers = actual_model.transformer.h
    else:
        raise AttributeError(f"Could not find layers attribute in model structure. actual_model type: {type(actual_model)}")
    
    return layers, actual_model


def register_lora_gating_hooks(model, num_layers):
    """Register hooks to gate only LoRA adapter delta.
    
    Hooks read from model._gate_mask[layer_idx] tensor (updated each step).
    Do NOT call .item() - keep tensors in computation graph.
    
    Args:
        model: PEFT-wrapped model
        num_layers: Number of transformer layers
    
    Returns:
        List of hook handles for cleanup
    """
    hook_handles = []
    layers, actual_model = get_model_layers(model)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        
        for module_name in target_modules:
            if hasattr(layer, module_name):
                module = getattr(layer, module_name)
                
                # Check if module has LoRA adapters
                if hasattr(module, 'lora_B'):
                    # Hook on adapter submodules to gate only the adapter contribution
                    def make_adapter_hook(lidx):
                        def adapter_hook(adapter_module, input, output):
                            # Hook on adapter's forward (lora_B)
                            # Gate the adapter output (this is the delta contribution)
                            gate_value = model._gate_mask[lidx]  # Tensor, no .item()
                            return output * gate_value
                        return adapter_hook
                    
                    # CRITICAL: Gate only once (on lora_B output) to avoid m^2 effect
                    # LoRA computation: lora_delta = lora_B(lora_A(input)) * scale
                    # Gating both lora_A and lora_B would give: m * lora_B(m * lora_A(x)) = m^2 * lora_delta
                    # Instead, gate only lora_B output: lora_B(lora_A(x)) * m = m * lora_delta
                    
                    # Gate only lora_B (not lora_A) to avoid double gating
                    # Handle dict, ModuleDict, or single module
                    if isinstance(module.lora_B, (dict, torch.nn.ModuleDict)):
                        # Multiple adapters: hook all
                        for adapter_name in module.lora_B.keys():
                            handle = module.lora_B[adapter_name].register_forward_hook(
                                make_adapter_hook(layer_idx)
                            )
                            hook_handles.append(handle)
                    else:
                        # Single adapter
                        handle = module.lora_B.register_forward_hook(
                            make_adapter_hook(layer_idx)
                        )
                        hook_handles.append(handle)
    
    return hook_handles


def main(
    train_files: list[str],
    wrong_unlearn_files: list[str],
    fixed_wrong_unlearn_files: list[str],
    val_files: list[str],
    dev_set: str,
    base_model: str,
    lr: float,
    name: str,
    data_root: str = "data",
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
    use_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bf16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_double_quant: bool = True,
    max_seq_len: int = MAX_SEQ_LEN,
    grad_accum_steps: int = 1,
    gradient_checkpointing: bool = False,
    attn_backend: Optional[str] = None,
):
    assert (keep_set and keep_set_weight) or (not keep_set and not keep_set_weight)

    assert (
        (unlearn_type.value == UnlearnType.GD.value and train_files)
        or (unlearn_type.value == UnlearnType.WHP.value and wrong_unlearn_files)
        or (unlearn_type.value == UnlearnType.FWF.value and fixed_wrong_unlearn_files)
        or just_eval
    ), f"{unlearn_type=}, {UnlearnType.GD=}, {unlearn_type == UnlearnType.GD}"
          

    if not disable_wandb:
        wandb.init(project=project_name, config={**locals(), "hydra_dict": hydra_dict}, name=name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = int(max_seq_len) if max_seq_len is not None else MAX_SEQ_LEN
    grad_accum_steps = max(1, int(grad_accum_steps))
    warmup_steps = max(1, int(warmup_steps))
    tokenizer = AutoTokenizer.from_pretrained(base_model, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    label_possibilities = [tokenizer.encode(f"{t}. ", add_special_tokens=False)[0] for t in doc_to_choice]
    
    if model is not None:
        model = model
    else:
        attn_impl = get_attn_implementation(attn_backend)
        if use_4bit:
            maybe_set_bnb_cuda_version()
            compute_dtype = resolve_dtype(bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_double_quant,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=compute_dtype,
                quantization_config=bnb_config,
                attn_implementation=attn_impl,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            ).to(device)
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
    if gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    if freeze_layers is not None:
        freeze_model_layers(model, freeze_layers)

    # LoRA setup
    if lora_rank > 0:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Gate parameters for learned top-K selection
    layer_selection_mode = hydra_dict.get("layer_selection_mode", "none")
    lora_layer_budget_k = hydra_dict.get("lora_layer_budget_k", None)
    # Convert OmegaConf ListConfig to Python int if needed
    from omegaconf import OmegaConf
    if lora_layer_budget_k is not None:
        if OmegaConf.is_config(lora_layer_budget_k):
            lora_layer_budget_k = OmegaConf.to_container(lora_layer_budget_k, resolve=True)
        # If it's still a list, take the first element (shouldn't happen, but be safe)
        if isinstance(lora_layer_budget_k, (list, tuple)) and len(lora_layer_budget_k) > 0:
            lora_layer_budget_k = lora_layer_budget_k[0]
        # Ensure it's an int
        if lora_layer_budget_k is not None:
            lora_layer_budget_k = int(lora_layer_budget_k)
    gate_logits = None
    num_layers = model.config.num_hidden_layers if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers') else (model.base_model.config.num_hidden_layers if hasattr(model, 'base_model') and hasattr(model.base_model.config, 'num_hidden_layers') else None)

    if layer_selection_mode == "learned_topk_hard":
        if lora_rank == 0:
            raise ValueError("layer_selection_mode='learned_topk_hard' requires lora_rank > 0")
        if lora_layer_budget_k is None:
            raise ValueError("lora_layer_budget_k must be set when layer_selection_mode='learned_topk_hard'")
        if num_layers is None:
            raise ValueError("Could not determine num_layers from model config")
        if lora_layer_budget_k <= 0:
            raise ValueError(f"lora_layer_budget_k must be > 0, got {lora_layer_budget_k}")
        if lora_layer_budget_k > num_layers:
            raise ValueError(f"lora_layer_budget_k ({lora_layer_budget_k}) > num_layers ({num_layers})")
        
        # Initialize gate logits (small random values)
        # Use local RNG to avoid affecting global random state
        gate_seed = hydra_dict.get("gate_seed", data_seed)
        if gate_seed is None:
            gate_seed = data_seed
        rng = torch.Generator(device=device)
        rng.manual_seed(gate_seed)
        gate_logits = torch.nn.Parameter(
            torch.randn(num_layers, device=device, generator=rng) * 0.01  # Small initialization
        )
        # Register as model parameter so optimizer includes it
        model.register_parameter("gate_logits", gate_logits)
        
        # Initialize gate mask tensor (updated each step, read by hooks)
        # Compute initial Top-K mask at tau_start to satisfy spec (only K blocks active)
        gate_tau_start = hydra_dict.get("gate_tau_start", 10.0)
        s_init = torch.sigmoid(gate_logits / gate_tau_start)
        _, topk_indices_init = torch.topk(s_init, k=lora_layer_budget_k, dim=0)
        m_init = torch.zeros_like(s_init)
        m_init.scatter_(0, topk_indices_init, 1.0)
        m_st_init = m_init + (s_init - s_init.detach())  # STE
        model._gate_mask = m_st_init  # Initialize to Top-K mask, not all ones
        
        # Register hooks once (do not re-register in training loop)
        hook_handles = register_lora_gating_hooks(model, num_layers)
        model._lora_gating_hooks = hook_handles  # Store for cleanup

    # Only optimize trainable parameters (LoRA params if lora_rank > 0).
    optimizer = Lion(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        use_triton=True,
    )

    # Handle dataset loading based on unlearn_type
    # If just_eval=True and train_files=[], we skip training so dataset can be empty
    if just_eval and len(train_files) == 0 and len(wrong_unlearn_files) == 0 and len(fixed_wrong_unlearn_files) == 0:
        train_dataset = []  # Empty dataset for evaluation-only mode
    elif unlearn_type.value == UnlearnType.GD.value:
        train_dataset = load_jsonl(
            [_data_path(data_root, file) for file in train_files]
        )
    elif unlearn_type.value == UnlearnType.WHP.value:
        train_dataset = load_jsonl(
            [_data_path(data_root, file) for file in wrong_unlearn_files]
        )
    elif unlearn_type.value == UnlearnType.FWF.value:
        train_dataset = load_jsonl(
            [_data_path(data_root, file) for file in fixed_wrong_unlearn_files]
        )
    elif unlearn_type.value == UnlearnType.LORA.value:
        # LORA can use any of the file lists, prefer train_files
        if train_files:
            train_dataset = load_jsonl(
                [_data_path(data_root, file) for file in train_files]
            )
        elif wrong_unlearn_files:
            train_dataset = load_jsonl(
                [_data_path(data_root, file) for file in wrong_unlearn_files]
            )
        elif fixed_wrong_unlearn_files:
            train_dataset = load_jsonl(
                [_data_path(data_root, file) for file in fixed_wrong_unlearn_files]
            )
        else:
            train_dataset = []  # Empty for evaluation-only
    else:
        raise Exception("Unlearning type not handled")
    
    if train_dataset:
        random.Random(data_seed).shuffle(train_dataset)

    if max_samples is not None:
        train_dataset = train_dataset[:max_samples]
        print(f"capped samples at {max_samples=}")

    val_datasets_lst = [
        (_data_path(data_root, file), load_jsonl([_data_path(data_root, file)]))
        for file in val_files
    ]
    dev_dataset = load_jsonl([_data_path(data_root, dev_set)])
    retaing_dev_dataset = load_jsonl([_data_path(data_root, retain_dev_file)])
    retain_dataset = load_jsonl(
        [_data_path(data_root, file) for file in retain_files]
    )
    val_retain_datasets_lst = [
        (_data_path(data_root, file), load_jsonl([_data_path(data_root, file)]))
        for file in val_retain_files
    ]
    val_retain_datasets_5_shot_lst = val_retain_datasets_lst.copy()

    if data_format.value == DataFormat.MCQ.value:
        train_dataset = load_jsonl(
            [_data_path(data_root, file) for file in val_files]
        )
        retain_dataset = load_jsonl(
            [_data_path(data_root, file) for file in val_retain_files]
        )

    if k_shot != 0:
        val_datasets_lst = [
            (f, make_k_shot(val_dataset, dev_dataset, k_shot))
            for f, val_dataset in val_datasets_lst
        ]

    if keep_set is not None:
        assert k_shot == 0
        keep_dataset = json.load(open(_data_path(data_root, keep_set, ".json")))
        batch_size //= 2
    
    forget_accs = {}
    forget_accs_calibrated = {}
    forget_logits_dict = {}
    retain_accs = {}
    retain_accs_calibrated = {}
    retain_logits_dict = {}
    retain_accs_5_shot = {}
    retain_accs_5_shot_calibrated = {}
    retain_logits_5_shot_dict = {}

    samples = {}
    eval_5_shot = False

    @torch.no_grad()
    def eval(time: int):
        model.eval()
        # Update gate mask for evaluation using current learned gate_logits
        # This ensures evaluation uses the current learned selection, not stale mask from last training step
        if gate_logits is not None:
            gate_tau_end = hydra_dict.get("gate_tau_end", 0.1)
            s_eval = torch.sigmoid(gate_logits / gate_tau_end)
            _, topk_indices_eval = torch.topk(s_eval, k=lora_layer_budget_k, dim=0)
            m_eval = torch.zeros_like(s_eval)
            m_eval.scatter_(0, topk_indices_eval, 1.0)
            # Use hard mask for evaluation (no STE needed since we're not backpropagating)
            model._gate_mask = m_eval
        
        val_batches_lst = [(f, [val_dataset[i : i + val_batch_size] for i in range(0, len(val_dataset), val_batch_size)]) for f, val_dataset in val_datasets_lst]
        retain_batches_lst = [(f, [val_retain_dataset[i : i + val_batch_size] for i in range(0, len(val_retain_dataset), val_batch_size)]) for f, val_retain_dataset in val_retain_datasets_lst]
        retain_batches_5_shot_lst = [(f, [val_retain_dataset_5_shot[i : i + val_batch_size] for i in range(0, len(val_retain_dataset_5_shot), val_batch_size)]) for f, val_retain_dataset_5_shot in val_retain_datasets_5_shot_lst]
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_labels = []
        label_possibilities_tensor = torch.tensor(label_possibilities, device=device)
        all_files_forget_acc = 0
        all_files_forget_acc_calibrated = 0
        all_files_forget_loss = 0
        all_files_retain_acc = 0
        all_files_retain_acc_calibrated = 0
        all_files_retain_loss = 0
        all_files_retain_5_shot_acc = 0
        all_files_retain_5_shot_acc_calibrated = 0
        all_files_retain_5_shot_loss = 0
        for j, (f, val_batches) in tqdm(enumerate(val_batches_lst), desc=f"Forget-eval"):
            forget_accs[f] = {}
            forget_logits_dict[f] = {}
            forget_accs_calibrated[f] = {}
            total_forget_correct = 0
            total_forget_count = 0
            total_forget_loss = 0
            forget_logits_lst = []
            last_labels_forget_lst = []
            for i, batch in enumerate(val_batches):
                tokens, last_pos_label_ids_forget_local = process_batch(
                    batch,
                    device,
                    tokenizer,
                    label_possibilities,
                    max_seq_len=max_seq_len,
                    print_a_prompt=i==0 and time==0,
                    print_prefix="val prompts=",
                )
                forget_eval_loss, forget_acc, forget_logits_local = get_loss_and_acc(model, tokens, last_pos_label_ids_forget_local, label_possibilities)
                total_forget_correct += forget_acc
                total_forget_count += last_pos_label_ids_forget_local.shape[0]
                total_forget_loss += forget_eval_loss
                last_labels_forget_lst.append(last_pos_label_ids_forget_local)
                forget_logits_lst.append(forget_logits_local)

            # Handle empty batches (shouldn't happen, but be safe)
            if total_forget_count > 0:
                total_forget_acc = total_forget_correct / total_forget_count
                total_forget_loss /= len(val_batches)
            else:
                total_forget_acc = 0.0
                total_forget_loss = 0.0
            all_files_forget_acc += total_forget_acc
            all_files_forget_loss += total_forget_loss
            forget_accs[f][time] = total_forget_acc
            last_labels_forget = torch.cat(last_labels_forget_lst, dim=0)
            forget_logits = np.concatenate(forget_logits_lst, axis=0)
            forget_logits_dict[f][time] = forget_logits
            forget_logits_standardized = forget_logits - forget_logits.mean(axis=0)
            forget_logits_tensor = torch.tensor(forget_logits_standardized, device=device)
            forget_labels = label_possibilities_tensor[forget_logits_tensor.argmax(dim=-1)]
            forget_acc_calibrated = (forget_labels == last_labels_forget).float().mean().item()
            all_files_forget_acc_calibrated += forget_acc_calibrated
            forget_accs_calibrated[f][time] = forget_acc_calibrated

        # Handle case where val_datasets_lst is empty (evaluation-only on retain set)
        if len(val_datasets_lst) > 0:
            all_files_forget_acc /= len(val_datasets_lst)
            all_files_forget_acc_calibrated /= len(val_datasets_lst)
            all_files_forget_loss /= len(val_datasets_lst)
        else:
            # No forget evaluation, set to 0
            all_files_forget_acc = 0.0
            all_files_forget_acc_calibrated = 0.0
            all_files_forget_loss = 0.0
        
        for j, (f, retain_batches) in tqdm(enumerate(retain_batches_lst), desc=f"Retain-eval"):
            retain_logits_dict[f] = {}
            retain_accs_calibrated[f] = {}
            retain_accs[f] = {}
            total_retain_acc = 0
            total_retain_loss = 0
            retain_logits_lst = []
            last_labels_retain_lst = []
            for i in range(len(retain_batches)):
                if i == 0:
                    print(f"Printing retain batches")
                tokens, last_pos_label_ids_retain_local = process_batch(
                    retain_batches[i],
                    device,
                    tokenizer,
                    label_possibilities,
                    max_seq_len=max_seq_len,
                    print_a_prompt=i==0 and time==0,
                    print_prefix="retain prompts",
                )
                retain_eval_loss, retain_acc, retain_logits_local = get_loss_and_acc(model, tokens, last_pos_label_ids_retain_local, label_possibilities)
                total_retain_acc += retain_acc
                total_retain_loss += retain_eval_loss
                last_labels_retain_lst.append(last_pos_label_ids_retain_local)
                retain_logits_lst.append(retain_logits_local)

            total_retain_acc /= len(val_retain_datasets_lst[j][1])
            total_retain_loss /= len(val_retain_datasets_lst[j][1])
            all_files_retain_acc += total_retain_acc
            all_files_retain_loss += total_retain_loss
            retain_logits = np.concatenate(retain_logits_lst, axis=0)
            retain_logits_dict[f][time] = retain_logits
            retain_logits_standardized = retain_logits - retain_logits.mean(axis=0)
            retain_logits_tensor = torch.tensor(retain_logits_standardized, device=device)
            retain_labels = label_possibilities_tensor[retain_logits_tensor.argmax(dim=-1)]
            last_labels_retain = torch.cat(last_labels_retain_lst, dim=0)
            retain_acc_calibrated = (retain_labels == last_labels_retain).float().mean().item()
            all_files_retain_acc_calibrated += retain_acc_calibrated
            retain_accs_calibrated[f][time] = retain_acc_calibrated
            retain_accs[f][time] = total_retain_acc
        
        all_files_retain_acc /= len(val_retain_datasets_lst)
        all_files_retain_acc_calibrated /= len(val_retain_datasets_lst)
        all_files_retain_loss /= len(val_retain_datasets_lst)
            
        if eval_5_shot:
            for j, (f, retain_batches_5_shot) in tqdm(enumerate(retain_batches_5_shot_lst), desc=f"Retain-5-shot-eval"):
                retain_logits_5_shot_dict[f] = {}
                retain_accs_5_shot_calibrated[f] = {}
                retain_accs_5_shot[f] = {}
                total_retain_acc_5_shot = 0
                total_retain_5_shot_loss = 0
                retain_logits_5_shot_lst = []
                last_labels_retain_5_shot_lst = []
                for i in range(len(retain_batches_5_shot)):
                    tokens, last_pos_label_ids_retain_5_shot_local = process_batch(
                        retain_batches_5_shot[i],
                        device,
                        tokenizer,
                        label_possibilities,
                        max_seq_len=max_seq_len,
                        print_a_prompt=False,
                    )
                    retain_5_shot_eval_loss, retain_acc, retain_5_shot_logits_local = get_loss_and_acc(model, tokens, last_pos_label_ids_retain_5_shot_local, label_possibilities)
                    total_retain_acc_5_shot += retain_acc
                    total_retain_5_shot_loss += retain_5_shot_eval_loss
                    last_labels_retain_5_shot_lst.append(last_pos_label_ids_retain_5_shot_local)
                    retain_logits_5_shot_lst.append(retain_5_shot_logits_local)

                total_retain_acc_5_shot /= len(val_retain_datasets_5_shot_lst[j][1])
                total_retain_5_shot_loss /= len(val_retain_datasets_5_shot_lst[j][1])
                all_files_retain_5_shot_acc += total_retain_acc_5_shot
                all_files_retain_5_shot_loss += total_retain_5_shot_loss
                retain_logits_5_shot = np.concatenate(retain_logits_5_shot_lst, axis=0)
                retain_logits_5_shot_dict[f][time] = retain_logits_5_shot
                retain_logits_5_shot_standardized = retain_logits_5_shot - retain_logits_5_shot.mean(axis=0)
                retain_logits_5_shot_tensor = torch.tensor(retain_logits_5_shot_standardized, device=device)
                retain_5_shot_labels = label_possibilities_tensor[retain_logits_5_shot_tensor.argmax(dim=-1)]
                last_labels_retain_5_shot = torch.cat(last_labels_retain_5_shot_lst, dim=0)
                retain_acc_5_shot_calibrated = (retain_5_shot_labels == last_labels_retain_5_shot).float().mean().item()
                all_files_retain_5_shot_acc_calibrated += retain_acc_5_shot_calibrated
                retain_accs_5_shot_calibrated[f][time] = retain_acc_5_shot_calibrated
                retain_accs_5_shot[f][time] = total_retain_acc_5_shot

            all_files_retain_5_shot_acc /= len(val_retain_datasets_5_shot_lst)
            all_files_retain_5_shot_acc_calibrated /= len(val_retain_datasets_5_shot_lst)
            all_files_retain_5_shot_loss /= len(val_retain_datasets_5_shot_lst)

        
        samples[time] = sample_tokens(model, tokenizer, device, max_length=15)


        if not disable_wandb:
            wandb.log(
                {
                    "unlearning/forget_acc": all_files_forget_acc,
                    "unlearning/retain_acc": all_files_retain_acc,
                    "unlearning_other/retain_acc_5_shot": all_files_retain_5_shot_acc if eval_5_shot else None,
                    "unlearning_other/forget_acc_calibrated": all_files_forget_acc_calibrated,
                    "unlearning_other/retain_acc_calibrated": all_files_retain_acc_calibrated,
                    "unlearning_other/retain_acc_5_shot_calibrated": all_files_retain_5_shot_acc_calibrated if eval_5_shot else None,
                    "unlearning_other/eval_forget_loss": all_files_forget_loss,
                    "unlearning_other/eval_retain_loss": all_files_retain_loss,
                    "unlearning_other/eval_retain_5_shot_loss": all_files_retain_5_shot_loss,
                    "unlearning_other/epoch": time, 
                }
            )
        return {
            "unlearning/forget_acc": all_files_forget_acc,
            "unlearning/retain_acc": all_files_retain_acc,
            "unlearning_other/retain_acc_5_shot": all_files_retain_5_shot_acc if eval_5_shot else None,
            "unlearning_other/forget_acc_calibrated": all_files_forget_acc_calibrated,
            "unlearning_other/retain_acc_calibrated": all_files_retain_acc_calibrated,
            "unlearning_other/retain_acc_5_shot_calibrated": all_files_retain_5_shot_acc_calibrated if eval_5_shot else None,
            "unlearning_other/eval_forget_loss": all_files_forget_loss,
            "unlearning_other/eval_retain_loss": all_files_retain_loss,
            "unlearning_other/eval_retain_5_shot_loss": all_files_retain_5_shot_loss,
            "unlearning_other/epoch": time, 
        }

    evaled_0 = False
    eval(0); evaled_0 = True
    optimizer_step = 0

    for epoch in range(epochs):
        if just_eval:
            break

        model.train()
        random.Random(epoch).shuffle(train_dataset)
        batches = [train_dataset[i : i + batch_size] for i in range(0, len(train_dataset), batch_size)]
        retain_batches = [retain_dataset[i : i + batch_size] for i in range(0, len(retain_dataset), batch_size)]

        if keep_set:
            random.Random(epoch).shuffle(keep_dataset)
            keep_batches = [keep_dataset[i : i + batch_size] for i in range(0, len(keep_dataset), batch_size)]

        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(batches, desc=f"Training epoch {epoch}")):
            j = i % len(retain_batches)

            # Compute gate mask (if learned top-K enabled)
            selected_blocks_step = None
            tau = None  # Initialize for reuse in logging
            if gate_logits is not None:
                # Compute temperature (annealed) - ONCE, reuse for logging
                gate_tau_start = hydra_dict.get("gate_tau_start", 10.0)
                gate_tau_end = hydra_dict.get("gate_tau_end", 0.1)
                gate_warmup_steps = hydra_dict.get("gate_warmup_steps", 0)
                
                total_steps = len(batches) * epochs
                current_step = epoch * len(batches) + i
                
                if current_step < gate_warmup_steps:
                    tau = gate_tau_start
                else:
                    progress = (current_step - gate_warmup_steps) / max(1, total_steps - gate_warmup_steps)
                    tau = gate_tau_start - (gate_tau_start - gate_tau_end) * min(1.0, progress)
                
                # Compute soft scores and hard mask
                s = torch.sigmoid(gate_logits / tau)
                _, topk_indices = torch.topk(s, k=lora_layer_budget_k, dim=0)
                m = torch.zeros_like(s)
                m.scatter_(0, topk_indices, 1.0)
                m_st = m + (s - s.detach())  # STE
                
                selected_blocks_step = topk_indices.cpu().tolist()
                
                # Update model field that hooks read from (tensor stays in graph)
                model._gate_mask = m_st  # Hooks will read this tensor each forward pass
                
                # Runtime assertion: hard mask should sum to exactly K
                if current_step % 100 == 0:
                    assert m.sum().item() == lora_layer_budget_k, f"Hard mask sum {m.sum()} != K {lora_layer_budget_k}"

            forget_loss = get_loss(
                model,
                batch,
                device,
                tokenizer,
                label_possibilities,
                unlearn_type=unlearn_type,
                mcq=mcq,
                print_prompts=i==0 and epoch==0,
                prompts_prefix="forget prompts",
                data_format=data_format,
                loss_type=loss_type,
                max_seq_len=max_seq_len,
            )
            retain_loss = get_loss(
                model,
                retain_batches[j],
                device,
                tokenizer,
                label_possibilities,
                unlearn_type=UnlearnType.FWF,
                print_prompts=i==0 and epoch==0,
                prompts_prefix="retain prompts",
                data_format=data_format,
                loss_type=loss_type,
                max_seq_len=max_seq_len,
            )

            try:
                raw_loss = forget_loss + retain_coeff * retain_loss
                # Add optional L2 regularization on gate logits
                if gate_logits is not None:
                    gate_reg_coeff = hydra_dict.get("gate_reg_coeff", 0.0)
                    if gate_reg_coeff > 0:
                        reg_loss = gate_reg_coeff * (gate_logits ** 2).sum()
                        raw_loss = raw_loss + reg_loss
            except Exception as e:
                print(f"""
                    error. {forget_loss=}\n{retain_loss=}
                    {retain_coeff=}\n{hydra_dict=}
                """)
                raise e

            loss = raw_loss / grad_accum_steps
            loss.backward()

            should_step = ((i + 1) % grad_accum_steps == 0) or (i + 1 == len(batches))
            if should_step:
                optimizer_step += 1
                for group in optimizer.param_groups:
                    group["lr"] = lr * max(0, min(1, optimizer_step / warmup_steps))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not disable_wandb:
                current_lr = optimizer.param_groups[0]["lr"]
                log_dict = {
                    "unlearning/train_loss": raw_loss.item(),
                    "unlearning_other/epoch": epoch + i / len(batches),
                    "unlearning_other/lr": current_lr,
                    "unlearning_other/grad_accum_steps": grad_accum_steps,
                    "unlearning_other/optimizer_step": optimizer_step,
                    "unlearning/forget_loss": forget_loss.item(),
                    "unlearning/retain_loss": retain_loss.item()
                }
                
                # Add gate logging if learned top-K enabled
                if gate_logits is not None and selected_blocks_step is not None:
                    # Reuse tau computed above, don't recompute
                    log_dict["gating/selected_blocks_step"] = selected_blocks_step
                    log_dict["gating/gate_scores_mean"] = gate_logits.mean().item()
                    log_dict["gating/gate_scores_std"] = gate_logits.std().item()
                    log_dict["gating/temperature"] = tau
                    log_dict["gating/num_selected"] = len(selected_blocks_step)
                    
                    # Log flip rate (how often top-K membership changes)
                    if hasattr(model, '_prev_selected_blocks_step'):
                        prev_selected = set(model._prev_selected_blocks_step)
                        curr_selected = set(selected_blocks_step)
                        flip_rate = len(prev_selected.symmetric_difference(curr_selected)) / (2 * lora_layer_budget_k)
                        log_dict["gating/flip_rate"] = flip_rate
                    model._prev_selected_blocks_step = selected_blocks_step
                
                wandb.log(log_dict)

        if (not just_eval and (epoch + 1) % eval_every) == 0:
            eval_res = eval(epoch + 1)
            print(f"{eval_res['unlearning/forget_acc']=}, {eval_res['unlearning/retain_acc']=}")
            if (
                save_name is not None
                and eval_res["unlearning/forget_acc"] < 0.5
                and eval_res["unlearning/retain_acc"] > 0.5
            ):
                temp_save_name = f"{save_name}_epoch{epoch + 1}_temp-save_forget{eval_res['unlearning/forget_acc']}_retain{eval_res['unlearning/retain_acc']}"
                print(f"saving with name {temp_save_name=}")
                model.save_pretrained(temp_save_name)
                tokenizer.save_pretrained(temp_save_name)


    if not just_eval or not evaled_0:
        eval(epochs)
    
    # Prepare gate metadata for return (compute before hardening)
    gate_metadata = {}
    if gate_logits is not None:
        gate_tau_end = hydra_dict.get("gate_tau_end", 0.1)
        s_final = torch.sigmoid(gate_logits / gate_tau_end)
        _, final_topk_indices = torch.topk(s_final, k=lora_layer_budget_k, dim=0)
        selected_blocks_final = final_topk_indices.cpu().tolist()  # Final canonical selection
        gate_metadata = {
            "lora_layer_budget_k": lora_layer_budget_k,
            "selected_blocks": selected_blocks_final,
            "final_gate_scores": gate_logits.detach().cpu().tolist(),
            "gate_tau_start": hydra_dict.get("gate_tau_start", 10.0),
            "gate_tau_end": hydra_dict.get("gate_tau_end", 0.1),
            "gate_seed": gate_seed,
        }
    
    if save_name is not None:
        selected_blocks_final = None
        if gate_logits is not None:
            # Remove forward hooks before saving
            if hasattr(model, '_lora_gating_hooks'):
                for handle in model._lora_gating_hooks:
                    handle.remove()
                delattr(model, '_lora_gating_hooks')
            
            # Get final selected blocks (canonical list for manifest) - use from gate_metadata
            selected_blocks_final = gate_metadata.get("selected_blocks", [])
            
            # Zero out LoRA weights for non-selected blocks
            # Handle PEFT's adapter structure (may have dict, ModuleDict, or single module)
            layers, actual_model = get_model_layers(model)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            # Use len(layers) which is more reliable than num_layers variable
            for layer_idx in range(len(layers)):
                if layer_idx not in selected_blocks_final:
                    layer = layers[layer_idx]
                    for module_name in target_modules:
                        if hasattr(layer, module_name):
                            module = getattr(layer, module_name)
                            
                            # Handle PEFT adapter structure (dict, ModuleDict, or single module)
                            if hasattr(module, 'lora_A'):
                                if isinstance(module.lora_A, (dict, torch.nn.ModuleDict)):
                                    # Multiple adapters: zero all
                                    for adapter_name in module.lora_A.keys():
                                        if hasattr(module.lora_A[adapter_name], 'weight'):
                                            module.lora_A[adapter_name].weight.data.zero_()
                                elif hasattr(module.lora_A, 'weight'):
                                    # Single adapter
                                    module.lora_A.weight.data.zero_()
                            
                            if hasattr(module, 'lora_B'):
                                if isinstance(module.lora_B, (dict, torch.nn.ModuleDict)):
                                    # Multiple adapters: zero all
                                    for adapter_name in module.lora_B.keys():
                                        if hasattr(module.lora_B[adapter_name], 'weight'):
                                            module.lora_B[adapter_name].weight.data.zero_()
                                elif hasattr(module.lora_B, 'weight'):
                                    # Single adapter
                                    module.lora_B.weight.data.zero_()
            
            # Remove gate_logits parameter before saving (it's not needed for inference)
            # Use _parameters.pop for reliable removal (delattr may not work reliably)
            if hasattr(model, '_parameters') and 'gate_logits' in model._parameters:
                model._parameters.pop('gate_logits', None)
            if hasattr(model, '_gate_mask'):
                delattr(model, '_gate_mask')
        
        # Merge LoRA weights into base model if applicable.
        # Do this AFTER zeroing weights, BEFORE saving
        if lora_rank > 0:
            model = model.merge_and_unload()  # Merge LoRA into base (non-selected blocks already zeroed)
        model.save_pretrained(save_name)
        tokenizer.save_pretrained(save_name)
        
        # Save metadata file with model specifications
        metadata = {
            "model_path": save_name,
            "base_model": base_model,
            "unlearn_type": unlearn_type.value if hasattr(unlearn_type, 'value') else str(unlearn_type),
            "lora_rank": lora_rank,
            "lr": lr,
            "epochs": epochs,
            "retain_coeff": retain_coeff,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "warmup_steps": warmup_steps,
            "data_seed": data_seed,
            "eval_every": eval_every,
            "data_format": data_format.value if hasattr(data_format, 'value') else str(data_format),
            "loss_type": loss_type.value if hasattr(loss_type, 'value') else str(loss_type),
            "max_samples": max_samples,
            "max_seq_len": max_seq_len,
            "grad_accum_steps": grad_accum_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "use_4bit": use_4bit,
            "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": bnb_4bit_quant_type,
            "bnb_4bit_double_quant": bnb_4bit_double_quant,
            "project_name": project_name,
        }
        if freeze_layers is not None:
            metadata["freeze_layers"] = freeze_layers
        if hydra_dict:
            metadata["hydra_dict"] = hydra_dict
        
        metadata_path = os.path.join(save_name, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    if results_file is not None:
        lock = FileLock(f"{results_file}.lock")
        with lock:
            if os.path.exists(results_file):
                with open(results_file, "r+") as f:
                    results = json.load(f)
                    if save_name not in results:
                        results[save_name] = {}
                    results[save_name]["+".join(val_files)] = forget_accs
                    results[save_name]["+".join(val_retain_files)] = retain_accs
                    f.seek(0)
                    f.truncate()
                    json.dump(results, f, indent=4)
            else:
                with open(results_file, "w+") as f:
                    results = {}
                    if save_name not in results:
                        results[save_name] = {}
                    results[save_name]["+".join(val_files)] = forget_accs
                    results[save_name]["+".join(val_retain_files)] = retain_accs
                    f.seek(0)
                    f.truncate()
                    json.dump(results, f, indent=4)
    if not disable_wandb:
        wandb.finish()
    
    return (
        save_name,
        forget_accs, forget_accs_calibrated, forget_logits_dict,
        retain_accs, retain_accs_calibrated, retain_logits_dict,
        retain_accs_5_shot, retain_accs_5_shot_calibrated,
        retain_logits_5_shot_dict,
        samples,
        gate_metadata  # Gate metadata for manifest entries
    )


@ray.remote(num_gpus=1)
def remote_main(
    train_files: list[str],
    wrong_unlearn_files: list[str],
    fixed_wrong_unlearn_files: list[str],
    val_files: list[str],
    dev_set: str,
    base_model: str,
    lr: float,
    name: str,
    data_root: str = "data",
    k_shot: int = 4,
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
    use_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bf16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_double_quant: bool = True,
    max_seq_len: int = MAX_SEQ_LEN,
    grad_accum_steps: int = 1,
    gradient_checkpointing: bool = False,
    attn_backend: Optional[str] = None,
):
    return main(
        train_files=train_files,
        wrong_unlearn_files=wrong_unlearn_files,
        fixed_wrong_unlearn_files=fixed_wrong_unlearn_files,
        val_files=val_files,
        dev_set=dev_set,
        base_model=base_model,
        lr=lr,
        name=name,
        data_root=data_root,
        k_shot=k_shot,
        epochs=epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        warmup_steps=warmup_steps,
        retain_files=retain_files,
        val_retain_files=val_retain_files,
        retain_dev_file=retain_dev_file,
        max_samples=max_samples,
        data_seed=data_seed,
        eval_every=eval_every,
        keep_set=keep_set,
        keep_set_weight=keep_set_weight,
        train_on_wrong_answer=train_on_wrong_answer,
        train_set_size=train_set_size,
        val_set_size=val_set_size,
        kind=kind,
        save_name=save_name,
        version=version,
        model=model,
        retain_coeff=retain_coeff,
        project_name=project_name,
        unlearn_type=unlearn_type,
        results_file=results_file,
        just_eval=just_eval,
        disable_wandb=disable_wandb,
        hydra_dict=hydra_dict,
        freeze_layers=freeze_layers,
        mcq=mcq,
        data_format=data_format,
        loss_type=loss_type,
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

@ray.remote(num_gpus=1)
@torch.no_grad()
def just_eval(
    train_files: list[str],
    wrong_unlearn_files: list[str],
    fixed_wrong_unlearn_files: list[str],
    val_files: list[str],
    dev_set: str,
    base_model: str,
    lr: float,
    name: str,
    data_root: str = "data",
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
    use_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bf16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_double_quant: bool = True,
    max_seq_len: int = MAX_SEQ_LEN,
    grad_accum_steps: int = 1,
    gradient_checkpointing: bool = False,
    attn_backend: Optional[str] = None,
):
    return main(
        train_files=train_files,
        wrong_unlearn_files=wrong_unlearn_files,
        fixed_wrong_unlearn_files=fixed_wrong_unlearn_files,
        val_files=val_files,
        dev_set=dev_set,
        base_model=base_model,
        lr=lr,
        name=name,
        data_root=data_root,
        k_shot=k_shot,
        epochs=epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        warmup_steps=warmup_steps,
        retain_files=retain_files,
        val_retain_files=val_retain_files,
        retain_dev_file=retain_dev_file,
        max_samples=max_samples,
        data_seed=data_seed,
        eval_every=eval_every,
        keep_set=keep_set,
        keep_set_weight=keep_set_weight,
        train_on_wrong_answer=train_on_wrong_answer,
        train_set_size=train_set_size,
        val_set_size=val_set_size,
        kind=kind,
        save_name=save_name,
        version=version,
        model=model,
        retain_coeff=retain_coeff,
        project_name=project_name,
        unlearn_type=unlearn_type,
        results_file=results_file,
        just_eval=just_eval,
        disable_wandb=disable_wandb,
        hydra_dict=hydra_dict,
        freeze_layers=freeze_layers,
        mcq=mcq,
        data_format=data_format,
        loss_type=loss_type,
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
