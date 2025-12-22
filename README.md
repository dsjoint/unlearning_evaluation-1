
# Language Model Unlearning and Fine-tuning Research

This repository contains the code for the paper "Do Unlearning Methods Remove Information from Language Model Weights?".
![Mutual Information Graph](images/mi.png)

## Installation

### Default Installation (without flash-attn)

```bash
pip install -r requirements.txt
```

This uses PyTorch's built-in scaled dot-product attention (SDPA), which works on all GPUs.

### Optional: Enable Flash Attention (for faster training)

If you have a compatible GPU (Ampere or newer) and CUDA toolkit installed:

```bash
pip install -r requirements.txt
pip install -r requirements-flash.txt
```

**Note:** Flash attention is optional. The codebase automatically falls back to SDPA if flash-attn is not installed.

### Attention Backend Configuration

You can control the attention implementation via the `attn_backend` config option:

```bash
# Auto-detect (default): uses flash if available, otherwise SDPA
python pipeline.py attn_backend=auto

# Force SDPA (PyTorch scaled dot-product attention)
python pipeline.py attn_backend=sdpa

# Force flash attention (will fall back to SDPA if not available)
python pipeline.py attn_backend=flash_attention_2

# Force eager attention (slowest, most compatible)
python pipeline.py attn_backend=eager
```

## Repository Structure

- `pipeline.py`: Main orchestration script for experiments.
- `unlearn_corpus.py`: Implementation of most unlearning methods.
- `finetune_corpus.py`: Used for fine-tuning and RTT.
- `conf/`: Hydra configuration files.
- `data/`: Directory for dataset files.
- `utils/`: Utility modules (attention backend, etc.).

## Key Components
- The main experimental logic is in `pipeline.py`. Start here to understand the overall flow.
- For specific method implementations, refer to `unlearn_corpus.py`.
- RTT details can be found in `finetune_corpus.py`.
- Experiment configurations are managed through Hydra. Check the `conf/` directory for different setups.
   
## Running Experiments

1. Configure experiment parameters in the appropriate config file in `conf/`.
2. Execute experiments using:
   ```
   python pipeline.py
   ```

## Data

- Datasets should be placed in the `data/` directory.
### Dateset Directories
1. Years: `data/dates-years-trimmed`
2. MMLU: `data/mmlu_cats_random_trimmed`
3. WMDP-Deduped: `data/wmdp-deduped`
4. Random Birthdays: `data/random_bd`

### Dateset Files Naming Interpretation
1. The original MCQ questions are called `split_*.jsonl`.
2. The GPT-4o generated text splits have the prefix `corpus_`.
3. The text with incorrect facts (used for RIA) are prefixed with `whp_`.
