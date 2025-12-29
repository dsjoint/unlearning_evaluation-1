# Language Model Unlearning and Fine-tuning Research

**Purpose**: Main entry point for the repository.  
**Audience**: New users, contributors, researchers.  
**Canonical for**: Installation, quick start, repository overview.

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

## Quick Start

1. **Install dependencies** (see Installation above)
2. **Ensure data is available**: The pipeline automatically validates required data artifacts before running. See [docs/DATA.md](docs/DATA.md) for data requirements.
3. **Run the default experiment**:
   ```bash
   python pipeline.py
   ```

For detailed workflows and examples, see [AGENTS.md](AGENTS.md).

## Repository Structure

- `pipeline.py`: Main orchestration script for experiments.
- `unlearn_corpus.py`: Implementation of most unlearning methods.
- `finetune_corpus.py`: Used for fine-tuning and RTT.
- `analyze_year_concept.ipynb`: Notebook for year concept disruption evaluation and visualization.
- `conf/`: Hydra configuration files.
- `data/`: Directory for dataset files.
- `utils/`: Utility modules (attention backend, metrics, etc.).

## Documentation

- **[AGENTS.md](AGENTS.md)**: Complete codebase guide with entry points, pipeline flow, I/O schemas, and method reference.
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**: Configuration guide for model switching, resource settings, and experiment parameters.
- **[docs/DATA.md](docs/DATA.md)**: Data materialization, validation, formats, and schemas.
- **[docs/PAPER.md](docs/PAPER.md)**: Paper protocol documentation (A/B/C conditions, RTT protocol).

## Output Locations

Results are returned as dictionaries from Ray remote functions. The pipeline saves:
- **Models**: `models/` directory - Unlearned models, fine-tuned models, and baseline RTT models
- **Metrics**: Returned as dictionaries from remote functions (can be processed/written to CSV as needed)
- **Error logs**: `pipeline_error.log` - Error logs if exceptions occur
- **WandB logs**: Remote WandB project (configured via `wandb_project_name`)

The pipeline includes a baseline pre-flight check that validates the model knows the information before unlearning. Set `baseline_min_forget_acc=0` to disable this check.

## Year Concept Evaluation

To evaluate general year understanding (ordering, arithmetic, classification) on unlearned models:

1. Generate the evaluation dataset (one-time):
   ```bash
   python scripts/generate_year_concept_eval.py
   ```

2. Generate models using `pipeline.py` (year concept evaluation is NOT run during pipeline)

3. Evaluate and visualize using `analyze_year_concept.ipynb`:
   - Open the notebook and set `RUN_NAME` and `MODEL_ID`
   - Run all cells to evaluate models and generate visualizations
   - Results are written to `evals/pipeline/year_concept/{timestamp}--num{i}.csv`

See [AGENTS.md](AGENTS.md) section "(e.1) Run Year Concept Evaluation" for detailed instructions.

## Prerequisites

- Python 3.10+ (tested with Python 3.12.3)
- CUDA-enabled GPU(s)
- Hugging Face account (for model downloads)
- Weights & Biases account (for logging)
- `.env` file with `OPENAI_API_KEY` (only for data generation scripts)
