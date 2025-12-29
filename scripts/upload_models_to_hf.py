#!/usr/bin/env python3
"""
Upload model checkpoints to HuggingFace Hub.

Usage:
    python scripts/upload_models_to_hf.py --run-name 2025-12-28_05-13-18 --hf-username dsjoint
    python scripts/upload_models_to_hf.py --run-name 2025-12-28_05-13-18 --hf-username dsjoint --private
    
    Or use the helper script:
    bash scripts/upload_models.sh
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder
from tqdm import tqdm

def upload_run_to_hf(run_name: str, hf_username: str, private: bool = False, models_dir: str = "models", token: str = None):
    """Upload a specific run's models to HuggingFace Hub.
    
    Args:
        run_name: The run name folder (e.g., "2025-12-28_05-13-18")
        hf_username: Your HuggingFace username
        private: Whether to make the repository private
        models_dir: Root directory containing model checkpoints
        token: HuggingFace token (if None, uses HF_TOKEN env var or cached token)
    """
    # Use token from arg, env var, or cached login
    if token:
        api = HfApi(token=token)
    else:
        api = HfApi()  # Will use HF_TOKEN env var or cached token
    
    run_path = Path(models_dir) / run_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Create repository name
    repo_id = f"{hf_username}/unlearning-eval-{run_name}"
    
    print(f"Creating repository: {repo_id}")
    try:
        api.create_repo(repo_id, exist_ok=True, repo_type="model", private=private)
    except Exception as e:
        print(f"Note: Repository may already exist: {e}")
    
    print(f"Uploading models from {run_path}...")
    print(f"This may take a while for large models...")
    
    upload_folder(
        folder_path=str(run_path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.lock", "*.tmp", "__pycache__", "*.pyc"],
    )
    
    print(f"\n✅ Upload complete!")
    print(f"Models available at: https://huggingface.co/{repo_id}")
    print(f"\nTo download later:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")


def list_runs(models_dir: str = "models"):
    """List all available runs."""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    runs = [d.name for d in models_path.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    
    if not runs:
        print("No runs found (no manifest.json files)")
        return
    
    print("Available runs:")
    for run in sorted(runs):
        run_path = models_path / run
        size = sum(f.stat().st_size for f in run_path.rglob('*') if f.is_file()) / (1024**3)
        print(f"  {run} ({size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Upload model checkpoints to HuggingFace Hub")
    parser.add_argument("--run-name", type=str, help="Run name to upload (e.g., 2025-12-28_05-13-18)")
    parser.add_argument("--hf-username", type=str, help="Your HuggingFace username")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--list", action="store_true", help="List available runs")
    parser.add_argument("--token", type=str, help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    if args.list:
        list_runs(args.models_dir)
        return
    
    if not args.run_name or not args.hf_username:
        parser.error("--run-name and --hf-username are required (or use --list)")
    
    # Check for token (priority: --token flag > HF_TOKEN env var > .hf_token file > cached login)
    token = args.token
    if not token:
        token = os.getenv("HF_TOKEN")
    if not token:
        # Try to load from .hf_token file (in project root)
        token_file = Path(__file__).parent.parent / ".hf_token"
        if token_file.exists():
            with open(token_file, 'r') as f:
                token = f.read().strip().split('=')[-1]  # Handle HF_TOKEN=... format
    if not token:
        print("⚠️  Warning: No token provided. Using cached login if available.")
        print("   To provide a token: set HF_TOKEN env var, use --token flag, or create .hf_token file")
        print("   To login: hf auth login")
    
    upload_run_to_hf(args.run_name, args.hf_username, args.private, args.models_dir, token)


if __name__ == "__main__":
    main()

