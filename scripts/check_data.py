from __future__ import annotations

import os
import sys
from typing import Iterable

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.requirements import get_dataset_requirements, resolve_dataset_path
from data.validate_data import validate_required_artifacts
from unlearn_corpus import load_jsonl


def _get_runtime_cwd() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        return HydraConfig.get().runtime.cwd
    except Exception:
        return os.getcwd()


def _resolve_data_root(cfg: DictConfig) -> str:
    data_root = os.getenv("UNLEARN_DATA_ROOT")
    if not data_root:
        data_root = OmegaConf.select(cfg, "data_root") or "data"
    if not os.path.isabs(data_root):
        data_root = os.path.join(_get_runtime_cwd(), data_root)
    return data_root


def _normalize_dataset_name(dataset: str) -> str:
    if hasattr(dataset, "name"):
        return dataset.name
    return str(dataset)


def _load_sample_jsonl(path: str) -> None:
    if path.endswith(".jsonl"):
        file_path = path
    else:
        file_path = path + ".jsonl"
    if os.path.isfile(file_path):
        rows = load_jsonl([file_path])
        if not rows:
            raise ValueError(f"Empty dataset file: {file_path}")


@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    data_root = _resolve_data_root(cfg)
    datasets = list(OmegaConf.select(cfg, "datasets", default=[]))
    if not datasets:
        raise ValueError("No datasets specified in config.")

    validate_required_artifacts(datasets, data_root)

    for dataset in datasets:
        dataset_name = _normalize_dataset_name(dataset)
        requirements = get_dataset_requirements(dataset_name)
        required = requirements.get("required", [])
        if not required:
            continue
        rel_path = resolve_dataset_path(required[0], data_root)
        abs_path = os.path.join(data_root, rel_path)
        if os.path.isdir(abs_path):
            continue
        _load_sample_jsonl(abs_path)

    print("Data validation passed.")


if __name__ == "__main__":
    main()
