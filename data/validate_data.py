from __future__ import annotations

import os
from typing import Iterable

from data.requirements import (
    DATASET_REQUIREMENTS,
    get_dataset_requirements,
    normalize_rel_path,
    path_exists,
    resolve_dataset_path,
)


def _normalize_dataset_name(dataset: str) -> str:
    if hasattr(dataset, "name"):
        return dataset.name
    return str(dataset)


def _unique_list(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def find_missing_artifacts(
    datasets: Iterable[str], data_root: str
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for dataset in datasets:
        dataset_name = _normalize_dataset_name(dataset)
        requirements = get_dataset_requirements(dataset_name)
        required_paths = requirements.get("required", [])
        missing_paths = []
        for rel_path in required_paths:
            resolved = resolve_dataset_path(rel_path, data_root)
            if not path_exists(resolved, data_root):
                missing_paths.append(normalize_rel_path(resolved))
        if missing_paths:
            missing[dataset_name] = _unique_list(missing_paths)
    return missing


def _format_materialize_command(
    datasets: Iterable[str], data_root: str
) -> str:
    datasets_list = ",".join(_unique_list([_normalize_dataset_name(d) for d in datasets]))
    cmd = f"python scripts/materialize_data.py datasets=[{datasets_list}]"
    if data_root and os.path.abspath(data_root) != os.path.abspath("data"):
        cmd += f" data_root={data_root}"
    return cmd


def validate_required_artifacts(
    datasets: Iterable[str], data_root: str
) -> None:
    dataset_names = [_normalize_dataset_name(d) for d in datasets]
    unknown = [d for d in dataset_names if d not in DATASET_REQUIREMENTS]
    if unknown:
        raise KeyError(f"Unknown datasets: {unknown}")

    missing = find_missing_artifacts(dataset_names, data_root)
    if missing:
        lines = ["Missing required dataset artifacts:"]
        for dataset_name, paths in missing.items():
            lines.append(f"- {dataset_name}:")
            for path in paths:
                lines.append(f"  - {path}")
        lines.append(
            "Materialize with: "
            + _format_materialize_command(dataset_names, data_root)
        )
        raise FileNotFoundError("\n".join(lines))
