from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import random
import sys
from typing import Iterable

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.requirements import (
    MMLU_CATS_FORGET,
    MMLU_CATS_RETAIN,
    resolve_dataset_path,
)
from data.validate_data import find_missing_artifacts, validate_required_artifacts

DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_SEED = 42


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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _split_list(items: list[str], num_splits: int) -> list[list[str]]:
    if num_splits <= 0:
        raise ValueError("num_splits must be > 0")
    per_split = max(1, len(items) // num_splits)
    splits = []
    start = 0
    for idx in range(num_splits):
        end = start + per_split
        if idx == num_splits - 1:
            end = len(items)
        splits.append(items[start:end])
        start = end
    return splits


def _extract_text(row: dict) -> str:
    for key in ("text", "content", "document", "markdown", "article"):
        if key in row and row[key]:
            return str(row[key]).strip()
    return ""


def _load_dataset_safe(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc
    return load_dataset(*args, **kwargs)


def materialize_fineweb(
    data_root: str,
    manifest: list[dict],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> None:
    def collect_texts(source_dataset):
        texts = []
        for row in source_dataset:
            text = _extract_text(row)
            if text:
                texts.append(text)
            if len(texts) >= sample_size:
                break
        return texts

    source = "HuggingFaceFW/fineweb-edu"
    note = None
    try:
        dataset = _load_dataset_safe(
            "HuggingFaceFW/fineweb-edu", split="train", streaming=True
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        texts = collect_texts(dataset)
    except Exception as exc:
        source = "allenai/c4"
        note = f"fallback_from_fineweb_edu: {exc}"
        dataset = _load_dataset_safe(
            "allenai/c4", "en", split="train", streaming=True
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        texts = collect_texts(dataset)

    if len(texts) < sample_size:
        raise RuntimeError(
            f"FineWeb sample insufficient: {len(texts)} < {sample_size}"
        )
    splits = _split_list(texts, 5)
    artifacts = []
    for idx, split in enumerate(splits):
        rel_path = f"fineweb_edu_seed-42/split_{idx}.jsonl"
        abs_path = os.path.join(data_root, rel_path)
        _write_jsonl(abs_path, [{"text": text} for text in split])
        artifacts.append(rel_path)

    manifest.append(
        {
            "dataset": "fineweb_edu_seed-42",
            "artifacts": artifacts,
            "source": source,
            "seed": seed,
            "split_scheme": {"splits": 5, "total_samples": sample_size},
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "note": note,
        }
    )


def materialize_wikitext(
    data_root: str,
    manifest: list[dict],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> None:
    dataset = _load_dataset_safe(
        "wikitext", "wikitext-103-raw-v1", split="train"
    )
    dataset = dataset.shuffle(seed=seed)
    texts = []
    for row in itertools.islice(dataset, sample_size * 2):
        text = _extract_text(row)
        if text:
            texts.append(text)
        if len(texts) >= sample_size:
            break
    if len(texts) < sample_size:
        raise RuntimeError(
            f"Wikitext sample insufficient: {len(texts)} < {sample_size}"
        )
    rel_path = "wikitext/wikitext_dataset.jsonl"
    abs_path = os.path.join(data_root, rel_path)
    _write_jsonl(abs_path, [{"text": text} for text in texts])
    manifest.append(
        {
            "dataset": "wikitext",
            "artifacts": [rel_path],
            "source": "wikitext-103-raw-v1",
            "seed": seed,
            "split_scheme": {"splits": 1, "total_samples": sample_size},
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        }
    )


def _row_categories(row: dict) -> list[str]:
    for key in ("category", "categories", "category_name", "label"):
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, list):
                return [str(v).lower() for v in value]
            return [str(value).lower()]
    return []


def _row_text(row: dict) -> str:
    if "prompt" in row and "response" in row:
        return f"{row['prompt']}\n{row['response']}".strip()
    for key in ("instruction", "input"):
        if key in row and row[key]:
            output = row.get("output") or row.get("response") or ""
            return f"{row[key]}\n{output}".strip()
    return _extract_text(row)


def materialize_beavertails(
    data_root: str,
    manifest: list[dict],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> None:
    dataset = _load_dataset_safe("PKU-Alignment/BeaverTails", split="train")
    dataset = dataset.shuffle(seed=seed)

    buckets = {
        "criminal_activities_dataset": [],
        "social_issues_dataset": [],
    }
    for row in dataset:
        categories = _row_categories(row)
        text = _row_text(row)
        if not text:
            continue
        if any("criminal" in c for c in categories):
            if len(buckets["criminal_activities_dataset"]) < sample_size:
                buckets["criminal_activities_dataset"].append(text)
        if any("social" in c for c in categories):
            if len(buckets["social_issues_dataset"]) < sample_size:
                buckets["social_issues_dataset"].append(text)
        if all(len(v) >= sample_size for v in buckets.values()):
            break

    artifacts = []
    for name, texts in buckets.items():
        if not texts:
            raise RuntimeError(
                f"BeaverTails category '{name}' missing samples."
            )
        rel_path = f"beavertails/{name}.jsonl"
        abs_path = os.path.join(data_root, rel_path)
        _write_jsonl(abs_path, [{"text": text} for text in texts])
        artifacts.append(rel_path)

    manifest.append(
        {
            "dataset": "beavertails",
            "artifacts": artifacts,
            "source": "PKU-Alignment/BeaverTails",
            "seed": seed,
            "split_scheme": {"splits": 2, "samples_per_split": sample_size},
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        }
    )


def _mmlu_subject_matches(subject: str, category: str) -> bool:
    subject = subject.lower()
    category = category.lower()
    if category in subject:
        return True
    if category == "stem":
        stem_keywords = [
            "math",
            "physics",
            "chemistry",
            "biology",
            "engineering",
            "computer",
            "statistics",
            "astronomy",
        ]
        return any(keyword in subject for keyword in stem_keywords)
    if category == "social sciences":
        keywords = ["psychology", "sociology", "economics", "politics"]
        return any(keyword in subject for keyword in keywords)
    return False


def _normalize_mmlu_answer(answer, choices: list[str]) -> int:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        if answer.isdigit():
            return int(answer)
        answer = answer.strip().lower()
        if answer in ["a", "b", "c", "d"]:
            return ["a", "b", "c", "d"].index(answer)
    return 0 if choices else 0


def materialize_mmlu(
    data_root: str,
    manifest: list[dict],
    sample_per_cat: int = 500,
    seed: int = DEFAULT_SEED,
) -> None:
    try:
        dataset = _load_dataset_safe("cais/mmlu", "all", split="test")
    except Exception:
        dataset = _load_dataset_safe("cais/mmlu", split="test")

    subject_key = "subject" if "subject" in dataset.column_names else None
    categories = list(dict.fromkeys(MMLU_CATS_FORGET + MMLU_CATS_RETAIN))
    artifacts = []

    for idx, category in enumerate(categories):
        if subject_key:
            subset = [
                row
                for row in dataset
                if _mmlu_subject_matches(str(row.get(subject_key, "")), category)
            ]
        else:
            subset = list(dataset)
        if not subset:
            subset = list(dataset)
        rng = random.Random(seed + idx)
        rng.shuffle(subset)
        subset = subset[:sample_per_cat]

        mcq_rows = []
        corpus_rows = []
        for row in subset:
            question = row.get("question")
            choices = row.get("choices")
            answer = _normalize_mmlu_answer(row.get("answer"), choices)
            if question is None or choices is None:
                continue
            mcq_rows.append(
                {
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                }
            )
            correct_answer = choices[answer]
            wrong_answers = [
                c for i, c in enumerate(choices) if i != answer
            ]
            corpus_rows.append(
                {
                    "text": f"{question} Answer: {correct_answer}",
                    "split": f"mmlu_{category}",
                    "original_question": question,
                    "correct_answer": correct_answer,
                    "wrong_answers": wrong_answers,
                }
            )

        mcq_rel = f"mmlu_cats_random_trimmed/mmlu_{category}.jsonl"
        corpus_rel = (
            f"mmlu_cats_random_trimmed/corpus_mmlu_{category}.jsonl"
        )
        _write_jsonl(os.path.join(data_root, mcq_rel), mcq_rows)
        _write_jsonl(os.path.join(data_root, corpus_rel), corpus_rows)
        artifacts.extend([mcq_rel, corpus_rel])

    dev_rel = "mmlu_cats_random_trimmed/dev.jsonl"
    dev_rows = []
    rng = random.Random(seed)
    dataset_list = list(dataset)
    rng.shuffle(dataset_list)
    for row in dataset_list[:200]:
        question = row.get("question")
        choices = row.get("choices")
        answer = _normalize_mmlu_answer(row.get("answer"), choices)
        if question is None or choices is None:
            continue
        dev_rows.append(
            {"question": question, "choices": choices, "answer": answer}
        )
    _write_jsonl(os.path.join(data_root, dev_rel), dev_rows)
    artifacts.append(dev_rel)

    manifest.append(
        {
            "dataset": "mmlu_cats_random_trimmed",
            "artifacts": artifacts,
            "source": "cais/mmlu",
            "seed": seed,
            "split_scheme": {
                "splits": len(categories),
                "samples_per_split": sample_per_cat,
            },
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        }
    )


def _materialize_missing_paths(
    missing_paths: list[str],
    data_root: str,
    manifest: list[dict],
) -> None:
    if any(path.startswith("fineweb_edu_seed-42/") for path in missing_paths):
        materialize_fineweb(data_root, manifest)
    if any(path.startswith("wikitext/") for path in missing_paths):
        materialize_wikitext(data_root, manifest)
    if any(path.startswith("mmlu_cats_random_trimmed/") for path in missing_paths):
        materialize_mmlu(data_root, manifest)
    if any(path.startswith("beavertails/") for path in missing_paths):
        materialize_beavertails(data_root, manifest)

    unresolved = [
        path
        for path in missing_paths
        if path.startswith(
            (
                "wmdp/",
                "wmdp-deduped/",
                "day_of_the_month/",
                "random_bd/",
                "dates-years-trimmed/",
            )
        )
    ]
    if unresolved:
        raise FileNotFoundError(
            "Missing required artifacts not auto-materialized:\n"
            + "\n".join(f"- {path}" for path in unresolved)
        )


def _write_manifest(data_root: str, entries: list[dict]) -> None:
    manifest = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "data_root": data_root,
        "entries": entries,
    }
    _ensure_dir(data_root)
    manifest_path = os.path.join(data_root, "MANIFEST.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)


@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    data_root = _resolve_data_root(cfg)
    datasets = list(OmegaConf.select(cfg, "datasets", default=[]))
    if not datasets:
        raise ValueError("No datasets specified in config.")

    missing = find_missing_artifacts(datasets, data_root)
    if missing:
        manifest_entries: list[dict] = []
        missing_paths = list(
            dict.fromkeys(
                itertools.chain.from_iterable(missing.values())
            )
        )
        _materialize_missing_paths(missing_paths, data_root, manifest_entries)
        _write_manifest(data_root, manifest_entries)

    validate_required_artifacts(datasets, data_root)
    print("Data materialization complete.")


if __name__ == "__main__":
    main()
