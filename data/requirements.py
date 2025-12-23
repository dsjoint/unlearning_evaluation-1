from __future__ import annotations

import os
from typing import Iterable

MMLU_CATS_FORGET = ["STEM", "business", "chemistry", "culture", "geography"]
MMLU_CATS_RETAIN = [
    "health",
    "history",
    "law",
    "philosophy",
    "social sciences",
]


def split_paths(prefix: str, indices: Iterable[int]) -> list[str]:
    return [f"{prefix}{i}" for i in indices]


def mmlu_paths(prefix: str, categories: Iterable[str]) -> list[str]:
    return [f"{prefix}{category}" for category in categories]


DATASET_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "YEARS": {
        "required": [
            *split_paths("dates-years-trimmed/corpus_split_", range(5)),
            "dates-years-trimmed/dev",
            *split_paths("dates-years-trimmed/split_", range(5)),
            *split_paths("fineweb_edu_seed-42/split_", range(5)),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "YEARS_MMLU_RETAIN": {
        "required": [
            *split_paths("dates-years-trimmed/corpus_split_", range(5)),
            "dates-years-trimmed/dev",
            *split_paths("dates-years-trimmed/split_", range(5)),
            *mmlu_paths(
                "mmlu_cats_random_trimmed/corpus_mmlu_", MMLU_CATS_FORGET
            ),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "YEARS_TF": {
        "required": [
            *split_paths("dates-years-trimmed/corpus_split_", range(5)),
            *split_paths("dates-years-trimmed/tf_split_", range(5)),
            "dates-years-trimmed/dev",
            *split_paths("dates-years-trimmed/split_", range(5)),
            *split_paths("fineweb_edu_seed-42/split_", range(5)),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "MMLU": {
        "required": [
            *mmlu_paths(
                "mmlu_cats_random_trimmed/corpus_mmlu_", MMLU_CATS_FORGET
            ),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_FORGET),
            *mmlu_paths(
                "mmlu_cats_random_trimmed/corpus_mmlu_", MMLU_CATS_RETAIN
            ),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "WMDP_CORPUS": {
        "required": [
            "wmdp/bio-forget-corpus",
            "wmdp/cyber-forget-corpus",
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            "wikitext/wikitext_dataset",
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "WMDP_CORPUS_FINEWEB": {
        "required": [
            "wmdp/bio-forget-corpus",
            "wmdp/cyber-forget-corpus",
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            *split_paths("fineweb_edu_seed-42/split_", range(5)),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "WMDP_CORPUS_MMLU": {
        "required": [
            "wmdp/bio-forget-corpus",
            "wmdp/cyber-forget-corpus",
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            *mmlu_paths(
                "mmlu_cats_random_trimmed/corpus_mmlu_", MMLU_CATS_RETAIN
            ),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "WMDP_MCQ_CORPUS": {
        "required": [
            *split_paths("wmdp-deduped/corpus_split_", range(5)),
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            "wikitext/wikitext_dataset",
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [
            *split_paths("wmdp-deduped/whp_corpus_split_", range(5)),
            *split_paths("wmdp-deduped/fwf_corpus_split_", range(5)),
        ],
    },
    "WMDP_MCQ_CORPUS_FINEWEB": {
        "required": [
            *split_paths("wmdp-deduped/corpus_split_", range(5)),
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            *split_paths("fineweb_edu_seed-42/split_", range(5)),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [
            *split_paths("wmdp-deduped/whp_corpus_split_", range(5)),
            *split_paths("wmdp-deduped/fwf_corpus_split_", range(5)),
        ],
    },
    "WMDP_MCQ_FINEWEB": {
        "required": [
            *split_paths("wmdp-deduped/mcq_split_", range(5)),
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            *split_paths("fineweb_edu_seed-42/split_", range(5)),
        ],
        "optional": [],
    },
    "WMDP_MCQ_WIKITEXT": {
        "required": [
            *split_paths("wmdp-deduped/mcq_split_", range(5)),
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            "wikitext/wikitext_dataset",
        ],
        "optional": [],
    },
    "WMDP_MCQ_LETTER_ANSWER": {
        "required": [
            *split_paths("wmdp-deduped/mcq_split_", range(5)),
            *split_paths("wmdp-deduped/split_", range(5)),
            "wmdp-deduped/dev",
            *split_paths("fineweb_edu_seed-42/split_", range(5)),
        ],
        "optional": [],
    },
    "BEAVERTAILS": {
        "required": [
            "beavertails/criminal_activities_dataset",
            "beavertails/social_issues_dataset",
        ],
        "optional": [],
    },
    "RANDOM_BD": {
        "required": [
            *split_paths("random_bd/corpus_split_", range(5)),
            *split_paths("random_bd/split_", range(5)),
            *split_paths("fineweb_edu_seed-42/split_", range(5)),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [
            *split_paths("random_bd/whp_corpus_split_", range(5)),
            *split_paths("random_bd/fwf_corpus_split_", range(5)),
        ],
    },
    "RANDOM_BD_SAME_RETAIN": {
        "required": [
            *split_paths("random_bd/corpus_split_", range(5)),
            *split_paths("random_bd/split_", range(5)),
            *split_paths("random_bd/corpus_split_", range(5, 10)),
            *split_paths("random_bd/split_", range(5, 10)),
        ],
        "optional": [],
    },
    "RANDOM_BD_ALL_SPLITS": {
        "required": [
            *split_paths("random_bd/split_", range(10)),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "RANDOM_BD_WITH_MMLU": {
        "required": [
            *split_paths("random_bd/split_", range(10)),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
        ],
        "optional": [],
    },
    "RANDOM_BD_WITH_MMLU_CORPUS": {
        "required": [
            *split_paths("random_bd/corpus_split_", range(5)),
            *split_paths("random_bd/split_", range(5)),
            *mmlu_paths(
                "mmlu_cats_random_trimmed/corpus_mmlu_", MMLU_CATS_RETAIN
            ),
            *mmlu_paths("mmlu_cats_random_trimmed/mmlu_", MMLU_CATS_RETAIN),
            "mmlu_cats_random_trimmed/dev",
        ],
        "optional": [],
    },
    "DAY_OF_THE_MONTH": {
        "required": [
            *split_paths("day_of_the_month/split_", range(5)),
            "day_of_the_month/dev",
        ],
        "optional": [],
    },
}


def normalize_rel_path(rel_path: str) -> str:
    cleaned = rel_path.replace("\\", "/").lstrip("/")
    if cleaned.startswith("data/"):
        cleaned = cleaned[len("data/") :]
    return cleaned


def _file_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def path_exists(rel_path: str, data_root: str) -> bool:
    rel_path = normalize_rel_path(rel_path)
    base_path = os.path.join(data_root, rel_path)
    if _file_nonempty(base_path) or _dir_nonempty(base_path):
        return True
    jsonl_path = base_path + ".jsonl"
    if _file_nonempty(jsonl_path):
        return True
    return False


def resolve_dataset_path(rel_path: str, data_root: str) -> str:
    rel_path = normalize_rel_path(rel_path)

    if rel_path == "wmdp/bio-forget-coprus":
        candidate = "wmdp/bio-forget-corpus"
        if path_exists(candidate, data_root):
            return candidate

    if rel_path.startswith("wrong-dates-years-trimmed/"):
        suffix = rel_path.split("/", 1)[1]
        candidate = f"dates-years-trimmed/whp_{suffix}"
        if path_exists(candidate, data_root):
            return candidate

    if rel_path.startswith("fixed-wrong-dates-years-trimmed/"):
        suffix = rel_path.split("/", 1)[1]
        candidate = f"dates-years-trimmed/fwf_{suffix}"
        if path_exists(candidate, data_root):
            return candidate

    return rel_path


def get_dataset_requirements(dataset: str) -> dict[str, list[str]]:
    if dataset not in DATASET_REQUIREMENTS:
        raise KeyError(f"Unknown dataset requirement: {dataset}")
    return DATASET_REQUIREMENTS[dataset]
