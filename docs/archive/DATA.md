DATA.md - Data Materialization and Validation
=============================================

Overview
--------
This repo expects datasets under a data root (default: `data/`). Some datasets
are included, while others must be materialized or supplied externally. Use the
scripts below to build minimal artifacts or to validate that required files are
present.

Data Root
---------
By default, data lives in `data/`. Override this via:
- Environment: `UNLEARN_DATA_ROOT=/path/to/data`
- Hydra override: `data_root=/path/to/data`

Materialize Data
----------------
Create minimal artifacts for selected datasets:
```
python scripts/materialize_data.py datasets=[YEARS]
```
To use a custom data root:
```
UNLEARN_DATA_ROOT=/mnt/data python scripts/materialize_data.py datasets=[YEARS]
```

Validation
----------
Validate required artifacts (fails fast with actionable errors):
```
python scripts/check_data.py datasets=[YEARS]
```

Pipeline
--------
After materialization, you can run:
```
python pipeline.py datasets=[YEARS]
```

What Gets Materialized
----------------------
- FineWeb-Edu sample: `fineweb_edu_seed-42/split_0..4` (JSONL with `{"text": ...}`).
  If FineWeb-Edu fails to load, the materializer falls back to `allenai/c4`
  and records the fallback in `data/MANIFEST.json`.
- WikiText sample: `wikitext/wikitext_dataset.jsonl`
- MMLU categories: `mmlu_cats_random_trimmed/*` (MCQ + corpus + dev)
- BeaverTails categories: `beavertails/criminal_activities_dataset.jsonl`,
  `beavertails/social_issues_dataset.jsonl`

External / User-Supplied Data
-----------------------------
Some artifacts are not auto-materialized and must be provided manually:
- WMDP (`wmdp-deduped/*`, `wmdp/*`)
- Day of the Month (`day_of_the_month/*`)
- Dates-years and RandomBD datasets if missing

Notes
-----
- Generated datasets are small, deterministic samples intended to make the
  pipeline runnable, not to replicate paper-scale training.
