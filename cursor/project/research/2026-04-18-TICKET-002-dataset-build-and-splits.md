---
date: 2026-04-18T11:13:41+08:00
researcher: riche
git_commit: 26c01727e996da4fcc64221713a2f75fad464f18
branch: main
repository: job-sentry-backend
topic: "TICKET-002: Build training dataset and splits (dataset build, combined_text, risk_class, merged splits)"
tags: [research, codebase, TICKET-002, datasets, splits, risk_class, merged_train, combined_text]
status: complete
last_updated: 2026-04-18
last_updated_by: riche
---

# Research: TICKET-002 — Build training dataset and splits

**Date**: 2026-04-18T11:13:41+08:00  
**Researcher**: riche  
**Git Commit**: `26c01727e996da4fcc64221713a2f75fad464f18`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

What exists in the repository today that corresponds to `cursor/project/tickets/TICKET-002-dataset-build-and-splits.md`: building `combined_text`, producing train/val/test artifacts (including `merged_train.csv` / `merged_val.csv` / `merged_test.csv`), stratification, leakage checks, and the relationship to `job_postings_rows` and `risk_class`?

## Summary

The **ticket text** defines preparing splits from `job_postings_rows.csv` with a **`risk_class`** label, **`combined_text`** from `job_title`, `job_desc`, `skills_desc`, and `company_profile`, null-safe text handling, metadata for traceability, **stratified** train/val/test with a **fixed seed** on **`risk_class`**, exports under `artifacts/data/processed/`, plus leakage checks (no row overlap by `id`).

**What exists in the working tree today spans two related but distinct pipelines:**

1. **Row-level merge (upstream of multiclass labels; aligned with TICKET-007)** — `datasets_row_merge.py` and `scripts/combine_job_postings_rows.py` read `fake_job_postings_rows.csv` and `job_postings_rows.csv`, normalize types, deduplicate, derive **`warning_label`** and **`risk_class`** from `fraudulent` and parsed `warnings` JSON, and write **`artifacts/datasets/combined_job_postings_rows.csv`** plus a summary JSON. This module **does not** perform train/validation/test splitting and **does not** write `merged_*.csv` under `artifacts/data/processed/`.

2. **Notebook-driven D1 + D2 pipeline and Phase 6 “merged” exports** — Jupyter notebooks under `artifacts/ipynb/` build **`combined_text`** from the four unified text fields (with Dataset 1 using renamed columns `title` → `job_title`, etc.). Per-dataset train/val/test splits use **`sklearn.model_selection.train_test_split`** with **`stratify`** set to the binary **`fraudulent`** column and **`random_state=42`**. `artifacts/ipynb/phase6_scam_detection.ipynb` (and the Phase 6 section of `artifacts/ipynb/dataset2_scam_detection.ipynb`) **harmonize** Dataset 1 and Dataset 2 frames to `UNIFIED_COLUMNS`, **concatenate only within the same split** (train with train, etc.), and save **`merged_train.csv`**, **`merged_val.csv`**, **`merged_test.csv`** to the processed data directory. Those merged files use **`fraudulent`** and **`dataset_source` (`D1` / `D2`)** in the unified schema; they **do not** include a **`risk_class`** column in the `UNIFIED_COLUMNS` list.

**Repository metadata:** At research time, `hack/spec_metadata.sh` was **not** present in this workspace. `git_commit` above is the current `HEAD`; several dataset artifacts, notebooks, tickets, and Python modules (`datasets_row_merge.py`, `scripts/`, `tests/test_row_merge.py`) existed **only in the working tree** (untracked or not in this commit), so **GitHub blob permalinks** for those paths are not used here.

## Detailed Findings

### TICKET-002 specification (ticket file)

The ticket is stored at `cursor/project/tickets/TICKET-002-dataset-build-and-splits.md`. It lists acceptance criteria: split files readable by the notebook, reasonable stratified class distribution, and **data leakage checks (no row overlap by `id`)**. Deliverables include a data-prep script or notebook cells and saved CSVs under `artifacts/data/processed/`.

### Ticket ordering relative to other tickets

`cursor/project/tickets/README.md` places **TICKET-007** → **TICKET-001** → **TICKET-002** on the dependency-aware critical path and notes `artifacts/datasets/job_postings_rows.csv` as source material. **TICKET-007** (`cursor/project/tickets/TICKET-007-combine-row-datasets.md`) explicitly states notebook compatibility for downstream splits is tracked under **TICKET-002**.

### Row-level CSV merge and `risk_class` derivation (`datasets_row_merge.py`, `scripts/combine_job_postings_rows.py`)

- **`datasets_row_merge.py`** documents merge of `fake_job_postings_rows` + `job_postings_rows` for TICKET-007. It defines **`REQUIRED_COLUMNS`** including `job_title`, `job_desc`, `skills_desc`, `company_profile`, `fraudulent`, `warnings`, and `id` (see `REQUIRED_COLUMNS` at lines 17–30).
- **Null-safe text handling**: `normalize_dtypes` fills **`TEXT_COLUMNS`** with `""` and casts to `str` (lines 105–109).
- **`derive_labels`** (lines 145–159) sets **`warning_label`** from non-empty `warnings` JSON `flags` lists via **`parse_warnings_flags`**, then sets **`risk_class`** as `2` if `fraudulent == 1`, else `1` if `warning_label == 1`, else `0`.
- **`merge_dataframes`** (lines 162–200) concatenates sources with **`dataset_source`** `fake_rows` / `job_rows`, applies **`normalize_dtypes`**, **`dedupe_exact`**, **`dedupe_near`** (normalized key from the four text columns), then **`derive_labels`**, and returns a summary dict including **`risk_class_counts`**.
- **`scripts/combine_job_postings_rows.py`** is a CLI that calls **`merge_sources`**, reorders columns to **`REQUIRED_COLUMNS` + `dataset_source`, `warning_label`, `risk_class`**, writes **`combined_job_postings_rows.csv`** and embeds output paths in the summary JSON (lines 53–70).

There is **no** call to `train_test_split` or equivalent in these `.py` files (confirmed by repository search for `risk_class` / `train_test_split` under `*.py`).

### Tests for row merge (`tests/test_row_merge.py`)

Tests cover **`parse_warnings_flags`**, **`risk_class` precedence** via **`derive_labels`**, **normalized text key** behavior, **merge** of tiny fixtures (including near-dedupe), **`merge_sources`**, and column mismatch errors. They assert **`risk_class`** presence on exported CSVs from the merge pipeline, not split-file overlap.

### Notebook: stratified splits on `fraudulent` (Dataset 1)

`artifacts/ipynb/main_scam_detection.ipynb` sets **`target_column = 'fraudulent'`**, builds **`y`** from that column, and calls **`train_test_split`** twice (approximately 70% / 30%, then 50% / 50% of the remainder) with **`stratify=y`** / **`stratify=y_temp`** and **`random_state=42`** (see notebook source around the lines that set `target_column` and `train_test_split`).

### Notebook: `combined_text` and D1 field mapping (Phase 6)

`artifacts/ipynb/phase6_scam_detection.ipynb` defines **`UNIFIED_COLUMNS`** to include **`job_title`**, **`job_desc`**, **`skills_desc`**, **`company_profile`**, **`salary_range`**, **`employment_type`**, **`fraudulent`**, **`combined_text`**. **`harmonize_d1`** renames **`title` → `job_title`**, **`description` → `job_desc`**, **`requirements` → `skills_desc`**, and if **`combined_text`** is missing, builds it by concatenating the four fields with **`fillna("")`** (see notebook JSON source in the cell defining `harmonize_d1` / `harmonize_d2`). **`harmonize_d2`** uses the same four-field rule when **`combined_text`** is absent.

### Notebook: Phase 6 merge into `merged_*.csv`

In the same notebook, **`train_merged = pd.concat([d1_train_h, d2_train_h], ...)`** (and analogous **`val_merged`**, **`test_merged`**) documents “no cross-split leakage” in the section markdown. The following code fills text columns, casts **`fraudulent`** to int, and writes **`merged_train.csv`**, **`merged_val.csv`**, **`merged_test.csv`** to **`_merged_dir`** (the processed data directory used by that notebook).

### On-disk `merged_train.csv` sample (processed artifacts)

A header read of `artifacts/data/processed/merged_train.csv` shows columns including **`job_title`**, **`job_desc`**, **`skills_desc`**, **`company_profile`**, **`salary_range`**, **`employment_type`**, **`fraudulent`**, **`combined_text`**, **`dataset_source`** — consistent with **`UNIFIED_COLUMNS` + `dataset_source`**, and **without** a **`risk_class`** column in the first row of that file.

### Consumption of `merged_*.csv` elsewhere

`artifacts/ipynb/phase6_deep_learning.ipynb` and `artifacts/ipynb/phase6_hybrid_fused.ipynb` load **`merged_train.csv`**, **`merged_val.csv`**, **`merged_test.csv`** from the processed data directory (see notebook cells that call **`pd.read_csv`** with those filenames).

### `cursor/project/notes/` keyword search

No note files under `cursor/project/notes/` matched keywords **`merged_train`**, **`risk_class`**, or **`TICKET-002`** in a locator pass; **`cursor/project/notes/dataset2-linguistic-feature-set.md`** documents linguistic features from the Dataset 2 notebook and references **`combined_text`**, not row-level splits.

## Code References

- `cursor/project/tickets/TICKET-002-dataset-build-and-splits.md` — Ticket scope, acceptance criteria, deliverables.
- `cursor/project/tickets/README.md` — Ordering: TICKET-007 → TICKET-001 → TICKET-002 on the critical path.
- `cursor/project/tickets/TICKET-007-combine-row-datasets.md` — Points downstream split work to TICKET-002; references `scripts/combine_job_postings_rows.py` in “Implementation (repository)”.
- `datasets_row_merge.py` — `REQUIRED_COLUMNS`, `TEXT_COLUMNS`, `normalize_dtypes`, `derive_labels` / `risk_class`, `merge_dataframes`, `merge_sources`.
- `scripts/combine_job_postings_rows.py` — CLI defaults and CSV/JSON outputs for the combined row dataset.
- `tests/test_row_merge.py` — Tests for merge and `risk_class` derivation.
- `artifacts/ipynb/main_scam_detection.ipynb` — `target_column = 'fraudulent'`, stratified `train_test_split` with `random_state=42`.
- `artifacts/ipynb/phase6_scam_detection.ipynb` — `UNIFIED_COLUMNS`, `harmonize_d1` / `harmonize_d2`, `pd.concat` per split, `merged_train.csv` / `merged_val.csv` / `merged_test.csv` writes.
- `artifacts/data/processed/merged_train.csv` — Example on-disk merged split (header includes `fraudulent`, `combined_text`, `dataset_source`).

## Architecture Documentation

- **Two label granularities appear in different artifacts**: binary **`fraudulent`** for the classic D1/D2 notebook pipeline and merged **`merged_*.csv`** files; **3-way `risk_class`** on the **row-combined** CSV produced by **`datasets_row_merge.py`** (TICKET-001 precedence embedded in `derive_labels`).
- **Stratification**: notebook per-dataset splitting stratifies on **`fraudulent`**. The row-merge Python code does not stratify or split.
- **Leakage control (Phase 6)**: harmonized frames are concatenated **within** train, val, and test separately; markdown in the notebook describes avoiding cross-split leakage. **D1** uses **`job_id`** in other notebook sections for alignment where applicable; **`merged_train.csv`** as sampled does not list an `id` column from the row-level schema (the Phase 6 unified schema differs from the `*_rows.csv` schema).

## Historical Context (from cursor/project/notes/)

- No historical notes were found that duplicate TICKET-002’s split specification; **`cursor/project/notes/dataset2-linguistic-feature-set.md`** records linguistic feature formulas tied to **`combined_text`** in the Dataset 2 notebook.

## Related Research

- `cursor/project/research/2026-04-18-TICKET-007-combine-row-datasets.md` — Documents TICKET-007 and relationships to Phase 6 merged files (note: that document’s “no implemented script” clause predates **`datasets_row_merge.py`** in the current working tree).

## Open Questions

- Whether a **train/val/test split stratified on `risk_class`** will be implemented **directly from `combined_job_postings_rows.csv` or `job_postings_rows.csv`** in Python (outside notebooks) is **not** represented in the searched `*.py` files beyond merge and label derivation.
- **`hack/spec_metadata.sh`** was not found in the repository at research time; metadata in this file was gathered manually.

## GitHub permalinks

Not applied: the dataset modules, notebooks, artifacts, and ticket paths cited here were **not verified as present in commit `26c01727e996da4fcc64221713a2f75fad464f18`** on the remote; use local paths above for navigation until files are committed and pushed.
