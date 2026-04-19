---
date: 2026-04-18T09:28:49+08:00
researcher: riche
git_commit: 26c01727e996da4fcc64221713a2f75fad464f18
branch: main
repository: job-sentry-backend
topic: "TICKET-007: Combine fake_job_postings_rows.csv and job_postings_rows.csv"
tags: [research, codebase, TICKET-007, datasets, artifacts, phase6, merged_train]
status: complete
last_updated: 2026-04-18
last_updated_by: riche
---

# Research: TICKET-007 — Combine `fake_job_postings_rows.csv` and `job_postings_rows.csv`

**Date**: 2026-04-18T09:28:49+08:00  
**Researcher**: riche  
**Git Commit**: `26c01727e996da4fcc64221713a2f75fad464f18`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

What exists in the repository today regarding [TICKET-007](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/TICKET-007-combine-row-datasets.md) (combining `artifacts/datasets/fake_job_postings_rows.csv` and `artifacts/datasets/job_postings_rows.csv` into `artifacts/datasets/combined_job_postings_rows.csv`), and how does that relate to other dataset merge or training prep code?

## Summary

The **ticket text** for TICKET-007 is the only formal specification in-repo: it defines required columns, deduplication expectations, a `dataset_source` provenance column (`fake_rows` / `job_rows`), output path `artifacts/datasets/combined_job_postings_rows.csv`, and reporting of `fraudulent`, warning counts, and multiclass `risk_class` distribution.

**There is no implemented script, notebook cell, or Python module** that references `fake_job_postings_rows.csv`, `job_postings_rows.csv`, or `combined_job_postings_rows.csv` by name. A workspace check at research time showed `artifacts/datasets/job_postings_rows.csv` and `artifacts/datasets/fake_job_postings_rows.csv` present on disk, while **`artifacts/datasets/combined_job_postings_rows.csv` was not present**.

**Separate from TICKET-007**, the notebooks `artifacts/ipynb/phase6_scam_detection.ipynb` (and the Phase 6 section of `artifacts/ipynb/dataset2_scam_detection.ipynb`) implement a **different merge**: they harmonize **Dataset 1** splits produced from **`fake_job_postings.csv`** (`fake_job_postings_{train,val,test}.csv`) with **Dataset 2** splits (`dataset2_{train,val,test}.csv`), concatenate per split, tag `dataset_source` as **`D1` / `D2`**, and write `merged_{train,val,test}.csv` under the processed data directory. Downstream notebooks such as `phase6_deep_learning.ipynb` and `phase6_hybrid_fused.ipynb` consume those merged files.

**Related ticket text** (`TICKET-001`, `TICKET-002`) describes future work on `job_postings_rows.csv` (warnings JSON → labels, `risk_class` splits) but does not add executable code in the tracked tree for those paths.

**Historical notes**: No documents under `cursor/project/notes/` were found that mention TICKET-007 or the `*_rows.csv` merge by keyword search; cross-references exist under `cursor/project/tickets/` only.

## Detailed Findings

### TICKET-007 specification (authoritative scope for the ticket)

The ticket [TICKET-007-combine-row-datasets.md](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/TICKET-007-combine-row-datasets.md) states:

- **Inputs**: `artifacts/datasets/fake_job_postings_rows.csv` and `artifacts/datasets/job_postings_rows.csv`.
- **Validation**: Same required columns (`id`, `job_title`, `job_desc`, `skills_desc`, `company_profile`, `rate_min`, `rate_max`, `currency`, `rate_type`, `created_at`, `fraudulent`, `warnings`).
- **Processing**: Concatenate, normalize types, remove exact and near-duplicates (normalized text payload), add `dataset_source` (`fake_rows` / `job_rows`).
- **Output**: `artifacts/datasets/combined_job_postings_rows.csv`.
- **Acceptance**: Readable CSV; duplicate rules and counts documented; row counts and class distributions (`fraudulent`, warning-present, derived `risk_class`); schema compatible with training preprocessing.

### Tracked code and notebooks: `*_rows.csv` merge

- **Grep across `*.py`, `*.ipynb`, and `*.md`**: References to `fake_job_postings_rows`, `job_postings_rows`, or `combined_job_postings` appear **only** under `cursor/project/tickets/` (not in application code or notebooks).
- **`app/` Python tree**: No matches for `artifacts/datasets`, `fake_job_postings.csv`, or `job_postings_rows`.

### Artifacts on disk (workspace state at research time)

- `artifacts/datasets/job_postings_rows.csv` — present.
- `artifacts/datasets/fake_job_postings_rows.csv` — present.
- `artifacts/datasets/combined_job_postings_rows.csv` — **not** present.

(Untracked `artifacts/` content is listed in git status; paths are relative to the repo root.)

### Phase 6 merge (existing, different inputs and outputs)

`artifacts/ipynb/phase6_scam_detection.ipynb` documents and implements merging **after** independent preprocessing of:

- Dataset 1: splits from the pipeline that uses **`fake_job_postings.csv`** (filenames `fake_job_postings_train.csv`, etc.).
- Dataset 2: `dataset2_{train,val,test}.csv` from the Dataset 2 notebook path.

The notebook defines `UNIFIED_COLUMNS`, `harmonize_d1` / `harmonize_d2` (adding `dataset_source` of **`"D1"`** and **`"D2"`**), `pd.concat` per split, fill/cast steps, and writes **`merged_train.csv`**, **`merged_val.csv`**, **`merged_test.csv`** to `PROCESSED_DATA_DIR` (same logical content is embedded in `dataset2_scam_detection.ipynb` Phase 6).

`artifacts/ipynb/phase6_deep_learning.ipynb` loads `merged_train.csv` / `merged_val.csv` / `merged_test.csv` from `PROCESSED_DATA_DIR` and uses columns including `dataset_source`; it computes `dataset_source_encoded` as `(df["dataset_source"] == "D2").astype(int)`.

This pipeline is **not** the TICKET-007 `*_rows.csv` → `combined_job_postings_rows.csv` step; it uses different source files and different `dataset_source` labels.

### Ticket cross-links (multiclass / `job_postings_rows.csv`)

- [TICKET-001-label-schema-and-mapping.md](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/TICKET-001-label-schema-and-mapping.md): Describes parsing `warnings` from **`job_postings_rows.csv`** and deriving `risk_class` (`legit` / `warning` / `fraud`) with precedence rules; deliverables include notebook/cell implementation (not located in tracked code for this ticket’s scope).
- [TICKET-002-dataset-build-and-splits.md](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/TICKET-002-dataset-build-and-splits.md): References building splits from **`job_postings_rows.csv`** with `risk_class`.

### Ticket index

[cursor/project/tickets/README.md](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/README.md) lists TICKET-007 first in priority and shows the dependency chain `TICKET-007` → `TICKET-001` → … and notes using `artifacts/datasets/job_postings_rows.csv` as source in the roadmap narrative.

## Code References

- [cursor/project/tickets/TICKET-007-combine-row-datasets.md](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/TICKET-007-combine-row-datasets.md) — Full merge objective, scope, acceptance criteria, deliverables for the row-level combined CSV.
- [cursor/project/tickets/TICKET-001-label-schema-and-mapping.md](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/TICKET-001-label-schema-and-mapping.md) — `warnings` / `risk_class` contract tied to `job_postings_rows.csv`.
- [cursor/project/tickets/TICKET-002-dataset-build-and-splits.md](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/cursor/project/tickets/TICKET-002-dataset-build-and-splits.md) — Splits from `job_postings_rows.csv` with `risk_class`.
- `artifacts/ipynb/phase6_scam_detection.ipynb` — Phase 6 harmonization, `dataset_source` `D1`/`D2`, `merged_{train,val,test}.csv` (JSON source lines include `UNIFIED_COLUMNS`, `harmonize_d1`/`harmonize_d2`, `merged_train.csv` writes).
- `artifacts/ipynb/dataset2_scam_detection.ipynb` — Same Phase 6 merge block embedded (e.g. `UNIFIED_COLUMNS` / harmonize / `merged_train.csv`).
- `artifacts/ipynb/phase6_deep_learning.ipynb` — Reads `merged_*.csv`, lists columns including `dataset_source`, `dataset_source_encoded` feature.
- `artifacts/ipynb/phase6_hybrid_fused.ipynb` — References `merged_train.csv` etc. for fused training.

## Architecture Documentation

- **TICKET-007 (planned)**: Single consolidated **`combined_job_postings_rows.csv`** under `artifacts/datasets/`, row-level schema with rates/currency/`warnings`, provenance `fake_rows` | `job_rows`, deduplication, and multiclass reporting.
- **Implemented Phase 6 merge**: **Stratified split CSVs** from the Kaggle-style D1 pipeline and Dataset 2 pipeline → unified columns → **`merged_{train,val,test}.csv`** with binary `fraudulent` and `dataset_source` **`D1`/`D2`**; consumed by Phase 6 DL / hybrid fused notebooks. This is the documented path for combined D1+D2 training in those artifacts.

## Historical Context (from cursor/project/notes/)

No matching documents were found in `cursor/project/notes/` for TICKET-007 or the `*_rows.csv` filenames (per targeted search). Supplementary context for the broader Phase 6 / merged-data story appears in other research notes (see Related Research).

## Related Research

- [cursor/project/research/2026-03-22-salary-hours-structured-fields-phase6-fused.md](2026-03-22-salary-hours-structured-fields-phase6-fused.md) — `merged_train.csv` column usage and phase6 fused text-only behavior.
- [cursor/project/research/2026-03-23-phase6-fused-only-artifact-usage.md](2026-03-23-phase6-fused-only-artifact-usage.md) — Phase6 fused artifact layout and inference.
- [cursor/project/research/2025-03-22-phase6-fused-vs-codebase-gaps.md](2025-03-22-phase6-fused-vs-codebase-gaps.md) — Broader phase6 vs codebase alignment (historical).

## Open Questions

- None required for documenting **current** state: the ticket deliverables for the `*_rows.csv` merge are **not** yet reflected as tracked scripts or the specified output filename; any future implementation would establish the duplicate counts and class distributions described in the ticket.

## Metadata note

The command workflow referenced `hack/spec_metadata.sh`; that path is **not present** in this repository at commit `26c01727e996da4fcc64221713a2f75fad464f18`. Metadata above was taken from `git` and the system clock. `HEAD` was an ancestor of `origin/main` at research time (suitable for GitHub permalinks to this commit).
