# TICKET-007: Combine `fake_job_postings_rows.csv` and `job_postings_rows.csv`

## Objective

Create a single consolidated dataset by combining:

- `artifacts/datasets/fake_job_postings_rows.csv`
- `artifacts/datasets/job_postings_rows.csv`

The merged output should preserve schema compatibility for downstream multiclass training (`legit`, `warning`, `fraud`).

## Scope

- Validate both files share the same required columns:
  - `id, job_title, job_desc, skills_desc, company_profile, rate_min, rate_max, currency, rate_type, created_at, fraudulent, warnings`
- Concatenate rows from both sources into one dataframe.
- Normalize data types for key fields (`fraudulent`, numeric rates, text columns).
- Remove exact duplicates and near-duplicates (same normalized text payload).
- Add/retain source provenance column (for example: `dataset_source` = `fake_rows` or `job_rows`).
- Save merged artifact to `artifacts/datasets/combined_job_postings_rows.csv`.

## Acceptance Criteria

- Combined CSV is created and readable.
- Duplicate handling is documented (rule + count removed).
- Final row count and class distribution are reported:
  - `fraudulent` counts
  - warning-present counts
  - derived multiclass counts (`risk_class`)
- Output schema is confirmed compatible with training notebook preprocessing.

## Deliverables

- One reproducible merge step (script or notebook cell).
- Final merged CSV plus a short merge summary note (row counts before/after dedupe).

## Implementation (repository)

From the repo root: `python3 scripts/combine_job_postings_rows.py` (defaults under `artifacts/datasets/`). Logic lives in `datasets_row_merge.py`; machine-readable counts and duplicate rules are written to `artifacts/datasets/combined_job_postings_rows.summary.json` alongside `combined_job_postings_rows.csv`. Notebook compatibility for downstream splits is tracked under TICKET-002.
