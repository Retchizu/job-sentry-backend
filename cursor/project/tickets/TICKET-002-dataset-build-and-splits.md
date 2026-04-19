# TICKET-002: Build training dataset and splits

## Objective

Prepare train/val/test CSVs from `job_postings_rows.csv` with the new `risk_class` label.

## Scope

- Build `combined_text` from relevant text fields (`job_title`, `job_desc`, `skills_desc`, `company_profile`).
- Ensure null-safe string processing.
- Preserve useful metadata columns for traceability.
- Split data into train/val/test with fixed seed and stratification on `risk_class`.
- Export artifacts:
  - `merged_train.csv`
  - `merged_val.csv`
  - `merged_test.csv`

## Acceptance Criteria

- Split files exist and are readable by the notebook.
- Stratified class distribution is reasonable across splits.
- Data leakage checks are run (no row overlap by `id`).

## Deliverables

- Data-preparation script or notebook cells.
- Saved split CSV files under `artifacts/data/processed/`.
