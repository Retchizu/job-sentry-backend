# TICKET-001: Define label schema and mapping

## Objective

Create a deterministic target label strategy that supports both warning and fraud detection.

## Scope

- Define final class schema for training:
  - `0 = legit`
  - `1 = warning`
  - `2 = fraud`
- Parse `warnings` JSON column from `job_postings_rows.csv`.
- Convert `warnings` into `warning_label` (binary).
- Create precedence rules for final `risk_class` target:
  - If `fraudulent == 1`, class is `fraud` regardless of warning flags.
  - Else if warning flags exist, class is `warning`.
  - Else class is `legit`.
- Document edge cases (invalid JSON, empty flags, missing values).

## Acceptance Criteria

- Mapping logic is written and reproducible.
- Class counts are printed and reviewed.
- A short label contract is documented for downstream training and inference.

## Deliverables

- Notebook/data-prep cell implementing the mapping.
- A markdown note in notebook comments describing the class precedence.
