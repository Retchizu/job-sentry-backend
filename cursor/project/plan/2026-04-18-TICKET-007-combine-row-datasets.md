# TICKET-007: Combine row-level job posting CSVs — Implementation Plan

## Overview

Implement a **reproducible merge** of `artifacts/datasets/fake_job_postings_rows.csv` and `artifacts/datasets/job_postings_rows.csv` into **`artifacts/datasets/combined_job_postings_rows.csv`**, with schema validation, type normalization, deduplication (exact + normalized-text), provenance column **`dataset_source`** (`fake_rows` | `job_rows`), and **reported distributions** including binary `fraudulent`, warning presence, and derived **`risk_class`** aligned with [TICKET-001](../tickets/TICKET-001-label-schema-and-mapping.md). Persist a **machine-readable merge summary** alongside the CSV (counts and rules).

## Current State Analysis

- **Ticket + research**: [TICKET-007](../tickets/TICKET-007-combine-row-datasets.md) defines inputs, required columns, processing, and deliverables. [Research 2026-04-18](../research/2026-04-18-TICKET-007-combine-row-datasets.md) confirms **no script or notebook** currently references these filenames; Phase 6 merge in `artifacts/ipynb/phase6_scam_detection.ipynb` is a **different pipeline** (D1/D2 splits → `merged_*.csv`).
- **Dependencies**: Root [`requirements.txt`](../../requirements.txt) already includes **`pandas>=2.0.0`**; no `Makefile` is present—verification uses **`pytest`** directly.
- **Sample data**: Both inputs use the same header  
  `id,job_title,job_desc,skills_desc,company_profile,rate_min,rate_max,currency,rate_type,created_at,fraudulent,warnings`.  
  The `warnings` field is JSON such as `{"flags":["..."],"note":"..."}` (non-empty `flags` ⇒ warning-present for reporting).

## Desired End State

1. **Runnable merge step**: One command (e.g. `python scripts/combine_job_postings_rows.py`) produces the combined CSV and summary from configurable default paths under `artifacts/datasets/`.
2. **Output artifacts**:
   - `artifacts/datasets/combined_job_postings_rows.csv` — readable by `pandas.read_csv`, containing original columns per row plus **`dataset_source`**, **`risk_class`** (int `0|1|2`), and **`warning_label`** (int `0|1`) for alignment with TICKET-001 wording.
   - `artifacts/datasets/combined_job_postings_rows.summary.json` — row counts, duplicate removal counts, and distribution tables (see Phase 2).
3. **Deterministic label rules**: `risk_class` matches TICKET-001 precedence (specified below); edge cases for invalid `warnings` JSON are defined and counted.
4. **Automated tests** for parsing, normalization, deduplication, and label derivation **without** requiring large CSVs in the repo.
5. **Manual check**: Re-run merge locally and spot-check that a training notebook that expects these columns can load the file (document which notebook path to open).

### Key Discoveries

- **Warnings JSON**: Live samples use an object with a **`flags` list**; “warning-present” for metrics = `flags` exists and `len(flags) > 0` after parse.
- **Layout**: Repo has no `scripts/` directory today—this plan adds **`scripts/`** for the CLI only, plus **`datasets_row_merge.py`** at the repo root for importable logic and tests.

## What We're NOT Doing

- **Not** modifying Phase 6 notebooks, `merged_train.csv`, or the Kaggle `fake_job_postings.csv` / Dataset 2 xlsx pipelines.
- **Not** implementing TICKET-002 train/val/test splits (depends on combined file + label strategy; separate ticket).
- **Not** changing FastAPI or inference code.
- **Not** committing multi-megabyte CSVs to git (artifacts remain local/untracked unless the project later adds LFS or a data policy).
- **Not** duplicating full TICKET-001 documentation here—reference the ticket; implement the **minimal** parsing needed for merge outputs and tests.

## Label and dedupe definitions (fixed for this plan)

**`warning_label` (binary)**  
- `1` if `warnings` parses as JSON and `flags` is a non-empty list.  
- `0` if `warnings` is null/empty, `flags` is missing or empty, or JSON is invalid (invalid rows counted in summary under `warnings_parse_errors`).

**`risk_class` (multiclass, int)** — same precedence as TICKET-001:

| Condition | `risk_class` | Meaning |
|-----------|--------------|---------|
| `fraudulent == 1` | `2` | fraud |
| Else if `warning_label == 1` | `1` | warning |
| Else | `0` | legit |

**Normalized text payload** (for near-duplicate removal):  
Concatenate `job_title`, `job_desc`, `skills_desc`, `company_profile` in that order with a single space between fields; apply Unicode NFC normalization, strip, collapse internal whitespace to single spaces, **casefold** for comparison. Two rows with the same normalized payload after exact-dedupe step: keep the **first** occurrence in concat order (`fake_rows` block then `job_rows`, original row order preserved within each file).

**Exact duplicates**: `DataFrame.duplicated(keep="first")` on **all columns** after both sources are loaded and `dataset_source` is attached (so cross-file identical rows still dedupe if every field including source matches—or optionally exclude `dataset_source` from exact match only if product wants; **this plan uses all columns including `dataset_source`** so two rows identical in every field including source collapse to one; if the same text appears in both files with different `id`, they are **not** exact dupes unless all fields match).

Clarification for cross-source duplicate text: **Near-dedupe** (step 2) removes same **normalized text payload** regardless of `id`/`dataset_source`, keeping first in concat order—this matches the ticket’s “near-duplicates (same normalized text payload)”.

Order of operations:

1. Load both CSVs, assert required columns, assign `dataset_source`.
2. Concatenate (`ignore_index=True`).
3. Normalize dtypes (Phase 2).
4. Count and drop **exact** duplicate rows (full row).
5. Count and drop **near** duplicates by normalized payload (keep first).
6. Compute `warning_label` and `risk_class`.
7. Write CSV + summary JSON.

## Implementation Approach

Keep **`app/`** unchanged. Add:

1. **`datasets_row_merge.py`** at the repository root — all pure logic (validation, dtypes, dedupe, labels, summary dict) so tests can `import datasets_row_merge` with no `PYTHONPATH` changes.
2. **`scripts/combine_job_postings_rows.py`** — thin CLI (`argparse`) that imports `datasets_row_merge`, writes CSV + JSON, prints stdout summary.

## Phase 1: Core merge library (`datasets_row_merge.py`)

### Overview

Implement validation, dtype normalization, deduplication, label derivation, and summary structure.

### Changes Required

#### 1. New module `datasets_row_merge.py`

**Changes**:

- Constants: `REQUIRED_COLUMNS` tuple matching TICKET-007.
- `assert_same_columns(fake_df, job_df) -> None` — raise `ValueError` with diff if mismatch.
- `assign_source(df, source: Literal["fake_rows","job_rows"]) -> DataFrame` — copy with `dataset_source` set.
- `normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame` — at minimum:
  - `fraudulent` → `int` (coerce invalid to NaN then drop or raise; **plan: drop rows with NaN `fraudulent` after coercion and count in summary**).
  - `rate_min`, `rate_max` → nullable float (`pd.NA` for empty).
  - Text columns → `str` with `fillna("")`.
  - `created_at` → parse datetime if possible, else leave as string (document in summary).
- `normalized_text_key(row) -> str` — implements payload definition above.
- `parse_warnings_flags(raw) -> tuple[bool, bool]` — returns `(has_flags, parse_error)` where `parse_error` is True if JSON invalid or schema unexpected.
- `derive_labels(df) -> DataFrame` — adds `warning_label` and `risk_class`.
- `dedupe_exact(df) -> tuple[pd.DataFrame, int]` — returns dropped count.
- `dedupe_near(df) -> tuple[pd.DataFrame, int]` — on normalized key.
- `merge_sources(path_fake, path_job) -> tuple[pd.DataFrame, dict]` — orchestrates; `dict` is summary stats before writing.

### Success Criteria

#### Automated Verification

- [x] `pytest tests/test_row_merge.py` passes (add in Phase 3).
- [x] `python -c "import datasets_row_merge; print(datasets_row_merge.REQUIRED_COLUMNS)"` runs from repo root.

### Manual Verification

- [ ] Import `datasets_row_merge` from a Python REPL at repo root.

---

## Phase 2: CLI, outputs, and summary JSON

### Overview

Wire defaults to `artifacts/datasets/`, write `combined_job_postings_rows.csv` and `combined_job_postings_rows.summary.json`, print human-readable summary to stdout.

### Changes Required

#### 1. `scripts/combine_job_postings_rows.py`

**Changes**:

- Args: `--fake`, `--job`, `--out-csv`, `--out-summary` with defaults:
  - `artifacts/datasets/fake_job_postings_rows.csv`
  - `artifacts/datasets/job_postings_rows.csv`
  - `artifacts/datasets/combined_job_postings_rows.csv`
  - `artifacts/datasets/combined_job_postings_rows.summary.json`
- Call merge pipeline; write CSV with `index=False`.
- **Summary JSON** fields (minimum):
  - `schema_version`: `"1.0"`
  - `inputs`: paths and row counts read
  - `concat_rows_before_dedupe`
  - `exact_duplicates_removed`
  - `near_duplicates_removed`
  - `final_rows`
  - `fraudulent_counts`: `{0: n, 1: m}`
  - `warning_label_counts`: `{0: ..., 1: ...}`
  - `risk_class_counts`: `{0: ..., 1: ..., 2: ...}`
  - `warnings_parse_errors`
  - `rules`: short strings echoing definitions above

### Success Criteria

#### Automated Verification

- [x] `python scripts/combine_job_postings_rows.py --help` exits 0.
- [x] With **temporary** tiny CSV fixtures in `tests/fixtures/` (checked into git, <20 rows each), a pytest **integration** test writes to `tmp_path` and asserts JSON schema keys and row counts (optional in Phase 3).

#### Manual Verification

- [ ] Run CLI against real `artifacts/datasets/` files; confirm output row count + summary match expectations printed in stdout.

**Implementation note**: Pause after manual run for stakeholder confirmation before relying on artifacts in downstream notebooks.

---

## Phase 3: Automated tests

### Overview

Unit tests for label parsing, dedupe logic, and end-to-end merge on tiny fixtures.

### Changes Required

#### 1. `tests/test_row_merge.py`

**Changes**:

- Fixtures: minimal DataFrames with edge cases—invalid JSON warnings, empty flags, `fraudulent=1` with flags (fraud wins), legit + flags → warning.
- Test normalized text collision drops second row.
- Test `REQUIRED_COLUMNS` enforced.

### Success Criteria

#### Automated Verification

- [x] `pytest tests/test_row_merge.py -q` passes.
- [x] `pytest` for full suite passes: `pytest -q`

#### Manual Verification

- [ ] None required if CI green locally.

---

## Phase 4: Notebook compatibility check (documentation-only in repo)

### Overview

Confirm which training notebook will consume `combined_job_postings_rows.csv` next (TICKET-002). No notebook code change in this task unless a **single cell** is explicitly added to load the new CSV for smoke test.

### Changes Required

- Add a **short comment** in `cursor/project/tickets/TICKET-007-combine-row-datasets.md` or a one-line note in `combined_job_postings_rows.summary.json` is sufficient; optionally add **README under `artifacts/datasets/`** only if the repo already documents artifacts there—**otherwise** document in the merge summary JSON only to avoid unsolicited markdown.

**Optional** (only if requested): one markdown file `cursor/project/notes/combined_job_postings_rows.md` describing run command—**out of scope unless user asks** per focused diffs.

### Success Criteria

#### Manual Verification

- [ ] Open the target preprocessing notebook (from TICKET-002 / Implementation Plan 2) and verify column names overlap expected inputs; record notebook path in the PR description.

---

## Testing Strategy

### Unit Tests

- `parse_warnings_flags` for valid JSON, empty flags, malformed JSON.
- `derive_risk_class` precedence.
- `normalized_text_key` stability.
- Deduplication counts.

### Integration

- Two 3-row CSVs in `tests/fixtures/` → merged output + JSON written to `tmp_path`.

### Manual Testing Steps

1. Run `python scripts/combine_job_postings_rows.py` from repo root with real artifacts.
2. `pandas.read_csv("artifacts/datasets/combined_job_postings_rows.csv")` and inspect `risk_class`, `dataset_source` value counts.
3. Compare summary JSON `final_rows` to CSV `len(df)`.

## Performance Considerations

- For files that fit in memory (current scope), single-pass pandas is sufficient.
- If either CSV grows beyond RAM, a future chunked refactor would be a separate task—**not** required now.

## Migration Notes

- First run **creates** new files; no DB migration.
- Downstream notebooks should switch input path from `job_postings_rows.csv` to `combined_job_postings_rows.csv` when TICKET-002 is implemented.

## References

- Ticket: [`cursor/project/tickets/TICKET-007-combine-row-datasets.md`](../tickets/TICKET-007-combine-row-datasets.md)
- Research: [`cursor/project/research/2026-04-18-TICKET-007-combine-row-datasets.md`](../research/2026-04-18-TICKET-007-combine-row-datasets.md)
- Label precedence: [`cursor/project/tickets/TICKET-001-label-schema-and-mapping.md`](../tickets/TICKET-001-label-schema-and-mapping.md)
- Dependencies: [`requirements.txt`](../../requirements.txt)
- Related pipeline (different inputs): `artifacts/ipynb/phase6_scam_detection.ipynb`

## Sync note

`humanlayer thoughts sync` was not available in the environment used to author this plan; run it if your workflow requires it.
