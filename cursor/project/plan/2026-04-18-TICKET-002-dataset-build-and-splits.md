# TICKET-002: Dataset build and stratified splits — Implementation Plan

## Overview

Implement a **reproducible Python pipeline** that takes the **row-level combined dataset** (post–TICKET-007), adds a null-safe **`combined_text`** column from the four text fields, performs **train/validation/test** splits with **fixed seed** and **stratification on `risk_class`**, verifies **no `id` overlap** across splits, and writes **`merged_train.csv`**, **`merged_val.csv`**, **`merged_test.csv`** under **`artifacts/data/processed/`**, plus automated tests. This closes the gap between **`datasets_row_merge.derive_labels`** (labels only) and the ticket acceptance criteria.

## Current State Analysis

- **TICKET-007** is implemented: `datasets_row_merge.py` merges `fake_job_postings_rows.csv` + `job_postings_rows.csv`, dedupes, and derives **`warning_label`** and **`risk_class`** (TICKET-001 precedence: fraud → 2, warning flags → 1, else 0). `scripts/combine_job_postings_rows.py` writes **`artifacts/datasets/combined_job_postings_rows.csv`**.
- **No Python module** currently performs **`train_test_split`** on **`risk_class`** or writes **`artifacts/data/processed/merged_*.csv`** for the row-level schema.
- **Phase 6 notebooks** (`artifacts/ipynb/phase6_scam_detection.ipynb`) write **different** `merged_*.csv` files: **D1+D2** harmonized tables stratified on **`fraudulent`**, without **`risk_class`**. Reusing the same filenames under `artifacts/data/processed/` **overwrites** those artifacts if the new script is run in place.

## Desired End State

1. Running **one documented command** (after TICKET-007) produces three CSVs:
   - `artifacts/data/processed/merged_train.csv`
   - `artifacts/data/processed/merged_val.csv`
   - `artifacts/data/processed/merged_test.csv`
2. Each row includes at least: original row-level columns used for training/traceability, **`combined_text`**, **`risk_class`**, and stable **`id`** for leakage checks.
3. **Stratified** split ratios match the existing notebook convention (**70% / 15% / 15%**): first split 70% train vs 30% holdout (stratified on `risk_class`), second split splits the 30% equally into val and test (stratified on `risk_class` in the holdout). **`random_state=42`** everywhere.
4. **Automated tests** validate stratification wiring, **disjoint `id` sets** across splits, and **`combined_text`** construction on synthetic data.
5. A **machine-readable summary** (JSON next to the CSVs or printed to stdout) records row counts, **`risk_class`** counts per split, and confirms zero `id` intersection.

### Key Discoveries

- Label logic is **already centralized** in `datasets_row_merge.py` (`derive_labels`, lines 145–159); TICKET-002 should **not** re-derive labels from raw `job_postings_rows.csv` unless explicitly requested — it should consume the **combined** output so **`fake_rows`** and **`job_rows`** stay in distribution.
- `TEXT_COLUMNS` in `datasets_row_merge.py` (lines 32–37) are exactly the ticket’s four fields for **`combined_text`**.
- Tests use **`pytest -q`** per `README.md` (lines 111–114).

## What We're NOT Doing

- **Not** changing Phase 6 notebook code in this ticket (`dataset2_scam_detection.ipynb` / `phase6_scam_detection.ipynb`); migrating **`phase6_deep_learning.ipynb`** / **`phase6_hybrid_fused.ipynb`** to the new row-level **`merged_*.csv`** schema is **TICKET-003+** (sequential model / training updates).
- **Not** implementing **TICKET-008** (`POST /predict`); only alignment note: API already uses TICKET-001 **0/1/2** semantics (`README.md` lines 64–72).
- **Not** re-running full notebook ETL (HTML cleaning, lemmatization, etc.) on row text unless a follow-up ticket requires it — TICKET-002 scope is **concatenation + null-safe strings** as written.
- **Not** adding `make` targets (no `Makefile` in repo); verification uses **`pytest`** directly.

## Implementation Approach

1. Add a small **library module** (e.g. `datasets_row_splits.py` at repo root, alongside `datasets_row_merge.py`) containing:
   - **`build_combined_text(df) -> pd.Series`**: for each row, `job_title`, `job_desc`, `skills_desc`, `company_profile` → `fillna("")`, join with single spaces, **`.str.strip()`** (match Phase 6 harmonize pattern).
   - **`stratified_train_val_test(df, *, label_col, id_col, random_state, train_frac, val_frac_of_rest)`** (or fixed 0.7 / 0.15 / 0.15): uses **`sklearn.model_selection.train_test_split`** twice with **`stratify`**, same pattern as `main_scam_detection.ipynb`.
   - **`assert_ids_disjoint(train, val, test, id_col="id")`**: `set` intersection must be empty for all pairs; raise with counts if not.
   - **`min_class_count` guard**: if any class in `risk_class` has **fewer than 2** rows, **`stratify`** is invalid — **fail fast** with a clear error listing per-class counts (do not silently drop stratification without documentation).

2. Add **CLI** `scripts/build_row_level_merged_splits.py` (name illustrative):
   - **Default `--input`**: `artifacts/datasets/combined_job_postings_rows.csv` (must exist after `python scripts/combine_job_postings_rows.py`).
   - **Optional `--job-only`**: filter `dataset_source == "job_rows"` before splitting, to satisfy a strict reading of “from `job_postings_rows.csv` only” when needed (document that default includes both sources for better coverage).
   - **`--out-dir`**: default `artifacts/data/processed/`.
   - **`--seed`**: default `42`.
   - After **`build_combined_text`**, select output columns: **all columns present in the input** that are needed for traceability, **plus** `combined_text` (ensure no duplicate column names). Minimum: **`REQUIRED_COLUMNS` fields + `dataset_source` + `warning_label` + `risk_class` + `combined_text`** (align with `combine_job_postings_rows.py` column order where practical).
   - Write **`merged_train.csv`**, **`merged_val.csv`**, **`merged_test.csv`** and **`merged_splits.summary.json`** (row counts, `risk_class` value_counts per split, input path, seed, `job_only` flag).

3. **Tests** in `tests/test_row_splits.py`:
   - Small synthetic DataFrame with **6+ rows**, balanced **`risk_class`**, string **`id`**: assert split sizes, stratify preserves approximate proportions, **`assert_ids_disjoint`** passes.
   - Test **`build_combined_text`** with `NaN` in one field.
   - Test error path: **unstratifiable** class (one sample in a class) raises expected error.

4. **Documentation**: Add a short subsection under **`README.md`** “Data: row-level splits (TICKET-002)” with the two commands: combine (TICKET-007) then split (TICKET-002), and a **warning** that **`merged_*.csv`** replace any prior Phase-6–generated files at the same paths.

## Phase 1: Core split + `combined_text` library

### Overview

Implement **`datasets_row_splits.py`** with **`build_combined_text`**, **stratified 70/15/15** split, **`id` disjoint assertions**, and summary dict construction.

### Changes Required

#### 1. New module `datasets_row_splits.py`

**File**: `datasets_row_splits.py`  
**Changes**: Add functions described above; depend on **`pandas`** and **`sklearn.model_selection.train_test_split`** (`scikit-learn` is already listed in `requirements.txt`).

```python
# Illustrative signatures (implement with docstrings and type hints)

def build_combined_text(df: pd.DataFrame) -> pd.Series: ...

def stratified_train_val_test(
    df: pd.DataFrame,
    *,
    label_col: str = "risk_class",
    id_col: str = "id",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...

def assert_split_ids_disjoint(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_col: str = "id",
) -> None: ...

def split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    label_col: str = "risk_class",
) -> dict: ...
```

### Success Criteria

#### Automated Verification

- [x] `python -c "import datasets_row_splits"` succeeds.
- [x] `pytest -q tests/test_row_splits.py` passes.
- [x] `pip install -r requirements.txt` still succeeds after any new dependency.

#### Manual Verification

- [ ] On a real `combined_job_postings_rows.csv`, split runs without error and printed/JSON summary shows plausible **`risk_class`** counts per split.

**Implementation Note**: After automated checks pass, confirm with real data once before Phase 2.

---

## Phase 2: CLI and artifacts

### Overview

Wire the library to **`scripts/build_row_level_merged_splits.py`**, default paths, **`--job-only`**, and write **`merged_*.csv`** + summary JSON under **`artifacts/data/processed/`**.

### Changes Required

#### 1. New script `scripts/build_row_level_merged_splits.py`

**File**: `scripts/build_row_level_merged_splits.py`  
**Changes**: argparse, read CSV, optional filter, call **`build_combined_text`**, split, assert disjoint **`id`**, write outputs.

#### 2. `README.md`

**File**: `README.md`  
**Changes**: Short “Row-level merged splits” section: prerequisite TICKET-007 command, TICKET-002 command, filename collision note vs Phase 6 notebooks.

### Success Criteria

#### Automated Verification

- [x] `python scripts/build_row_level_merged_splits.py --help` exits 0.
- [x] `pytest -q` passes for the whole repo (no regressions).

#### Manual Verification

- [ ] `artifacts/data/processed/merged_train.csv` opens in pandas and includes **`combined_text`** and **`risk_class`**.
- [ ] Summary JSON reflects the same row counts as the three CSVs.

---

## Phase 3: Requirements and integration polish

### Overview

Confirm **`scikit-learn`** remains available for **`train_test_split`** (already listed in **`requirements.txt`** lines 13–15); only bump or add deps if implementation reveals a gap.

### Changes Required

#### 1. `requirements.txt`

**File**: `requirements.txt`  
**Changes**: **None** expected; only edit if a new import is introduced (e.g. optional **`numpy`** typing helpers — **`pandas`** already pulls **`numpy`**).

### Success Criteria

#### Automated Verification

- [x] Fresh venv: `pip install -r requirements.txt && pytest -q` passes.

#### Manual Verification

- [ ] None required beyond Phase 2.

---

## Testing Strategy

### Unit Tests

- **`build_combined_text`**: `NaN`, empty string, normal text.
- **Splits**: exact sizes for a fixed small frame; **`id`** disjoint; **`risk_class`** stratify failure when a class has count 1.

### Integration / Smoke

- Optional: one test that loads **`tests/fixtures`** tiny combined CSV (may need a **new fixture** built from merged tiny outputs) end-to-end through the CLI via **`subprocess`** or by calling `main()` — only if it stays fast and stable.

### Manual Testing Steps

1. Run `python scripts/combine_job_postings_rows.py` to refresh **`combined_job_postings_rows.csv`**.
2. Run `python scripts/build_row_level_merged_splits.py`.
3. In a Python shell, load the three outputs and verify **`set(train.id) & set(val.id) == set()`** (and likewise for test).

## Performance Considerations

- Single **`pd.read_csv`** / writes; dataset fits typical memory for this project. No chunking required unless files grow substantially.

## Migration Notes

- **Backup** existing **`artifacts/data/processed/merged_*.csv`** if teams still rely on **Phase 6 D1+D2** binary **`fraudulent`** files before running the new script.
- Downstream notebooks that expect **`dataset_source` in {D1, D2}** will **not** read the new files without updates — call that out in README and defer notebook edits to **TICKET-003 / TICKET-004**.

## References

- Original ticket: `cursor/project/tickets/TICKET-002-dataset-build-and-splits.md`
- Upstream merge: `cursor/project/tickets/TICKET-007-combine-row-datasets.md`, `datasets_row_merge.py`, `scripts/combine_job_postings_rows.py`
- Label contract: `cursor/project/tickets/TICKET-001-label-schema-and-mapping.md`
- Research: `cursor/project/research/2026-04-18-TICKET-002-dataset-build-and-splits.md`
- Related roadmap: `cursor/project/tickets/README.md` (TICKET-007 → TICKET-001 → TICKET-002)
- API alignment (future training → deploy): `cursor/project/tickets/TICKET-008-backend-predict-deployment.md`
