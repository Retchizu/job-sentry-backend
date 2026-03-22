# Spelling, Grammar, Punctuation, and ALL CAPS Features — Implementation Plan

## Overview

This plan implements the four linguistic feature areas identified in the research document: **spelling error rate**, **grammar score**, **unusual punctuation frequency**, and **ALL CAPS** — in the Dataset 2 pipeline and, where applicable, in the backend. The research confirmed that grammar/spelling and unusual punctuation are either unimplemented or not wired; ALL CAPS is already present in Dataset 2 and as a warning in the backend. This plan wires and extends those features so they are available for training and, if desired, for inference.

## Current State Analysis

- **Dataset 2** (`thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb`): `add_linguistic_features` calls sentence stats, punctuation, sentiment, second-person, and scam-phrase functions but **does not** call `compute_grammar_error_features`. Grammar is implemented (LanguageTool) but unused. Punctuation already includes exclamation/question/ellipsis and `all_caps_word_ratio`; “unusual punctuation” as a single metric is not defined. Structural/linguistic columns are built from `combined_text_`* and a fixed list; new numeric columns with that prefix are picked up automatically.
- **Phase 6** (`phase6_scam_detection.ipynb`): Uses only `build_structural_features` (no linguistic columns). TF-IDF + structural features are stacked; classifier expects a fixed dimension.
- **Backend** (`app/preprocessing.py`, `app/traditional_ml.py`): Warning signals include “Excessive use of capital letters” (caps words > 5) and “Excessive exclamation marks”. Phase 6 predictor transforms text with TF-IDF only; if the classifier was trained with extra numeric features, the backend pads with zeros (feature mismatch). There is no computation of grammar, spelling, or unusual punctuation in the app.

### Key Discoveries

- Grammar: `compute_grammar_error_features` exists in Dataset 2 notebook (LanguageTool, truncated to 4000 chars); it is never called in `add_linguistic_features` (see notebook cell containing `add_linguistic_features` loop — no `compute_grammar_error_features` call).
- ALL CAPS: `compute_punctuation_features` already returns `all_caps_word_ratio`; it is appended as `combined_text_all_caps_word_ratio` in `add_linguistic_features`. The structural/linguistic column list includes any column starting with `combined_text_`, so this is already in the feature set for Dataset 2.
- Punctuation: `compute_punctuation_features` returns exclamation/question/ellipsis counts and max runs; “unusual punctuation” (e.g. runs ≥ 2 or specific symbols) is not yet defined or added.
- Backend: `detect_warning_signals` uses `re.findall(r"\b[A-Z]{3,}\b", combined_text)` and `len(caps_words) > 5`; no numeric ALL CAPS ratio is exposed on `PreprocessedInput`.

## Desired End State

1. **Dataset 2**
  - Grammar and spelling (and optionally a single grammar score) are computed and added as numeric columns in `add_linguistic_features`, and are included in the structural/linguistic feature set used for training (e.g. Phase 4.1).
  - Unusual punctuation is defined, computed in `compute_punctuation_features` (or a small helper), and added as columns in `add_linguistic_features`, and included in the same feature set.
  - ALL CAPS remains in the feature set (no change required beyond verification).
2. **Backend (optional)**
  - If we want an explicit ALL CAPS ratio for warnings or future numeric features: `preprocessing.py` computes the same ratio as the notebook and exposes it (e.g. on `PreprocessedInput` or inside `detect_warning_signals`).
  - If Phase 6 is later extended with linguistic features: the thesis pipeline and backend must agree on the same numeric feature set and order; this plan does not extend Phase 6 or the predictor in this iteration.

**Verification**

- In Dataset 2: After running the notebook, train/val/test DataFrames contain the new columns; `structural_linguistic_feature_columns` (or equivalent) includes them; Phase 4.1 training runs and uses the expanded feature set.
- In backend: If ALL CAPS ratio is added, a unit test or manual check confirms the ratio matches the notebook formula for a known input.

## What We're NOT Doing

- Adding LanguageTool (or another grammar/spelling dependency) to the backend in this plan.
- Changing Phase 6 merged pipeline to include linguistic features (that would require a separate plan: extend `build_structural_features` or add a parallel linguistic step, retrain, and then add backend feature construction).
- Implementing a separate spell-checker (e.g. PySpellChecker) for spelling; we use LanguageTool’s output for both grammar and spelling-derived metrics.
- Changing the list of scam keywords or warning thresholds beyond any explicit “use ratio for warning” if we add ALL CAPS ratio to the backend.

## Implementation Approach

- Implement and wire features in the Dataset 2 notebook first (grammar/spelling, unusual punctuation, verify ALL CAPS).
- Optionally add ALL CAPS ratio in the backend for consistency and future use in warnings or numeric features.
- Leave Phase 6 and full backend linguistic feature construction for a follow-up if we want the merged model to use these features.

---

## Phase 1: Wire Grammar and Spelling in Dataset 2

- [x] Phase 1 implemented (extend compute_grammar_error_features, wire in add_linguistic_features).

### Overview

Call `compute_grammar_error_features` inside `add_linguistic_features`, add the three existing keys as columns, and add optional spelling error ratio and grammar score derived from the same LanguageTool run.

### Changes Required

#### 1. Extend `compute_grammar_error_features` (optional but recommended)

**File**: `thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb` (cell defining `compute_grammar_error_features`)

- Keep existing return: `grammar_error_count`, `grammar_error_ratio`, `has_grammar_errors`.
- Add (from same `tool.check()` result):
  - **Spelling**: Either count issues where `match.rule.category.id == 'SPELLING'` (or equivalent LanguageTool API) and compute `spelling_error_ratio = spelling_count / max(1, word_count)` using token count from the same text, or use total `error_count` as a proxy and expose `grammar_and_spelling_error_ratio` (errors per word). Document the choice in the notebook.
  - **Grammar score**: e.g. `grammar_score = 1.0 - min(1.0, grammar_error_ratio / k)` with a chosen scale `k` (e.g. 2.0), or `1.0 - min(1.0, grammar_error_count / max(1, word_count))`. Add as one extra key in the returned dict.

#### 2. Wire grammar (and new keys) in `add_linguistic_features`

**File**: Same notebook, cell containing `add_linguistic_features`.

- In the per-row loop, after existing feature calls, call `gf = compute_grammar_error_features(text_value)`.
- Append to new lists: `grammar_error_counts`, `grammar_error_ratios`, `has_grammar_errors_list`, and if added: `spelling_error_ratios`, `grammar_scores`.
- After the loop, add columns:
  - `dataframe[f"{text_column}_grammar_error_count"] = grammar_error_counts`
  - `dataframe[f"{text_column}_grammar_error_ratio"] = grammar_error_ratios`
  - `dataframe[f"{text_column}_has_grammar_errors"] = has_grammar_errors_list`
  - If implemented: `dataframe[f"{text_column}_spelling_error_ratio"] = spelling_error_ratios`, `dataframe[f"{text_column}_grammar_score"] = grammar_scores`

Ensure column names use the same `text_column` prefix (e.g. `combined_text_`) so they are included in the structural/linguistic column list that uses `startswith("combined_text_")`.

#### 3. Ensure new columns are in the feature set

- The existing logic that builds `structural_linguistic_feature_columns` from columns starting with `combined_text_` will include the new columns. Verify that `get_numeric_feature_columns` does not exclude them (they are numeric and not in `EXCLUDE_COLS`). No change needed unless the notebook uses an explicit allow-list that omits them; in that case add the new column names to the list.

### Success Criteria

#### Automated Verification

- Dataset 2 notebook runs from “add_linguistic_features” through feature matrix build without error.
- New columns appear in `train_merged_dataframe` / `validation_merged_dataframe` / `test_merged_dataframe` (e.g. `combined_text_grammar_error_count`, `combined_text_grammar_error_ratio`, `combined_text_has_grammar_errors`, and if added `combined_text_spelling_error_ratio`, `combined_text_grammar_score`).
- Phase 4.1 training cell runs with the expanded feature set (no dimension mismatch).

#### Manual Verification

- For a few rows, spot-check that grammar/spelling counts and ratios are non-negative and plausible (e.g. more errors in clearly bad text).
- If LanguageTool is unavailable, features default to 0 and the notebook does not crash.

**Implementation Note**: After completing this phase and all automated checks pass, pause for manual confirmation before proceeding to Phase 2.

---

## Phase 2: Add Unusual Punctuation in Dataset 2

- [x] Phase 2 implemented (unusual_punctuation_count/frequency in compute_punctuation_features and add_linguistic_features).

### Overview

Define “unusual punctuation” (e.g. runs of `!` or `?` of length ≥ 2, and/or ellipsis), add counts and a frequency (per token or per character) in `compute_punctuation_features`, and add the corresponding columns in `add_linguistic_features`.

### Changes Required

#### 1. Define and compute unusual punctuation

**File**: `thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb` (cell defining `compute_punctuation_features`).

- **Definition**: Unusual punctuation = (1) runs of `!` of length ≥ 2, (2) runs of `?` of length ≥ 2, (3) occurrence of `...` (ellipsis). Count each run/occurrence as one “unusual” event.
- In `compute_punctuation_features`:
  - Count runs of `!` with length ≥ 2 (reuse or derive from existing max run logic; e.g. count runs with length >= 2).
  - Count runs of `?` with length ≥ 2.
  - Count ellipsis (e.g. `text.count("...")` already as `ellipsis_count_value`; can use that or a separate counter).
  - Set `unusual_punctuation_count = count_exclamation_runs_ge2 + count_question_runs_ge2 + ellipsis_count` (or define a variant; document in notebook).
  - Set `unusual_punctuation_frequency = unusual_punctuation_count / max(1, len(original_tokens))` (per token) or `unusual_punctuation_count / max(1, len(text))` (per character). Choose one and document.
- Add to the returned dict: `unusual_punctuation_count`, `unusual_punctuation_frequency`.

#### 2. Add columns in `add_linguistic_features`

- In the same loop, read `pf["unusual_punctuation_count"]` and `pf["unusual_punctuation_frequency"]` and append to new lists.
- After the loop: `dataframe[f"{text_column}_unusual_punctuation_count"] = ...`, `dataframe[f"{text_column}_unusual_punctuation_frequency"] = ...`.

#### 3. Feature set

- New columns have the `combined_text`_ prefix and will be picked up by the existing structural/linguistic column logic. Confirm they are numeric and not in `EXCLUDE_COLS`.

### Success Criteria

#### Automated Verification

- Dataset 2 notebook runs without error; new columns `combined_text_unusual_punctuation_count` and `combined_text_unusual_punctuation_frequency` exist in train/val/test DataFrames.
- Phase 4.1 feature matrix build and training run with the expanded set.

#### Manual Verification

- For a sample with “!!” or “???” or “...”, values are non-zero and consistent with the definition.

**Implementation Note**: After completing this phase and all automated checks pass, pause for manual confirmation before proceeding to Phase 3.

---

## Phase 3: Verify ALL CAPS in Feature Set and Optional Backend Ratio

- [x] Phase 3 implemented (ALL CAPS verified in Dataset 2 via combined_text_ prefix; backend compute_all_caps_ratio + PreprocessedInput.all_caps_ratio + warning).

### Overview

Confirm that ALL CAPS is already included in Dataset 2 training features; optionally add an explicit ALL CAPS ratio in the backend for warnings or future numeric features.

### Changes Required

#### 1. Dataset 2 (verification only)

- Confirm that `combined_text_all_caps_word_ratio` is present in the DataFrame after `add_linguistic_features` and is included in `structural_linguistic_feature_columns` (or in the columns passed to the Phase 4.1 combined matrix). No code change if already true.

#### 2. Backend — optional ALL CAPS ratio

**File**: `app/preprocessing.py`

- Add a helper, e.g. `def compute_all_caps_ratio(combined_text: str) -> float`, that:
  - Uses the same logic as the notebook: tokens with `re.findall(r"\b\S+\b", text)`, then `all_caps_count = sum(1 for t in tokens if t.isupper() and len(t) > 1)`, `ratio = all_caps_count / len(tokens)` if tokens else 0.
  - Returns a float in [0, 1].
- Call it from `preprocess_job_post` and either:
  - Add a field to `PreprocessedInput`, e.g. `all_caps_ratio: float`, and set it there, or
  - Use it inside `detect_warning_signals` (e.g. add a warning when ratio exceeds a threshold in addition to or instead of the current “caps words > 5” rule). Document the chosen behavior.

### Success Criteria

#### Automated Verification

- Dataset 2: `combined_text_all_caps_word_ratio` is in the feature columns used for training (spot-check or assert in notebook).
- Backend: If ALL CAPS ratio was added, `pytest` or a small script: for a known string, `compute_all_caps_ratio(text)` matches the notebook’s `compute_punctuation_features(text)["all_caps_word_ratio"]` (run notebook logic in test or hardcode expected value).

#### Manual Verification

- Backend: If ratio is exposed on `PreprocessedInput`, one prediction request returns a plausible value; if used in warnings, a post with many ALL CAPS words triggers the expected signal.

**Implementation Note**: After completing this phase and all automated checks pass, pause for manual confirmation.

---

## Phase 4: Documentation and Export (if needed for backend later)

- [x] Phase 4 implemented (cursor/project/notes/dataset2-linguistic-feature-set.md).

### Overview

Document the exact feature set and formulas so that a future backend (or Phase 6) can reproduce the same numeric features at inference time.

### Changes Required

- In the Dataset 2 notebook or in `cursor/project/notes/`, add a short section listing:
  - All linguistic feature column names (including grammar, spelling, unusual punctuation, ALL CAPS) and their order if order matters for a saved artifact.
  - Formulas: e.g. `all_caps_word_ratio`, `unusual_punctuation_count`/`frequency`, `grammar_error_ratio`, `spelling_error_ratio`, `grammar_score`.
- If the thesis later exports a Phase 6 (or Phase 4.1) artifact that includes these numeric columns, document the expected column order and scaling (e.g. StandardScaler) so the backend can mirror it.

### Success Criteria

#### Manual Verification

- A developer can read the doc and implement the same features in another environment (e.g. backend) without reverse-engineering the notebook.

---

## Testing Strategy

### Unit Tests (backend, if ALL CAPS ratio added)

- `compute_all_caps_ratio("")` returns 0.
- `compute_all_caps_ratio("Hello WORLD")` returns 1/2 (one all-caps word “WORLD”, two tokens) or equivalent per notebook definition.
- Optional: test that `detect_warning_signals` still triggers “Excessive use of capital letters” when appropriate.

### Notebook / Integration

- Run Dataset 2 notebook end-to-end after Phases 1 and 2; confirm no cell errors and that Phase 4.1 training completes.
- Optionally add a notebook assertion that the expected new column names exist in `structural_linguistic_feature_columns` or in the numeric feature list.

### Manual

- Spot-check grammar/spelling and unusual punctuation values on a few rows.
- If backend ratio is added, call `POST /predict` with a job post containing many ALL CAPS words and confirm warning or response field.

## Performance Considerations

- LanguageTool in the notebook can be slow for large corpora; text is already truncated to 4000 chars. Consider running `add_linguistic_features` with grammar only on a sample during development, or document that full-dataset run may take significant time.
- Backend: No LanguageTool in this plan; only optional lightweight ALL CAPS ratio computation.

## Migration Notes

- No database or artifact migration in this plan. If you later train a new Phase 4.1 or Phase 6 model with the new columns, save the new artifacts and update the backend to load them; if the backend ever needs to supply numeric features, it must compute the same set in the same order (see Phase 4 documentation).

## References

- Research: `cursor/project/research/2025-03-09-features-spelling-grammar-punctuation-caps.md`
- Dataset 2 notebook: `thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb` — `compute_punctuation_features`, `compute_grammar_error_features`, `add_linguistic_features`
- Backend: `app/preprocessing.py` — `detect_warning_signals`, `PreprocessedInput`; `app/traditional_ml.py` — `Phase6MergedPredictor`
- Implementation Plan 1: `cursor/project/notes/Implementation Plan 1.md` (Phase 2 feature engineering)

