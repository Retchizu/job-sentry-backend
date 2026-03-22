---
date: 2025-03-09T00:00:00Z
researcher: documentation
git_commit: a41aea8ccf37a2dc4a249e6d66f96daf13046f52
branch: main
repository: job-sentry-backend
topic: "Features: spelling error rate, grammar score, unusual punctuation, ALL CAPS — where they exist and how to add them"
tags: [research, codebase, features, linguistic, preprocessing, dataset2, phase6]
status: complete
last_updated: 2025-03-09
last_updated_by: documentation
---

# Research: Features — Spelling Error Rate, Grammar Score, Unusual Punctuation, ALL CAPS

**Date**: 2025-03-09  
**Repository**: job-sentry-backend  
**Branch**: main  
**Git Commit**: a41aea8ccf37a2dc4a249e6d66f96daf13046f52  

## Research Question

Where are features defined in the codebase, and how can we add (or expose):

- Spelling error rate  
- Grammar score  
- Unusual punctuation frequency  
- ALL CAPS words  

## Summary

- **ALL CAPS**: Already implemented in two places — (1) as a **linguistic feature** in Dataset 2 (`all_caps_word_ratio` in `compute_punctuation_features` and `add_linguistic_features`), and (2) as a **warning signal** in the backend (`app/preprocessing.py`: caps words count > 5).
- **Grammar / spelling**: A **grammar/spelling feature** exists in Dataset 2 (`compute_grammar_error_features` using LanguageTool) but is **not** currently wired into `add_linguistic_features`; the loop never calls it or adds its columns. Adding it means calling `compute_grammar_error_features` in that loop and appending the three columns (`grammar_error_count`, `grammar_error_ratio`, `has_grammar_errors`). A **spelling error rate** can be derived from the same LanguageTool output (e.g. errors / words or errors / sentences) or added as a separate metric.
- **Unusual punctuation**: **Punctuation features** already exist in Dataset 2 (exclamation count, question mark count, ellipsis, max runs). “Unusual punctuation frequency” can be added by extending `compute_punctuation_features` (e.g. ratio of unusual punctuation to total characters/tokens, or counts of specific unusual characters) and then ensuring those columns are included in the structural/linguistic column list and in `get_numeric_feature_columns` (or the Phase 6 equivalent).
- **Backend (`app/`)**: The live API uses `preprocessing.py` for warning signals and `traditional_ml.py` for Phase 6 TF-IDF + optional numeric features. To use new numeric features in the backend, the thesis pipeline must export them (same column set and order) and the backend must compute the same features at inference time (or load a feature spec and run the same functions).

## Detailed Findings

### 1. Where features are defined and used

| Location | Role |
|----------|------|
| **`thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb`** | Dataset 2 pipeline: linguistic features (sentence stats, punctuation, sentiment, second-person, scam phrases, **and** a grammar function that is not yet wired in), structural features, TF-IDF, and combined train/val/test matrices. |
| **`thesis-scam-job-post/ipynb/phase6_scam_detection.ipynb`** | Merged (D1+D2) pipeline: `build_structural_features` (length, word count, has_salary, has_company_profile, has_skills_desc, employment_type_enc, dataset_source_encoded) + TF-IDF; no linguistic feature columns. |
| **`app/preprocessing.py`** | Builds `combined_text`, boolean/length fields, and **warning signals** (keywords, missing fields, short description, **caps words > 5**, exclamation count > 3). Not used as numeric model features. |
| **`app/traditional_ml.py`** | Loads Phase 6 artifacts (single pipeline or tfidf + classifier). If the thesis uses TF-IDF + numeric features, the backend must supply the same numeric feature vector (same columns and order) when calling the classifier. |

So: **numeric features for the model** are created in the notebooks (Dataset 2 and/or Phase 6); the **backend** only consumes them (and must mirror feature construction if it does not load a full pipeline that does it).

### 2. ALL CAPS words

**Already implemented.**

- **Dataset 2 notebook**  
  - In **Punctuation and emphasis patterns**, `compute_punctuation_features(text)` computes:
    - `all_caps_word_ratio`: `(number of tokens that are all uppercase and length > 1) / total tokens`, using `re.findall(r"\b\S+\b", text)` and `t.isupper()`.
  - This is appended in `add_linguistic_features` as `combined_text_all_caps_word_ratio`.
  - The combined structural/linguistic list includes columns that `startswith("combined_text_")`, so `combined_text_all_caps_word_ratio` is included in the feature set used for training (e.g. `structural_linguistic_feature_columns` and thus `get_numeric_feature_columns` for Phase 4.1).

- **Backend**  
  - In **`app/preprocessing.py`**, `detect_warning_signals`:
    - Finds words matching `\b[A-Z]{3,}\b` in `combined_text`.
    - If `len(caps_words) > 5`, adds the warning “Excessive use of capital letters”.
  - This is for **human-facing warnings**, not for a numeric feature. If you want the same ALL CAPS metric as in the model (e.g. ratio), you can add a small helper that computes the same ratio as in the notebook and either attach it to `PreprocessedInput` or use it in a future numeric-feature path.

**How to “add” ALL CAPS:**  
For the **model**: the feature already exists; ensure the column is in the list that gets stacked with TF-IDF and that the same column set is used when exporting/saving Phase 4.1/Phase 6 artifacts. For the **backend**: optionally add an explicit “ALL CAPS ratio” in `preprocessing.py` (and use it in warnings or in a numeric feature vector if you add one).

### 3. Grammar score and spelling error rate

**Grammar/spelling feature exists but is not wired into the linguistic DataFrame.**

- **Dataset 2 notebook**  
  - **Grammar error detection** (LanguageTool):
    - `compute_grammar_error_features(text)` returns:
      - `grammar_error_count`: number of LanguageTool issues (text truncated to 4000 chars).
      - `grammar_error_ratio`: errors per sentence (using `compute_sentence_statistics` for sentence count).
      - `has_grammar_errors`: 1.0 if any errors, else 0.0.
    - LanguageTool is lazy-loaded via `_get_grammar_tool()`; if the library is missing or fails, the function returns zeros.
  - **Gap:** In `add_linguistic_features`, the per-row loop does **not** call `compute_grammar_error_features` and does **not** add these three columns to the dataframe. So the feature is implemented but unused in training.

**How to add grammar score and spelling error rate:**

1. **Wire grammar (and optional spelling) into Dataset 2**
   - In `add_linguistic_features`, for each row:
     - Call `gf = compute_grammar_error_features(text_value)`.
     - Append to lists: e.g. `grammar_error_counts.append(gf["grammar_error_count"])`, same for `grammar_error_ratio` and `has_grammar_errors`.
   - After the loop, add three columns to the dataframe, e.g.:
     - `dataframe[f"{text_column}_grammar_error_count"] = grammar_error_counts`
     - `dataframe[f"{text_column}_grammar_error_ratio"] = grammar_error_ratios`
     - `dataframe[f"{text_column}_has_grammar_errors"] = has_grammar_errors_list`
   - These columns will be picked up automatically for training if they are numeric and not in `EXCLUDE_COLS`, and if the structural/linguistic column list includes them (e.g. via `startswith("combined_text_")`).
2. **Spelling error rate**
   - Option A: Use the same LanguageTool output. LanguageTool reports both grammar and spelling; `grammar_error_count` (or a dedicated spelling count if you filter by rule category) can be turned into a rate, e.g. `spelling_error_ratio = spelling_errors / max(1, word_count)`.
   - Option B: Add a separate function (e.g. `compute_spelling_error_features`) that uses a spell-checker (e.g. PySpellChecker, or LanguageTool filtered to spelling rules) and returns a count and ratio; then add those columns in `add_linguistic_features` the same way.
3. **Grammar “score”**
   - You can define a “grammar score” as a single number, e.g. `1.0 - min(1.0, grammar_error_ratio / k)` for some scale factor `k`, or `1.0 - min(1.0, grammar_error_count / max(1, word_count))`. Add it as an extra column in the same loop so it becomes part of the numeric feature set.

### 4. Unusual punctuation frequency

**Punctuation features exist; “unusual” can be extended.**

- **Dataset 2 notebook**
  - `compute_punctuation_features(text)` already returns:
    - `exclamation_count`, `question_mark_count`, `ellipsis_count`
    - `maximum_exclamation_run`, `maximum_question_mark_run`
    - `all_caps_word_ratio`
  - These are added in `add_linguistic_features` with names like `combined_text_exclamation_count`, etc., and are included in the feature set via `combined_text_*` and the numeric column selection.

**How to add unusual punctuation frequency:**

1. **Define “unusual”** (e.g. multiple consecutive punctuation, or characters like `!!!`, `??`, `…`, or symbols not typical in normal job text).
2. **Extend `compute_punctuation_features`** (or add a small helper):
   - Count “unusual” punctuation events (e.g. runs of `!` or `?` of length ≥ 2, or presence of certain symbols).
   - Return e.g. `unusual_punctuation_count` and/or `unusual_punctuation_frequency` (e.g. count per token or per character).
3. **Add columns in `add_linguistic_features`** for these new keys, same pattern as existing punctuation features.
4. They will then be part of the numeric feature set as long as they are not in `EXCLUDE_COLS` and are included in the structural/linguistic list (e.g. under `combined_text_*`).

### 5. Backend (app) and Phase 6

- **Phase 6** (`phase6_scam_detection.ipynb`) currently uses only **structural** features from `build_structural_features` (no linguistic columns). So:
  - If you want spelling/grammar/unusual punctuation/ALL CAPS in the **merged** model, you must either:
    - Add these features in the merged pipeline (e.g. compute them in Phase 6 from `combined_text` and add to the feature matrix), or
    - Use Dataset-2-only artifacts that already include linguistic features (e.g. Phase 4.1 with the same numeric columns).
- **`app/traditional_ml.py`** expects the number of features to match the classifier’s `n_features_in_`. If the thesis pipeline adds new numeric columns, the backend must:
  - Compute the same features at inference (same names and order), or
  - Load a pipeline (e.g. joblib) that does both vectorization and numeric feature construction so the backend just passes raw text (and any required metadata).

## Code References

- **ALL CAPS (notebook):** `thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb` — `compute_punctuation_features` (all_caps_word_ratio), `add_linguistic_features` (combined_text_all_caps_word_ratio).
- **ALL CAPS (backend):** `app/preprocessing.py` — `detect_warning_signals`: `caps_words = re.findall(r"\b[A-Z]{3,}\b", combined_text)`, signal if `len(caps_words) > 5`.
- **Grammar (notebook):** `thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb` — `compute_grammar_error_features` (LanguageTool), `_get_grammar_tool`; **not** called in `add_linguistic_features`.
- **Punctuation (notebook):** Same notebook — `compute_punctuation_features` (exclamation, question, ellipsis, runs, all_caps_word_ratio); all wired in `add_linguistic_features`.
- **Structural/linguistic columns:** Same notebook — `structural_linguistic_feature_columns` (columns starting with `combined_text_` plus flags and length/count columns); `get_numeric_feature_columns` + `EXCLUDE_COLS` for Phase 4.1.
- **Phase 6 structural only:** `thesis-scam-job-post/ipynb/phase6_scam_detection.ipynb` — `build_structural_features` (no linguistic features).
- **Backend feature contract:** `app/traditional_ml.py` — `Phase6MergedPredictor._predict_vectorizer_classifier` uses `n_features_in_` and pads or errors on mismatch.

## Architecture Documentation

- **Dataset 2 pipeline:** Load → preprocess → **add_linguistic_features** (sentence, punctuation, sentiment, second-person, scam phrase; grammar present as function but not used) → **add_structural_features_to_dataframe** → TF-IDF on text column → combine TF-IDF + structural/linguistic numeric columns → train (e.g. Phase 4.1).
- **Phase 6 pipeline:** Harmonize D1/D2 → **build_structural_features** (no linguistic) → TF-IDF on `combined_text` → scale structural → hstack TF-IDF + structural → train.
- **Backend:** `preprocess_job_post` → `PreprocessedInput` (warning_signals use caps and exclamation heuristics); Phase 6 predictor expects a fixed feature dimension (TF-IDF + optional numeric); if numeric features are used, the app must produce them in the same order as training.

## Historical Context (from cursor/project/notes/)

- **Implementation Plan 1** (`cursor/project/notes/Implementation Plan 1.md`) describes Phase 2 feature engineering: grammar/language quality (grammar error detection, sentence structure, punctuation patterns, excessive exclamation, all caps), and sentiment/tone. The Dataset 2 notebook implements most of this; grammar is implemented but not yet connected in `add_linguistic_features`.

## Related Research

- `cursor/project/research/2025-03-09-hybrid-model-and-checkpoint.md`
- `cursor/project/research/2025-03-08-thesis-trained-model-usage.md`
- `cursor/project/research/2025-03-08-thesis-scam-job-post-models-integration.md`

## Open Questions

- None for this research scope. Implementation steps for the four feature types are described above.
