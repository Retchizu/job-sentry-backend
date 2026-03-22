# Dataset 2 — Linguistic Feature Set and Formulas

This document lists the linguistic feature columns produced by `add_linguistic_features` in the Dataset 2 notebook and their formulas, so that a future backend or Phase 6 pipeline can reproduce the same numeric features at inference time.

**Source**: `thesis-scam-job-post/ipynb/dataset2_scam_detection.ipynb` — `compute_punctuation_features`, `compute_grammar_error_features`, `add_linguistic_features`.

**Prefix**: All columns use the text column name (e.g. `combined_text_`). Column order below is logical; the actual order in the DataFrame is the order in which columns are assigned in `add_linguistic_features`.

---

## 1. Sentence statistics

| Column | Formula |
|--------|--------|
| `{text_column}_sentence_count` | From `compute_sentence_statistics`: number of sentences. |
| `{text_column}_average_sentence_length` | Mean token count per sentence. |
| `{text_column}_maximum_sentence_length` | Max token count over sentences. |

---

## 2. Token / character ratios

| Column | Formula |
|--------|--------|
| `{text_column}_non_alphabetic_token_ratio` | From `compute_non_alphabetic_token_ratio`: proportion of tokens that are non-alphabetic. |

---

## 3. Punctuation and emphasis

| Column | Formula |
|--------|--------|
| `{text_column}_exclamation_count` | Raw count of `!` in text. |
| `{text_column}_question_mark_count` | Raw count of `?` in text. |
| `{text_column}_ellipsis_count` | Count of substring `"..."` in text. |
| `{text_column}_maximum_exclamation_run` | Length of longest contiguous run of `!`. |
| `{text_column}_maximum_question_mark_run` | Length of longest contiguous run of `?`. |
| `{text_column}_all_caps_word_ratio` | **Tokens**: `re.findall(r"\b\S+\b", text)`. **All-caps**: tokens where `t.isupper() and len(t) > 1`. **Ratio**: `all_caps_count / len(tokens)` (0 if no tokens). |
| `{text_column}_unusual_punctuation_count` | Number of “unusual” events: (1) runs of `!` of length ≥ 2, (2) runs of `?` of length ≥ 2, (3) count of `"..."`. Sum of those three. |
| `{text_column}_unusual_punctuation_frequency` | `unusual_punctuation_count / max(1, len(tokens))` (per-token frequency). |

---

## 4. Grammar and spelling (LanguageTool)

Text is truncated to 4000 characters before calling LanguageTool. If LanguageTool is unavailable or errors, all grammar/spelling features are 0 (and `grammar_score` = 1.0).

| Column | Formula |
|--------|--------|
| `{text_column}_grammar_error_count` | Number of issues returned by `tool.check(truncated)`. |
| `{text_column}_grammar_error_ratio` | `grammar_error_count / max(1, sentence_count)`. |
| `{text_column}_has_grammar_errors` | 1.0 if `grammar_error_count > 0`, else 0.0. |
| `{text_column}_spelling_error_ratio` | Count of LanguageTool matches with category `SPELLING`; **ratio** = `spelling_count / max(1, word_count)` where `word_count = len(re.findall(r"\b\S+\b", truncated))`. |
| `{text_column}_grammar_score` | `1.0 - min(1.0, grammar_error_ratio / 2.0)` (scale factor k = 2.0). |

---

## 5. Sentiment (VADER)

| Column | Formula |
|--------|--------|
| `{text_column}_sentiment_negative` | VADER `neg`. |
| `{text_column}_sentiment_neutral` | VADER `neu`. |
| `{text_column}_sentiment_positive` | VADER `pos`. |
| `{text_column}_sentiment_compound` | VADER `compound`. |

---

## 6. Second-person and scam phrases

| Column | Formula |
|--------|--------|
| `{text_column}_second_person_pronoun_count` | Count of tokens in `{"you", "your", "yours", "yourself", "yourselves"}`. |
| `{text_column}_second_person_pronoun_ratio` | `count / len(tokens)`. |
| `{text_column}_scam_phrase_total_count` | Sum of phrase occurrences (from fixed scam phrase list). |
| `{text_column}_scam_phrase_any_indicator` | 1.0 if total count > 0, else 0.0. |

---

## Feature set for training (Phase 4.1 / structural+linguistic)

- **Selection**: `structural_linguistic_feature_columns` = columns where `col.startswith("combined_text_")` (excluding `combined_text`, `combined_text_processed`), or `col.endswith("_flag")`, or col in the explicit structural list (lengths, word counts, has_salary, employment_type_encoded, location, industry, etc.).
- **Scaling**: StandardScaler (or MinMaxScaler in Phase 4.1 numeric path) on the numeric feature matrix before stacking with TF-IDF.
- **Order**: Column order is the order in `train_structural_dataframe.columns` filtered by the above; when exporting an artifact, document or save the column list so the backend can mirror it.

---

## Backend alignment (optional)

- **ALL CAPS ratio**: `app/preprocessing.compute_all_caps_ratio(combined_text)` matches the notebook formula for `all_caps_word_ratio` (tokens via `re.findall(r"\b\S+\b", text)`, all-caps = `isupper() and len(t) > 1`).
- Grammar/spelling and unusual punctuation are **not** computed in the backend in this iteration; adding them would require LanguageTool (or equivalent) and the same formulas above.
