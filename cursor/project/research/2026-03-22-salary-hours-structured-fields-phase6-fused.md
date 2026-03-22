---
date: 2026-03-22T12:43:56Z
researcher: riche
git_commit: a41aea8ccf37a2dc4a249e6d66f96daf13046f52
branch: main
repository: job-sentry-backend
topic: "Adding salary and hours fields — compatibility with the current phase6 fused model"
tags: [research, codebase, phase6, fused, schemas, JobPostInput, HybridFusedClassifier]
status: complete
last_updated: 2026-03-22
last_updated_by: riche
metadata_note: "hack/spec_metadata.sh was not present in the repository; git hash, branch, and timestamps were gathered via shell commands."
---

# Research: Adding salary and hours fields — compatibility with the current phase6 fused model

**Date**: 2026-03-22T12:43:56Z  
**Researcher**: riche  
**Git Commit**: `a41aea8ccf37a2dc4a249e6d66f96daf13046f52`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

Can we add fields for salary and hours for the current (phase6 fused) model?

## Summary

The **deployed phase6 fused model** is a **text-only** classifier: it takes a single string per example, tokenizes it for **DistilBERT** and a word-level **BiLSTM**, and fuses those representations. There are **no** separate tensor inputs for structured fields such as salary amount or hours. Artifact metadata (`fused_meta.json`) describes only text lengths, vocabulary size, and MLP dimensions—**no** salary or hours feature shapes.

You **can** add `salary` and `hours` (or similar) to the **HTTP API** as optional fields. For them to influence **this** trained model, they must appear **inside the string** passed to inference (for example by extending `JobPostInput.combined_text()` so those values are concatenated into the text). The model does **not** consume parallel numeric or categorical inputs alongside text without **changing the network and retraining**.

Processed training data used for the fused notebook includes columns such as `salary_range` and `employment_type` on merged splits, but the **phase6 hybrid fused training path** loads **`combined_text`** lists only; a dedicated **`hours`** column was not identified in the `merged_train.csv` header (hours may appear only inside free text).

## Detailed Findings

### HTTP API input (what clients send)

`JobPostInput` accepts optional `text` or structured text fields `job_title`, `job_desc`, `skills_desc`, and `company_profile`. There is **no** `salary`, `salary_range`, or `hours` field today. `combined_text()` returns either `text` or a single string joined from the structured fields ([`app/schemas.py`](../../app/schemas.py)).

### Prediction path (what reaches the model)

`/predict` builds a list of strings via `combined_text()` and passes them to `FusedScamPredictor.predict_proba(texts)` ([`app/main.py`](../../app/main.py)). The predictor tokenizes those strings for DistilBERT and for the LSTM branch; no other features are passed ([`app/fused_predictor.py`](../../app/fused_predictor.py)).

### Model architecture (what the checkpoint implements)

`HybridFusedClassifier.forward` accepts three tensors: `input_ids`, `attention_mask` (DistilBERT), and `lstm_ids` (word IDs). The classifier head operates on the concatenation of pooled BERT output and BiLSTM final hidden states—**no** additional structured feature vector ([`app/hybrid_fused_model.py`](../../app/hybrid_fused_model.py)).

### Saved artifacts (what training recorded)

`artifacts/models/phase6_fused/fused_meta.json` lists hyperparameters such as `max_len_bert`, `max_len_bilstm`, `vocab_size`, `embed_dim`, `lstm_hidden`, `fusion_hidden`, and `threshold`. There is **no** metadata for auxiliary salary or hours inputs.

### Training notebook alignment

`artifacts/ipynb/phase6_hybrid_fused.ipynb` builds `train_texts` / `val_texts` / `test_texts` from the **`combined_text`** column only; `HybridFusedDataset` is defined over those text lists. That matches production inference, which is also **text-only**.

### Processed dataset columns (context for salary vs. hours)

`artifacts/data/processed/merged_train.csv` columns include `salary_range`, `employment_type`, `fraudulent`, `combined_text`, `dataset_source`, etc. There is **no** column named `hours` in the header; hour-related wording may appear inside `job_desc` or other text fields.

## Code References

- `app/schemas.py:10-33` — `JobPostInput` fields and `combined_text()` behavior.
- `app/main.py:94-122` — `/predict` collects `combined_text()` outputs and calls `predictor.predict_proba(texts)`.
- `app/fused_predictor.py:62-92` — `predict_proba` / `_predict_proba_batch`: tokenizer + `texts_to_lstm_batch` only.
- `app/hybrid_fused_model.py:42-119` — `HybridFusedClassifier` inputs and forward pass.
- `app/fused_loader.py:108-136` — loads `fused_meta.json` and constructs `HybridFusedClassifier` from meta fields only.
- `artifacts/models/phase6_fused/fused_meta.json` — artifact hyperparameters (no structured salary/hours slots).
- `artifacts/ipynb/phase6_hybrid_fused.ipynb` — training uses `combined_text` lists for `HybridFusedDataset`.

## Architecture Documentation

Production phase6 serving is a **single fused text model**: DistilBERT + BiLSTM on the **same** string, binary logits from a fused MLP. Structured columns that exist in CSV pipelines elsewhere (for example `salary_range`, `has_salary` in other notebooks) are **not** wired into this fused module or the FastAPI predictor in the application code reviewed here.

## Historical Context (from cursor/project/)

- `cursor/project/plan/2025-03-08-jobsentry-backend-implementation.md` — Older planning text describes a richer input schema (e.g. `salary_range`, `employment_type`) and structural features; the **current** fused app path implements the narrower text-only contract above.
- `cursor/project/notes/Implementation Plan 2.md` — Dataset discussion of `salary_range`, `has_salary`, and related structural features for **non–phase6-fused** pipelines.

## Related Research

- `cursor/project/research/2026-03-22-phase6-fused-production-inference-next-steps.md` — Phase6 fused inference context.
- `cursor/project/research/2025-03-09-features-spelling-grammar-punctuation-caps.md` — References merged pipelines with structural features (not the fused `HybridFusedClassifier` path).

## Open Questions

- Whether `combined_text` in the preprocessed CSVs was built **including** `salary_range` (or other fields) in the string—would require tracing the notebook or script that **wrote** `merged_*.csv` (not fully traced in this research pass).

## GitHub permalinks (committed tree)

Remote used in prior project research: `https://github.com/Retchizu/job-sentry-backend.git`

- [app/schemas.py at a41aea8](https://github.com/Retchizu/job-sentry-backend/blob/a41aea8ccf37a2dc4a249e6d66f96daf13046f52/app/schemas.py)
- [app/hybrid_fused_model.py at a41aea8](https://github.com/Retchizu/job-sentry-backend/blob/a41aea8ccf37a2dc4a249e6d66f96daf13046f52/app/hybrid_fused_model.py)

If the local branch differs from `a41aea8`, line numbers and content may differ until changes are committed and pushed.
