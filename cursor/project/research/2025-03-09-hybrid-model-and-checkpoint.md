---
date: 2025-03-09T00:00:00+08:00
researcher: Cursor Agent
git_commit: d0b7c79ca93391a2e76596fe788befa693db47c0
branch: main
repository: job-sentry-backend
topic: "Hybrid model support (Implementation Plan 1) and latest checkpoint usage"
tags: [research, codebase, hybrid, ensemble, checkpoint, DistilBERT, Implementation Plan 1]
status: complete
last_updated: 2025-03-09
last_updated_by: Cursor Agent
---

# Research: Hybrid Model Support and Latest Checkpoint Usage

**Date**: 2025-03-09  
**Researcher**: Cursor Agent  
**Git Commit**: d0b7c79ca93391a2e76596fe788befa693db47c0  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

1. What would it take to make the backend support the hybrid model described in Implementation Plan 1?  
2. Is the backend using the latest (final) training checkpoint?

## Summary

- **Hybrid model**: Implementation Plan 1 §3.3 defines a hybrid as an ensemble of traditional ML (Random Forest, Logistic Regression) and deep learning (Bi-LSTM, DistilBERT) with voting/stacking/weighted average. The backend currently serves **only** the thesis DistilBERT model. Traditional ML (`phase6_merged` joblib) and Bi-LSTM (Keras) are explicitly noted as not supported in config. No ensemble or combination logic exists in the codebase.

- **Checkpoint**: The codebase documents five DistilBERT checkpoints for `distilbert_run`: **783, 1566, 2349, 3608, 5412**. The default in code and `.env.example` is **checkpoint-1566**, not the numerically last checkpoint (5412). There is no “summary of trained models” in this repo that identifies which checkpoint is “best” or “final”; that information would live in the thesis-scam-job-post project. The current `.env` points at `thesis-scam-job-post/models/distilbert_run/` (run directory only), which typically does not contain `model.safetensors` (artifacts are under `checkpoint-XXX/`), so the app may be falling back to the base untrained model unless a specific checkpoint path is set.

## Detailed Findings

### 1. Hybrid model (Implementation Plan 1)

**Plan definition** ([`cursor/project/notes/Implementation Plan 1.md`](cursor/project/notes/Implementation%20Plan%201.md)):

- §3.1: Traditional ML — Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, XGBoost; trained on TF-IDF/BoW.
- §3.2: Deep learning — Bi-directional LSTM and DistilBERT.
- §3.3 **Hybrid**: Combine traditional ML (Random Forest, Logistic Regression) with deep learning (Bi-LSTM, DistilBERT); voting (hard/soft), stacking, weighted average of predictions.

**Backend today**:

- A single model is loaded: DistilBERT for sequence classification ([`app/model.py`](app/model.py)). No ensemble, no second model, no voting or stacking.
- Config ([`app/config.py:4-8`](app/config.py)) lists supported paths (DistilBERT checkpoints and phase32/phase42) and states: **"Not supported: bilstm_checkpoints (Keras), phase6_merged (joblib TF-IDF+LR/RF/XGB)."**
- Prediction flow ([`app/services/prediction.py`](app/services/prediction.py)) preprocesses input and calls `model.predict` / `model.predict_batch` on the single `ScamDetectionModel` (DistilBERT only).

So the backend does **not** support the Plan 1 hybrid. Supporting it would require: loading and using the traditional ML artifact (e.g. `phase6_merged` joblib), optionally Bi-LSTM (Keras), running the same feature pipeline (e.g. TF-IDF) for the traditional branch, and implementing the same combination strategy (voting/stacking/weighted average) as in the thesis notebook.

### 2. Checkpoint usage and “latest”

**Documented checkpoints** ([`app/config.py:5`](app/config.py)):

- `distilbert_run`: `thesis-scam-job-post/models/distilbert_run/checkpoint-{783|1566|2349|3608|5412}`

**Default in code** ([`app/config.py:12`](app/config.py)):

- `model_artifact_path` defaults to `Path("thesis-scam-job-post/models/distilbert_run/checkpoint-1566")`.

**Default in `.env.example`** ([`.env.example:2`](.env.example)):

- `JOBSENTRY_MODEL_ARTIFACT_PATH=thesis-scam-job-post/models/distilbert_run/checkpoint-1566`

So by default the backend is configured for **checkpoint-1566**, not the numerically last checkpoint **5412**. The repo does not define “final” or “best”; that would come from the thesis project (e.g. validation metrics or last step).

**Runtime path**:

- The loader expects `model.safetensors` inside the directory given by `JOBSENTRY_MODEL_ARTIFACT_PATH` ([`app/model.py:71-72`](app/model.py)). If the env is set to the run directory only (e.g. `thesis-scam-job-post/models/distilbert_run/`), that directory usually does not contain `model.safetensors` (files are under `checkpoint-XXX/`). In that case the app loads the base (untrained) DistilBERT and logs a warning. So the “latest” or any specific checkpoint is only used when the path points to a concrete checkpoint directory that contains `model.safetensors`.

### 3. Where the model is loaded and used

- **Startup**: [`app/main.py:22-31`](app/main.py) — lifespan calls `load_model()` and sets `app.state.model`.
- **Loader**: [`app/model.py:55-105`](app/model.py) — `load_model()` resolves `settings.model_artifact_path`, looks for `artifact_path / "model.safetensors"`, loads fine-tuned weights with LayerNorm key remap if present, otherwise falls back to base DistilBERT.
- **Prediction**: [`app/services/prediction.py`](app/services/prediction.py) — builds combined text via `preprocess_job_post`, then calls `model.predict(combined_text)` or `model.predict_batch(texts)`; no other model or ensemble is invoked.

## Code References

- `app/config.py:4-8` — Supported paths comment; “Not supported: bilstm_checkpoints, phase6_merged”
- `app/config.py:12` — Default `model_artifact_path` = checkpoint-1566
- `app/model.py:55-105` — `load_model()`: single DistilBERT, safetensors path, LayerNorm remap, fallback
- `app/model.py:71-72` — `safetensors_file = artifact_path / "model.safetensors"`
- `app/main.py:22-31` — Lifespan: load model into `app.state.model`
- `app/services/prediction.py` — Single-model predict; no ensemble

## Architecture Documentation

- **Model**: One production model type — DistilBERT for binary (scam/legitimate) sequence classification. No traditional ML, no Bi-LSTM, no ensemble.
- **Checkpoint**: One directory path; that directory must contain `model.safetensors`. Default is checkpoint-1566; checkpoint-5412 is the numerically last of the five listed but is not the default and is not documented as “final” or “best” in this repo.
- **Plan 1 hybrid**: Would require adding loaders and inference for phase6_merged (and optionally Bi-LSTM), shared feature pipeline, and combination logic (voting/stacking) as in the thesis.

## Historical Context (from cursor/project/notes/)

- **`cursor/project/notes/Implementation Plan 1.md`** — Defines hybrid (Phase 3.3): ensemble of traditional ML + Bi-LSTM + DistilBERT with voting/stacking/weighted average; deliverables include trained model files in pickle/ONNX/H5 and “Final hybrid model” in notebook Section 7.
- **`cursor/project/plan/2025-03-08-thesis-trained-model-remaining-work.md`** — Describes default checkpoint as checkpoint-1566; no summary of trained models in repo; thesis project is the source for which checkpoint to use.
- **`cursor/project/research/2025-03-08-thesis-scam-job-post-models-integration.md`** — Backend loads a single DistilBERT checkpoint; no multi-checkpoint or ensemble loading; “No document in this repo describes a summary of trained models from the thesis.”

## Related Research

- [2025-03-08-thesis-trained-model-usage](cursor/project/research/2025-03-08-thesis-trained-model-usage.md) — How the thesis DistilBERT is loaded and when it is used at runtime.
- [2025-03-08-thesis-scam-job-post-models-integration](cursor/project/research/2025-03-08-thesis-scam-job-post-models-integration.md) — Integration with thesis-scam-job-post models; single checkpoint, no ensemble.

## Open Questions

- Whether the thesis project defines a “best” or “final” checkpoint (e.g. 5412 as last step, or 1566 by validation metric); that would inform updating the default or documentation.
- Exact format and path of `phase6_merged` (and Bi-LSTM artifacts) in the thesis repo, and the exact combination method (voting/stacking/weights) used in the notebook, if hybrid support is to be implemented.
