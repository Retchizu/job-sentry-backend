---
date: 2025-03-08T00:00:00+08:00
researcher: Cursor Agent
git_commit: d0b7c79ca93391a2e76596fe788befa693db47c0
branch: main
repository: job-sentry-backend
topic: "Putting thesis-scam-job-post trained models into work in JobSentry backend"
tags: [research, codebase, models, thesis-scam-job-post, DistilBERT, model-artifacts]
status: complete
last_updated: 2025-03-08
last_updated_by: Cursor Agent
---

# Research: Putting thesis-scam-job-post Models Into Work

**Date**: 2025-03-08  
**Researcher**: Cursor Agent  
**Git Commit**: d0b7c79ca93391a2e76596fe788befa693db47c0  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

How does the JobSentry backend integrate with trained models from thesis-scam-job-post/models, and what is required to put those models into work?

## Summary

The backend is already wired to load a fine-tuned DistilBERT checkpoint from **thesis-scam-job-post/models**. It expects a single artifact file at a configurable path: **`model.safetensors`** inside a checkpoint directory (default: `thesis-scam-job-post/models/distilbert_run/checkpoint-1566`). The tokenizer is loaded from Hugging Face (`distilbert-base-uncased`), not from the thesis repo. To put the thesis model into work, the thesis project must expose that path (or a copy) containing `model.safetensors`; the backend then loads it at startup, remaps TensorFlow-style LayerNorm keys if present, and serves predictions via `POST /predict` and `POST /batch-predict`. No document in this repo describes a “summary of trained models” from the thesis; any such summary would inform which checkpoint path to use (e.g. a different checkpoint number or run).

## Detailed Findings

### 1. Model artifact path and configuration

- **Default path**  
  [`app/config.py:6`](app/config.py): `model_artifact_path` defaults to  
  `Path("thesis-scam-job-post/models/distilbert_run/checkpoint-1566")`.  
  So the backend assumes a directory, typically relative to the process working directory (e.g. repo root or a parent directory that contains both `job-sentry-backend` and `thesis-scam-job-post`).

- **Override via env**  
  Settings use `JOBSENTRY_` prefix and `.env` ([`app/config.py:12`](app/config.py)). The env var is **`JOBSENTRY_MODEL_ARTIFACT_PATH`**.  
  [`.env.example:2`](.env.example) documents:  
  `JOBSENTRY_MODEL_ARTIFACT_PATH=thesis-scam-job-post/models/distilbert_run/checkpoint-1566`  
  and notes that this path must point to the checkpoint directory that **contains `model.safetensors`**.

- **Single file used**  
  Only one file is read from that directory: **`model.safetensors`**. The loader does not read `config.json`, `tokenizer.json`, or any other file from the thesis path; the tokenizer and config come from `model_name` (see below).

### 2. Expected layout under thesis-scam-job-post/models

The code only requires:

- **Directory**: `{model_artifact_path}` → e.g. `thesis-scam-job-post/models/distilbert_run/checkpoint-1566`
- **File**: `model.safetensors` inside that directory

Full path used at runtime:  
`{model_artifact_path}/model.safetensors`  
e.g. `thesis-scam-job-post/models/distilbert_run/checkpoint-1566/model.safetensors`.

No other files under `thesis-scam-job-post/models` are read by the current backend code.

### 3. How the backend loads and uses the model

- **Startup**  
  [`app/main.py:22-31`](app/main.py): FastAPI lifespan calls `load_model()` and stores the result in `app.state.model`. If loading fails, `app.state.model` is set to `None` and the app still runs; `/health` then reports `model_loaded: false` and predict endpoints return 503.

- **Loader** ([`app/model.py:55-93`](app/model.py))  
  - Resolves `artifact_path = Path(settings.model_artifact_path)`.  
  - Loads tokenizer from **`settings.model_name`** (default `distilbert-base-uncased`) via Hugging Face, not from the thesis directory.  
  - Builds path `safetensors_file = artifact_path / "model.safetensors"`.  
  - If `model.safetensors` exists: loads state dict with `safetensors.torch.load_file`, remaps keys `.gamma` → `.weight` and `.beta` → `.bias` (for TF-style LayerNorm), loads into `DistilBertForSequenceClassification(config)` with `num_labels=2`, then moves model to device and sets eval mode.  
  - If `model.safetensors` is missing: logs a warning and uses the same architecture with **untrained** weights (predictions not meaningful).

- **Prediction flow**  
  [`app/services/prediction.py`](app/services/prediction.py): `predict_single` / `predict_batch` preprocess job post fields into `combined_text` (via `app/preprocessing.py`), then call `model.predict(combined_text)` or `model.predict_batch(texts)`. The model returns `prediction`, `confidence`, and `scam_probability`; the service adds `warning_signals` from preprocessing and returns a `PredictResponse`.

### 4. Config and model name

- **`app/config.py`** also defines:  
  `model_name` (default `distilbert-base-uncased`), `max_sequence_length` (512), `max_batch_size` (50), `confidence_threshold` (0.5).  
  These are used by the model and tokenizer; the thesis artifact path does not override `model_name`.

### 5. What the thesis project must provide

To put the thesis model into work:

1. **Checkpoint directory** at the path configured by `JOBSENTRY_MODEL_ARTIFACT_PATH` (or the default).  
2. **File**: `model.safetensors` in that directory, containing the fine-tuned DistilBERT weights (2-label classification).  
3. If the checkpoint was saved with TF-style LayerNorm keys (`gamma`/`beta`), the backend remaps them automatically; no change in the thesis export is required for that.

The tokenizer is fixed to `distilbert-base-uncased`; the thesis training must have used the same tokenizer so that tokenization at inference matches training.

## Code References

- `app/config.py:6` – default `model_artifact_path` and `model_name`
- `app/config.py:12` – env prefix and env_file for overrides
- `app/model.py:55-93` – `load_model()`: path resolution, safetensors load, LayerNorm remap, fallback
- `app/model.py:66` – `safetensors_file = artifact_path / "model.safetensors"`
- `app/main.py:22-31` – lifespan: load model into `app.state.model`
- `app/services/prediction.py` – use of `ScamDetectionModel` for single and batch predict
- `.env.example:1-2` – documented `JOBSENTRY_MODEL_ARTIFACT_PATH` and requirement for `model.safetensors`

## Architecture Documentation

- **Single production model**: The backend serves one model type only — DistilBERT for sequence classification (binary: scam vs legitimate). BiLSTM and sklearn models from the thesis are not loaded; the implementation summary states that only DistilBERT had usable weights and a standard tokenizer.
- **Artifact path**: One directory path points to one checkpoint; that directory must contain `model.safetensors`. No multi-checkpoint or ensemble loading is implemented.
- **Preprocessing**: Done in this repo (`app/preprocessing.py`); combined text and warning signals are built here to match the thesis pipeline (job_title + job_desc + skills_desc + company_profile), not loaded from the thesis repo.

## Historical Context (from cursor/project/notes/)

- **`cursor/project/notes/Implementation Plan 1.md`** – Describes scam job post detection pipeline, DistilBERT/Bi-LSTM/traditional ML, and saved model formats `.pkl`, `.h5`, `.pth`; folder layout includes `models/` for saved model files. No thesis-scam-job-post paths or `.safetensors`/checkpoint names.
- **`cursor/project/notes/Implementation Plan 2.md`** – Second dataset pipeline, DistilBERT fine-tuning and tokenizer; trained models saved as `.pkl`, `.h5`, `.pth`. No thesis artifact paths or checkpoint-1566.

No document in **cursor/project/notes** contains a “summary of trained models” from the thesis project (e.g. which run/checkpoint to use, or metrics per checkpoint). That information, if available in the thesis repo or elsewhere, would guide the choice of `JOBSENTRY_MODEL_ARTIFACT_PATH` (e.g. a different checkpoint or run under `thesis-scam-job-post/models`).

## Related Research

- [2025-03-08-jobsentry-backend-sequence-plan](cursor/project/research/2025-03-08-jobsentry-backend-sequence-plan.md) – Backend sequence plan and thesis model consumption.
- Implementation summary: `cursor/project/implementation/2025-03-08-NA-jobsentry-backend-implementation.md` – DistilBERT-only choice, LayerNorm remapping, default path, and manual verification steps.

## Open Questions

- **Trained models summary**: No “summary of trained models” document was found in this repository. If such a summary exists (e.g. in the thesis-scam-job-post repo), it could be used to confirm the best checkpoint (e.g. `checkpoint-1566` vs others) and to document metrics for the deployed model.
- **Path at runtime**: Default path is relative (`thesis-scam-job-post/models/...`). For deployment, either the thesis repo (or a copy of the checkpoint) must be present at that relative path, or `JOBSENTRY_MODEL_ARTIFACT_PATH` must be set to an absolute path where `model.safetensors` lives.
