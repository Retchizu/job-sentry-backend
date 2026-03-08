---
date: 2025-03-08T00:00:00+08:00
researcher: Cursor Agent
git_commit: d0b7c79ca93391a2e76596fe788befa693db47c0
branch: main
repository: job-sentry-backend
topic: "Are we using our trained model from thesis-scam-job-post?"
tags: [research, codebase, thesis-scam-job-post, DistilBERT, model-artifacts, load_model]
status: complete
last_updated: 2025-03-08
last_updated_by: Cursor Agent
---

# Research: Are We Using Our Trained Model from thesis-scam-job-post?

**Date**: 2025-03-08  
**Researcher**: Cursor Agent  
**Git Commit**: d0b7c79ca93391a2e76596fe788befa693db47c0  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

Are we using our trained model from thesis-scam-job-post?

## Summary

**Yes, by design.** The backend is built to load and use the thesis-trained DistilBERT. At startup it loads weights from a single artifact: **`model.safetensors`** inside the checkpoint directory **`thesis-scam-job-post/models/distilbert_run/checkpoint-1566`** (configurable via `JOBSENTRY_MODEL_ARTIFACT_PATH`). If that file exists, the app uses the fine-tuned model; if it does not exist, it falls back to the base (untrained) DistilBERT and logs a warning. This repository does not contain `model.safetensors`; the path is relative and expects the thesis project (or a copy of the checkpoint) to be available at that path when the app runs.

## Detailed Findings

### 1. Configuration: thesis checkpoint as default

- **Default path** ([`app/config.py:6`](app/config.py)):  
  `model_artifact_path` defaults to  
  `Path("thesis-scam-job-post/models/distilbert_run/checkpoint-1566")`.

- **Override** ([`app/config.py:12`](app/config.py)):  
  Env var `JOBSENTRY_MODEL_ARTIFACT_PATH` (documented in [`.env.example:2`](.env.example)) can point to another directory that **contains `model.safetensors`**.

So the codebase is explicitly configured to use the thesis-trained model from that checkpoint path.

### 2. How the trained model is loaded

- **Startup** ([`app/main.py:22-31`](app/main.py)):  
  FastAPI lifespan calls `load_model()` and stores the result in `app.state.model`.  
  If loading fails, `app.state.model` is set to `None` and predict endpoints return 503.

- **Loader** ([`app/model.py:55-93`](app/model.py)):  
  - Resolves `artifact_path` from `settings.model_artifact_path`.  
  - Builds `safetensors_file = artifact_path / "model.safetensors"`.  
  - **If `model.safetensors` exists**: loads state dict with `safetensors.torch.load_file`, remaps TF-style LayerNorm keys (`.gamma` → `.weight`, `.beta` → `.bias`), loads into `DistilBertForSequenceClassification(config)` with `num_labels=2`, and uses that for inference.  
  - **If `model.safetensors` is missing**: logs a warning and uses the same architecture with **untrained** weights (predictions not meaningful).

So **at runtime**, the thesis-trained model is used only when the artifact file is present at the configured path.

### 3. Artifact presence in this repo

- A search for `model.safetensors` under `job-sentry-backend` returns **no files**.  
- The default path is **relative** (`thesis-scam-job-post/models/...`). For the trained model to be used, either:  
  - The thesis project (or a copy of the checkpoint) must exist at that relative path (e.g. sibling directory of `job-sentry-backend`), or  
  - `JOBSENTRY_MODEL_ARTIFACT_PATH` must be set to an absolute path where `model.safetensors` lives.

### 4. Tokenizer and architecture

- Tokenizer is loaded from Hugging Face `distilbert-base-uncased` ([`app/model.py:60`](app/model.py), `settings.model_name`), not from the thesis repo.  
- The thesis training must have used the same tokenizer so that inference matches training.  
- Only the **DistilBERT** thesis model is integrated; BiLSTM and sklearn models from the thesis are not loaded (see [2025-03-08-thesis-scam-job-post-models-integration.md](cursor/project/research/2025-03-08-thesis-scam-job-post-models-integration.md)).

## Code References

- `app/config.py:6` – default `model_artifact_path` pointing to thesis checkpoint
- `app/config.py:12` – env prefix and `JOBSENTRY_MODEL_ARTIFACT_PATH`
- `app/model.py:55-93` – `load_model()`: path resolution, `model.safetensors` load, LayerNorm remap, fallback to untrained
- `app/model.py:66` – `safetensors_file = artifact_path / "model.safetensors"`
- `app/main.py:22-31` – lifespan: load model into `app.state.model`
- `.env.example:1-2` – documented path and requirement for `model.safetensors`

## Answer to the question

| Aspect | Answer |
|--------|--------|
| **Designed to use thesis model?** | Yes. Default path is thesis checkpoint; code loads fine-tuned weights from `model.safetensors`. |
| **Using it at runtime?** | Only if `model.safetensors` exists at the configured path when the app starts. |
| **In this repo?** | No `model.safetensors` in job-sentry-backend; path expects thesis project or env override. |

So: **we are set up to use the trained model from thesis-scam-job-post; we are actually using it only when that checkpoint (with `model.safetensors`) is available at the configured path.**

## Related Research

- [2025-03-08-thesis-scam-job-post-models-integration](cursor/project/research/2025-03-08-thesis-scam-job-post-models-integration.md) – How the thesis models are integrated and what the thesis project must provide.
