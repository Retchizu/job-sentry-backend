---

## date: 2025-03-08T00:00:00+08:00
researcher: Cursor Agent
git_commit: N/A
branch: N/A
repository: job-sentry-backend
topic: "JobSentry backend sequence plan using implementation plans and thesis model"
tags: [research, codebase, backend, JobSentry, FastAPI, implementation-plan]
status: complete
last_updated: 2025-03-08
last_updated_by: Cursor Agent

# Research: JobSentry Backend Sequence Plan

**Date**: 2025-03-08  
**Researcher**: Cursor Agent  
**Git Commit**: N/A  
**Branch**: N/A  
**Repository**: job-sentry-backend  

## Research Question

Create a sequence plan for building the JobSentry backend that detects fraudulent job posts, using the existing implementation plans and the completed preprocessing/training work located in the thesis project (thesis-scam-job-post).

## Summary

- **Implementation Plan 1** defines the full pipeline (data → features → models → evaluation) and **Phase 5: System Deployment and Interface Development** specifies backend scope: model serving, prediction API (Flask/FastAPI), preprocessing pipeline, and endpoints.
- **Implementation Plan 2** applies the same pipeline to a second dataset and reinforces preprocessing/feature-engineering consistency; the backend must use the same preprocessing and feature contract as training.
- **Current backend** is a minimal FastAPI app (`app/main.py`) with `GET /`, `GET /health`, and no model or preprocessing integration.
- The **sequence plan** below orders backend work into: config and model path → preprocessing pipeline → model loading and prediction service → API endpoints (single and batch predict) → response shape (prediction, confidence, warning signals) → testing and performance (target <2 s per prediction).

## Detailed Findings

### Implementation Plan 1 – Backend Scope (Phase 5)

From `cursor/project/notes/Implementation Plan 1.md`:

**5.1 Backend Development**

- **Model serving**: Save best model(s) in production-ready format; create prediction API (Flask/FastAPI); implement preprocessing pipeline as API endpoint; add confidence score calculation.
- **API endpoints**:
  - `POST /predict` – single job post prediction
  - `POST /batch-predict` – multiple job posts
  - `GET /health` – system health check
- **Returned fields**: prediction (scam/legitimate), confidence score, warning signals.

**5.2–5.3** cover frontend and optional browser extension; they consume the same API.

**Tools**: Flask/FastAPI, Docker for containerization. Success metric: response time <2 seconds per prediction.

### Implementation Plan 2 – Preprocessing and Features

From `cursor/project/notes/Implementation Plan 2.md`:

- **Preprocessing** (Phases 1–2): Text cleaning (HTML/URL removal, normalization), combined text field (e.g. `job_title + job_desc + skills_desc + company_profile`), handling of `salary_range`, `employment_type`, and pre-encoded `location`/`industry`.
- **Feature engineering** (Phase 3): Text length, scam keywords, structural flags (`has_salary`, `has_company_profile`, etc.), TF-IDF and/or embeddings, tokenization for Bi-LSTM and DistilBERT.
- **Model outputs**: Trained models in `.pkl`, `.h5`, `.pth`; same metrics (recall >90%, accuracy >85%).

The backend must reuse the **same** preprocessing and feature pipeline used at training so that live inputs are scored consistently.

### Current Backend State

- `**app/main.py`**: FastAPI app with title "Job Sentry API", `GET /` and `GET /health`. No routes for prediction or preprocessing.
- `**requirements.txt**`: `fastapi>=0.115.0`, `uvicorn[standard]>=0.32.0`. No ML/runtime dependencies (e.g. scikit-learn, TensorFlow/PyTorch, transformers, pandas).
- **Thesis model location**: Preprocessing and trained models are in a separate project referenced as **thesis-scam-job-post**. The backend can assume a configurable path (env or config) to model artifacts and, if needed, to shared preprocessing code.

## Backend Sequence Plan

Use this order when implementing the backend. Each step assumes the thesis-scam-job-post (or equivalent) artifacts are available at a chosen path or copied into this repo.


| Step   | Task                              | Details                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**  | **Environment and configuration** | Add a config module or env vars for: model artifact path (e.g. to thesis-scam-job-post or a copied `models/`), optional feature/config paths. Keep secrets and paths out of code.                                                                                                                                                                                                                                                                       |
| **2**  | **Dependencies**                  | Extend `requirements.txt` with the libraries used at inference: e.g. `scikit-learn`, `pandas`, and either `tensorflow` or `torch` plus `transformers` depending on whether the deployed model is Bi-LSTM, DistilBERT, or ensemble. Match versions used in thesis training where possible.                                                                                                                                                               |
| **3**  | **Preprocessing pipeline**        | Implement or reuse from thesis-scam-job-post a single preprocessing pipeline that: (a) accepts raw job post text/fields (title, description, company_profile, skills_desc, salary_range, employment_type, etc.), (b) cleans and normalizes text, (c) builds the same combined text and structural features used in training (Implementation Plan 1 §2, Plan 2 §2–3). Expose this as an internal function or module used by the prediction service only. |
| **4**  | **Model loading**                 | At startup, load the chosen production model(s) from the configured path. Support the formats produced in the thesis (e.g. pickle for sklearn, `.h5`/Keras for Bi-LSTM, `.pth` or Hugging Face for DistilBERT). If the chosen model is an ensemble (Plan 1 §3.3), load all required components and implement the same voting/stacking logic used in the notebook.                                                                                       |
| **5**  | **Prediction service**            | Implement a single entry point that: (1) takes a single job post (raw text or structured fields), (2) runs the preprocessing pipeline from step 3, (3) runs model inference, (4) returns binary prediction (scam/legitimate), confidence score, and optional warning signals (e.g. "Suspicious keywords", "Missing company details"). Reuse or mirror any confidence/warning logic from the thesis code.                                                |
| **6**  | **POST /predict**                 | Add `POST /predict` that accepts one job post (JSON body). Validate input (required fields, max length). Call the prediction service and return prediction, confidence, and warning signals in a consistent JSON schema.                                                                                                                                                                                                                                |
| **7**  | **POST /batch-predict**           | Add `POST /batch-predict` that accepts a list of job posts. Run preprocessing and inference for each (or in batch if the model supports it). Return a list of results with the same shape as single predict. Consider rate limits and timeouts so that large batches stay under the <2 s target per item where possible.                                                                                                                                |
| **8**  | **GET /health**                   | Keep existing `GET /health`; optionally extend it to verify that the model is loaded and optionally that a single dummy prediction succeeds (liveness/readiness).                                                                                                                                                                                                                                                                                       |
| **9**  | **Response schema**               | Standardize response JSON: e.g. `prediction` (scam                                                                                                                                                                                                                                                                                                                                                                                                      |
| **10** | **Testing and performance**       | Add functional tests for `/predict` and `/batch-predict` with sample job posts. Measure latency; aim for <2 seconds per prediction (Implementation Plan 1 §6.1). Add basic error handling and validation tests.                                                                                                                                                                                                                                         |


Optional later steps: API key or rate limiting, Dockerfile for deployment, and (if needed) a small adapter to import preprocessing code from thesis-scam-job-post instead of duplicating it.

## Code References

- `app/main.py:1–18` – FastAPI app, `GET /` and `GET /health` only; no prediction or preprocessing.
- `requirements.txt` – FastAPI and Uvicorn only; no ML stack.
- `cursor/project/notes/Implementation Plan 1.md` – Phases 1–6; Phase 5 describes backend and API.
- `cursor/project/notes/Implementation Plan 2.md` – Second-dataset preprocessing and features; same deployment expectations.

## Architecture Documentation

- **API**: FastAPI; production server Uvicorn.
- **Deployment target**: Local or cloud; Phase 5 suggests Docker.
- **Model source**: External (thesis-scam-job-post); backend depends on a configured path to artifacts and, if applicable, shared preprocessing code.
- **Data flow**: Client → JSON body → validation → preprocessing (align with training) → model inference → JSON response (prediction, confidence, warning_signals).

## Historical Context (from cursor/project/notes/)

- **Implementation Plan 1** – Defines the full scam detection pipeline and Phase 5 backend scope (model serving, FastAPI, `/predict`, `/batch-predict`, `/health`, confidence and warning signals).
- **Implementation Plan 2** – Same pipeline for a second dataset; reinforces that preprocessing and feature engineering must be consistent between training and inference.

## Related Research

- None in this repo yet. Future work could add a short “Model artifact layout in thesis-scam-job-post” note (paths to `.pkl`, `.h5`, tokenizers, and preprocessing scripts) to speed up steps 3–4.

## Open Questions

1. **Exact path to thesis-scam-job-post** – Repo path or copy strategy (e.g. copy only `models/` and preprocessing scripts into this repo) for steps 1 and 4.
2. **Which model to deploy first** – Single best model (e.g. DistilBERT or Random Forest) vs full hybrid ensemble; affects step 4 and dependencies.
3. **Input schema** – Whether `/predict` receives a single concatenated text field or structured fields (job_title, job_desc, company_profile, etc.) to match Plan 2; affects step 3 and API contract.

