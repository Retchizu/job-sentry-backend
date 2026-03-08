# JobSentry Backend Implementation Plan

## Overview

Implement the JobSentry prediction backend so it serves scam/legitimate classification for job posts. The backend will load a production model from the thesis project (thesis-scam-job-post), run the same preprocessing and feature pipeline used at training, and expose `POST /predict`, `POST /batch-predict`, and `GET /health` with responses that include prediction, confidence score, and warning signals. Target performance is <2 seconds per prediction (per Implementation Plan 1 §6.1).

## Current State Analysis

- `**app/main.py**`: FastAPI app with title "Job Sentry API", `GET /` and `GET /health` only; no prediction or preprocessing.
- `**requirements.txt**`: Only `fastapi>=0.115.0` and `uvicorn[standard]>=0.32.0`; no ML or data dependencies.
- **Config**: No config module or env/settings; no `config/`, `.env`, or `.env.example`.
- **Tests**: No test directory, test files, or pytest configuration.

Model artifacts and preprocessing logic live in a separate project (thesis-scam-job-post). The backend will depend on a configurable path to those artifacts and, if needed, to shared preprocessing code.

### Key Discoveries

- Implementation Plan 1 Phase 5 defines backend scope: model serving, FastAPI, `/predict`, `/batch-predict`, `/health`, and returned fields (prediction, confidence, warning signals).
- Implementation Plan 2 defines the input schema and preprocessing: `job_title`, `job_desc`, `skills_desc`, `company_profile`, `salary_range`, `employment_type`, and pre-encoded `location`/`industry`; combined text and structural features must match training.
- Research sequence (2025-03-08) orders work as: config → dependencies → preprocessing → model loading → prediction service → API endpoints → response schema → testing and performance.

## Desired End State

1. **Configuration**: Model artifact path (and optional feature/config paths) configurable via env or config module; no paths or secrets hardcoded.
2. **Dependencies**: `requirements.txt` includes inference stack (e.g. scikit-learn, pandas, and either TensorFlow/PyTorch + transformers as needed) aligned with thesis training versions where possible.
3. **Preprocessing**: A single preprocessing pipeline accepts raw job post fields, cleans/normalizes text, and produces the same combined text and structural features used in training; used only internally by the prediction service.
4. **Model**: At startup, the app loads the chosen production model(s) from the configured path (e.g. `.pkl`, `.h5`, `.pth` or Hugging Face); if ensemble, loads all components and applies the same voting/stacking logic as in the thesis.
5. **Prediction service**: Single entry point that takes one job post (raw/structured), runs preprocessing and inference, and returns binary prediction (scam/legitimate), confidence score, and optional warning signals.
6. **API**:
  - `POST /predict`: one job post (JSON); validation (required fields, max length); response: prediction, confidence, warning_signals.
  - `POST /batch-predict`: list of job posts; same response shape per item; consider timeouts/limits for <2 s per item.
  - `GET /health`: retained; optionally checks model loaded and/or a dummy prediction (liveness/readiness).
7. **Response schema**: Standardized JSON with `prediction`, `confidence`, `warning_signals` (and any agreed extras).
8. **Testing and performance**: Functional tests for `/predict` and `/batch-predict` with sample payloads; latency measured; target <2 s per prediction; validation and error-handling tests.

### Verification

- All new tests pass.
- `GET /health` returns 200 and (if implemented) confirms model and dummy prediction.
- `POST /predict` and `POST /batch-predict` return the defined schema and behave correctly on valid and invalid input.
- Measured p95 latency per prediction is under 2 seconds in the target environment.

## What We're NOT Doing

- Frontend or browser extension (Phase 5.2–5.3); this plan is backend-only.
- Training or retraining models; we consume existing thesis artifacts.
- API key auth or rate limiting in this plan (called out as optional later steps in research).
- Dockerfile or deployment automation in this plan (optional later).
- Changes to thesis-scam-job-post repo; we only read from a configured path or use copied artifacts.

## Implementation Approach

Follow the research sequence: (1) config and model path, (2) dependencies, (3) preprocessing pipeline, (4) model loading, (5) prediction service, (6) POST /predict, (7) POST /batch-predict, (8) GET /health enhancement, (9) response schema standardization, (10) testing and performance. Preprocessing and feature contract must match the thesis training pipeline so live inputs are scored consistently (Implementation Plan 2).

---

## Phase 1: Environment and Configuration

### Overview

Introduce a config module or env-based settings for model artifact path and optional feature/config paths. Keep secrets and paths out of application code.

### Changes Required

#### 1. Config module or env loading

**File**: New `app/config.py` (or equivalent)

- Load from environment: e.g. `MODEL_ARTIFACT_PATH`, optional `FEATURE_CONFIG_PATH` or similar.
- Use `os.getenv` with sensible defaults for local dev (e.g. relative `models/` or a path to thesis-scam-job-post).
- Expose a simple settings object (e.g. `Settings` dataclass or Pydantic) used by the rest of the app.

**File**: `.env.example` (optional but recommended)

- Document `MODEL_ARTIFACT_PATH` and any other required/optional variables so deployers know what to set.

### Success Criteria

#### Automated Verification

- Application starts without error when required env (or defaults) are set.
- No hardcoded absolute paths to model or thesis repo in project code (grep check).

#### Manual Verification

- Config values are correctly read when env vars are set.
- Defaults allow local runs (e.g. with a placeholder or existing artifact path).

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 2: Dependencies

### Overview

Extend `requirements.txt` with the libraries needed for inference so the app can load the chosen model and run preprocessing (e.g. scikit-learn, pandas, and either TensorFlow or PyTorch plus transformers). Match thesis training versions where practical.

### Changes Required

#### 1. requirements.txt

**File**: `requirements.txt`

- Add at minimum: `scikit-learn`, `pandas`, and either `tensorflow` or `torch` plus `transformers` (and any tokenizer/embedding deps) depending on whether the first deployed model is Bi-LSTM, DistilBERT, or sklearn-based.
- Pin or constrain versions to align with thesis-scam-job-post where documented; otherwise use compatible recent versions.

### Success Criteria

#### Automated Verification

- `pip install -r requirements.txt` completes successfully in a clean venv.
- No import errors when importing the chosen stack (e.g. sklearn, pandas, and the selected DL framework) in a minimal script or at app startup.

#### Manual Verification

- Versions are documented or pinned so future deploys stay reproducible.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 3: Preprocessing Pipeline

### Overview

Implement or reuse from thesis-scam-job-post a single preprocessing pipeline that: (a) accepts raw job post fields (job_title, job_desc, skills_desc, company_profile, salary_range, employment_type, location, industry, etc.), (b) cleans and normalizes text, (c) builds the same combined text and structural features used in training (Implementation Plan 1 §2, Plan 2 §2–3). Expose as an internal function or module used only by the prediction service.

### Changes Required

#### 1. Preprocessing module

**File**: New `app/preprocessing.py` (or `app/services/preprocessing.py`)

- Input: structured fields matching the API input schema (e.g. job_title, job_desc, company_profile, skills_desc, salary_range, employment_type, and if applicable location, industry).
- Text cleaning: HTML/URL removal, normalization, whitespace handling, encoding (per Plan 2 §2.1–2.2).
- Combined text: same construction as training (e.g. `job_title + " " + job_desc + " " + skills_desc + " " + company_profile` or use of existing `text` if that contract is chosen).
- Structural features: e.g. has_salary, has_company_profile, has_skills_desc, lengths; employment_type and location/industry handling consistent with Plan 2 §2.4 and §3.2.
- Output: feature representation (and combined text) that the loaded model expects (TF-IDF inputs, tokenized sequences, or both depending on model type).

If reusing from thesis-scam-job-post, add a thin wrapper or adapter in this repo that calls the shared code and keeps the same input/output contract.

### Success Criteria

#### Automated Verification

- Unit tests for preprocessing: given sample raw inputs, output shape and types match what the model expects (or a stub model interface).
- No regression in feature contract: same inputs as in thesis notebooks produce compatible feature representation.

#### Manual Verification

- Compare output for a few job posts against thesis pipeline output (if artifacts available) to confirm alignment.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 4: Model Loading

### Overview

At startup, load the chosen production model(s) from the configured path. Support formats produced in the thesis (e.g. pickle for sklearn, `.h5`/Keras for Bi-LSTM, `.pth` or Hugging Face for DistilBERT). If the chosen model is an ensemble (Plan 1 §3.3), load all components and implement the same voting/stacking logic as in the notebook.

### Changes Required

#### 1. Model loader

**File**: New `app/model.py` or `app/services/model.py`

- Read `MODEL_ARTIFACT_PATH` (and any extra paths) from config.
- Load artifact(s): support at least one of `.pkl`, `.h5`, `.pth`, or Hugging Face model/tokenizer.
- If ensemble: load each component and implement the same combination logic (voting/stacking) as in the thesis.
- Expose a single predict interface: input = preprocessed features (and/or raw text for DL models), output = probability or class + optional confidence.

#### 2. Application startup

**File**: `app/main.py`

- On startup (e.g. `lifespan` or `on_event("startup")`), call the model loader and store the loaded model in app state (e.g. `app.state.model`) so routes can access it.

### Success Criteria

#### Automated Verification

- With a dummy or small real artifact at `MODEL_ARTIFACT_PATH`, app starts and `app.state.model` is set.
- Unit test: loader returns a callable or object with a predict method that returns a value in the expected range (e.g. probability in [0, 1]).

#### Manual Verification

- With the real thesis model artifact path, startup loads without error and a single test prediction runs.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 5: Prediction Service

### Overview

Implement a single entry point that: (1) takes one job post (raw text or structured fields), (2) runs the preprocessing pipeline from Phase 3, (3) runs model inference, (4) returns binary prediction (scam/legitimate), confidence score, and optional warning signals (e.g. "Suspicious keywords", "Missing company details"). Reuse or mirror any confidence/warning logic from the thesis code.

### Changes Required

#### 1. Prediction service

**File**: New `app/services/prediction.py` (or `app/prediction.py`)

- Function or class that accepts one job post (dict or Pydantic model) and the loaded model (from app state).
- Call preprocessing, then model predict.
- Map model output to: `prediction` (scam/legitimate), `confidence` (float), `warning_signals` (list of strings).
- If thesis has confidence or warning logic, replicate or adapt it here.

### Success Criteria

#### Automated Verification

- Unit tests: for sample inputs, service returns dict (or structured object) with keys `prediction`, `confidence`, `warning_signals` and types as specified.
- Edge cases: empty or minimal input handled without crashing; confidence in [0, 1].

#### Manual Verification

- For a few known job posts, output is sensible and consistent with thesis behavior when available.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 6: POST /predict

### Overview

Add `POST /predict` that accepts one job post in the request body (JSON). Validate input (required fields, max length). Call the prediction service and return prediction, confidence, and warning signals in the standardized JSON schema.

### Changes Required

#### 1. Request and response schemas

**File**: New `app/schemas.py` (or under `app/api/`)

- Pydantic models: `JobPostInput` (job_title, job_desc, company_profile, skills_desc, salary_range, employment_type, and optionally location, industry, etc.) with validation (required fields, max lengths).
- `PredictResponse`: prediction (str or enum), confidence (float), warning_signals (list[str]).

#### 2. Predict endpoint

**File**: `app/main.py` or `app/api/routes.py`

- `POST /predict`: body = `JobPostInput`, validate, call prediction service, return `PredictResponse`.
- Use 422 or 400 for validation errors with clear messages.

### Success Criteria

#### Automated Verification

- Test: valid body returns 200 and response schema matches `PredictResponse`.
- Test: missing required field returns 422.
- Test: oversized field returns 422 or 400.

#### Manual Verification

- Request via docs UI or curl with a real job post returns sensible prediction and confidence.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 7: POST /batch-predict

### Overview

Add `POST /batch-predict` that accepts a list of job posts. Run preprocessing and inference for each (or in batch if the model supports it). Return a list of results with the same shape as single predict. Consider rate limits and timeouts so large batches stay under the <2 s per item target where possible.

### Changes Required

#### 1. Batch endpoint

**File**: `app/main.py` or `app/api/routes.py`

- `POST /batch-predict`: body = list of job posts (same schema as single), return list of `PredictResponse`.
- Optional: max batch size (e.g. 50) and timeout to avoid long-running requests; return 413 or 400 if over limit.
- Reuse preprocessing and prediction service per item (or batched inference if supported).

### Success Criteria

#### Automated Verification

- Test: list of 2–3 valid posts returns 200 and list of responses with correct length and schema.
- Test: empty list or over-size list handled (400/413 or documented behavior).

#### Manual Verification

- Batch of 10–20 posts completes within acceptable time (e.g. <2 s per item or document deviation).

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 8: GET /health Enhancement

### Overview

Keep existing `GET /health`; optionally extend it to verify that the model is loaded and that a single dummy prediction succeeds (liveness/readiness).

### Changes Required

#### 1. Health endpoint

**File**: `app/main.py`

- Retain current behavior: return `{"status": "ok"}`.
- Optional: if model is in app state, include e.g. `"model_loaded": true` and optionally run a minimal dummy prediction; if model not loaded or prediction fails, return 503 and `"status": "degraded"` or `"unhealthy"`.

### Success Criteria

#### Automated Verification

- Test: when model is loaded, `GET /health` returns 200.
- If readiness implemented: when model fails to load, health returns 503 (or documented behavior).

#### Manual Verification

- Health response is useful for orchestrators (e.g. Kubernetes) if used in deployment.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 9: Response Schema Standardization

### Overview

Standardize response JSON so all predict endpoints return the same shape: e.g. `prediction` (scam/legitimate), `confidence`, `warning_signals`. Document the schema (e.g. in OpenAPI or a small doc).

### Changes Required

#### 1. Schema and docs

**File**: `app/schemas.py` and `app/main.py`

- Ensure `POST /predict` and `POST /batch-predict` both use the same `PredictResponse` model.
- Add response_model to FastAPI route declarations so OpenAPI shows the exact schema.
- Optional: add a short markdown or comment describing the semantics of warning_signals and confidence.

### Success Criteria

#### Automated Verification

- OpenAPI schema for `/predict` and `/batch-predict` includes the same response structure.
- Existing tests still pass.

#### Manual Verification

- API docs at `/docs` clearly show request/response shapes.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Phase 10: Testing and Performance

### Overview

Add functional tests for `/predict` and `/batch-predict` with sample job posts. Measure latency and aim for <2 seconds per prediction (Implementation Plan 1 §6.1). Add basic error-handling and validation tests.

### Changes Required

#### 1. Test setup

**File**: New `tests/conftest.py`

- Pytest fixtures: FastAPI test client, optional fixture that loads a small/dummy model for tests so endpoints can be tested without the full thesis artifact.
- Sample job post payload(s) as fixtures.

#### 2. API tests

**File**: New `tests/test_predict_api.py` (or similar)

- Test `POST /predict`: valid input → 200 and schema; invalid → 422/400.
- Test `POST /batch-predict`: valid list → 200 and list of results; empty or over limit → appropriate status.
- Test `GET /health`: 200 when app is ready.

#### 3. Performance check

**File**: `tests/test_performance.py` or a script

- Measure latency for a single prediction (e.g. p50/p95) and optionally for a small batch; document or assert target <2 s per prediction in the target environment.
- If no real model in CI, document how to run the performance check locally with the real artifact.

### Success Criteria

#### Automated Verification

- All tests pass: `pytest tests/` (or equivalent).
- Linting/formatting passes if configured (e.g. ruff, black).

#### Manual Verification

- With real model, p95 latency per prediction is under 2 seconds (or deviation documented).
- Error handling and validation messages are clear for API consumers.

**Implementation Note**: After completing this phase and automated checks pass, pause for manual confirmation before proceeding.

---

## Testing Strategy

### Unit Tests

- Preprocessing: input → output shape and type; alignment with thesis feature contract.
- Model loader: with stub artifact, returns callable predict; optional test with real artifact.
- Prediction service: returns prediction, confidence, warning_signals; edge cases (empty, minimal input).

### Integration Tests

- `POST /predict` and `POST /batch-predict` with test client and stub or real model; status codes and response schema.
- `GET /health` with and without model loaded.

### Manual Testing Steps

1. Set `MODEL_ARTIFACT_PATH` to thesis model (or copy artifact into repo).
2. Start app: `uvicorn app.main:app --reload`.
3. Open `/docs`, call `POST /predict` with a sample job post; verify prediction and confidence.
4. Call `POST /batch-predict` with 5–10 posts; verify list of results and latency.
5. Call `GET /health` and confirm model_loaded or readiness.

## Performance Considerations

- Target: <2 seconds per prediction (Plan 1 §6.1). Measure with real model and document; if DL model is large, consider batch inference or model optimization.
- Batch endpoint: cap batch size and/or add timeout to avoid long-running requests.
- Preprocessing: avoid redundant work in batch (e.g. vectorizer fit only at load; transform per request).

## Migration Notes

- No database or existing API contract to migrate. New endpoints are additive; `GET /` and `GET /health` remain unchanged except optional health enhancement.
- If thesis-scam-job-post is not on the same machine, copy required artifacts (model files, tokenizers, preprocessing scripts) to a path pointed to by `MODEL_ARTIFACT_PATH`, or document the path convention for deployment.

## References

- Research: `cursor/project/research/2025-03-08-jobsentry-backend-sequence-plan.md`
- Implementation Plan 1: `cursor/project/notes/Implementation Plan 1.md` (Phase 5 backend, Phase 6 testing)
- Implementation Plan 2: `cursor/project/notes/Implementation Plan 2.md` (preprocessing, features, schema)
- Current app: `app/main.py` (lines 1–18)

