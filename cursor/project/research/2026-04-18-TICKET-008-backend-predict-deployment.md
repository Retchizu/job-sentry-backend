---
date: 2026-04-18T02:14:52Z
researcher: riche
git_commit: 26c01727e996da4fcc64221713a2f75fad464f18
branch: main
repository: job-sentry-backend
topic: "TICKET-008: Backend deployment for POST /predict — codebase as-is"
tags: [research, codebase, predict, fastapi, phase6, fused, inference, TICKET-008]
status: complete
last_updated: 2026-04-18
last_updated_by: riche
metadata_note: "hack/spec_metadata.sh was not present in the repository; git hash, branch, and timestamps were gathered manually."
---

# Research: TICKET-008 — Backend deployment for `/predict`

**Date**: 2026-04-18T02:14:52Z  
**Researcher**: riche  
**Git Commit**: `26c01727e996da4fcc64221713a2f75fad464f18`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

How does the repository today implement (or relate to) [`cursor/project/tickets/TICKET-008-backend-predict-deployment.md`](../tickets/TICKET-008-backend-predict-deployment.md): `POST /predict`, model loading at startup, request payload, preprocessing, inference, response shape, validation, errors, tests, and documentation?

## Summary

The backend exposes **`POST /predict`** on the FastAPI app in [`app/main.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py). The phase6 fused model and tokenizer are loaded **once at application startup** in the **`lifespan`** context manager and stored on **`app.state.predictor`**. Request and response bodies are defined in [`app/schemas.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py). Inference runs through [`FusedScamPredictor`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py), which loads artifacts via [`load_fused_artifacts`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py). The HTTP response uses **binary scam classification** (`scam_probabilities`, `predicted_scam`) plus **heuristic string warning codes**, not the three-way `legit` / `warning` / `fraud` label set described in TICKET-008. Integration tests live in [`tests/test_api.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_api.py). The root [`README.md`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/README.md) documents the predict contract in prose; interactive API docs are served at `/docs`.

## Detailed Findings

### TICKET-008 scope (reference)

The ticket file asks for `POST /predict`, startup loading of tokenizer/model, raw text or `combined_text`-style input, preprocessing plus inference, mapping class index to **`legit` / `warning` / `fraud`**, a JSON response including **`predicted_class`**, **`predicted_label`**, per-class probabilities, **`confidence`**, validation and error handling, and basic observability (latency + success/error logging). Acceptance criteria include callable example from local/dev.

### Route registration and handler

- [`create_app()`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L131-L147) registers `POST /predict` with response model `PredictResponse`.
- [`predict(request, body)`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L96-L128) reads `PredictRequest`, resolves the predictor from `request.app.state`, builds text strings per post, computes warnings, enforces `max_batch_size`, calls `predictor.predict_proba`, applies threshold to booleans, returns `PredictResponse`.

### Startup model loading (not per request)

- [`lifespan`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L39-L49) assigns `app.state.predictor = _load_predictor(settings)`.
- [`_load_predictor`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L23-L36) returns `None` if `JOBSENTRY_PHASE6_FUSED_DIR` is unset; otherwise constructs `FusedScamPredictor.from_artifact_dir(...)`.
- On load failure when the directory **is** set, startup raises `RuntimeError` after logging an exception ([`main.py` lines 44–48](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L44-L48)).

### Request payload

- [`PredictRequest`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L58-L59): `posts: list[JobPostInput]` with `min_length=1`.
- [`JobPostInput`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L31-L55): optional `text`, `job_title`, `job_desc`, `skills_desc`, `company_profile`, and optional `rate` (`RateInput`). [`combined_text()`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L41-L55) uses non-empty `text` if present, else joins structured fields; empty input raises `ValueError`. The docstring on [`RateInput`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L10-L11) states rate is not used by `combined_text()` yet.

### Preprocessing and inference

- Text fed to the model is exactly the string from `combined_text()` for each post ([`main.py` 105–120](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L105-L120)).
- [`FusedScamPredictor.predict_proba`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py#L62-L92) returns **P(scam)** per row as softmax class index 1 over two logits (`HybridFusedClassifier`).

### Response shape (actual vs TICKET-008)

- [`PredictResponse`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L62-L70): `scam_probabilities`, `predicted_scam` (list of bool), `threshold`, `warnings` (list of lists of string codes).
- Label mapping in the handler is **binary**: `predicted_scam[i] = scam_probabilities[i] >= threshold` ([`main.py` 120–127](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L120-L127)). There are no fields named `predicted_class`, `predicted_label`, `legit_probability`, `warning_probability`, `fraud_probability`, or `confidence` on the response model.

### Heuristic warnings (separate from model logits)

- [`compute_warnings`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/predict_warnings.py#L62-L71) returns sorted unique regex-based codes (e.g. `upfront_payment`, `off_platform_contact`). These are **not** the same as a `warning` class label in TICKET-008’s three-class scheme.

### Validation and HTTP errors

- **503** if `app.state.predictor` is `None` ([`main.py` 99–103](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L99-L103)).
- **422** if `combined_text()` raises `ValueError` or batch length exceeds `max_batch_size` ([`main.py` 109–118](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L109-L118)).
- Pydantic validation errors for malformed JSON or invalid `rate` constraints surface as FastAPI **422** responses (covered in tests).

### Settings ([`app/config.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/config.py))

- `JOBSENTRY_PHASE6_FUSED_DIR`, `JOBSENTRY_PHASE6_FUSED_CHECKPOINT`, `JOBSENTRY_DEVICE`, `JOBSENTRY_MAX_BATCH_SIZE`, `JOBSENTRY_CONFIDENCE_THRESHOLD` (defaults include `max_batch_size=50`, `confidence_threshold=0.5`).

### Observability (logging)

- Module logger in [`main.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py); info when no fused dir is set; exception log on failed load. The `predict` handler does not log per-request latency or success/error counts.

### Tests

- [`tests/test_api.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_api.py): 503 without model, happy path with injected fake predictor, rate field, warnings on scammy text, 422 for invalid rate, optional real-artifact health check when `artifacts/models/phase6_fused` exists.
- Additional unit tests: `tests/test_fused_predictor.py`, `tests/test_fused_loader.py`, `tests/test_schemas.py`, `tests/test_predict_warnings.py`, `tests/conftest.py` (settings cache clear).

### Documentation for callers

- [`README.md` Predict section](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/README.md#L52-L57) describes `POST /predict` body and response fields in prose. [`RootResponse`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L82-L85) exposes `docs: "/docs"` for OpenAPI/Swagger.

## Code References

- `app/main.py:39-49` — Lifespan loads predictor once; failure modes when dir set vs unset.
- `app/main.py:96-128` — `predict` handler: 503/422, `combined_text`, warnings, `predict_proba`, threshold, response build.
- `app/main.py:146` — `POST /predict` route registration.
- `app/schemas.py:10-70` — `RateInput`, `JobPostInput`, `PredictRequest`, `PredictResponse`.
- `app/fused_predictor.py:38-92` — `from_artifact_dir`, `predict_proba` / batch softmax P(scam).
- `app/fused_loader.py` — Artifact resolution and weight loading (referenced by predictor).
- `app/predict_warnings.py:9-71` — Regex patterns and `compute_warnings`.
- `app/config.py:13-47` — `Settings` and `get_settings`.
- `tests/test_api.py` — HTTP-level behavior for `/predict`.

## Architecture Documentation

- **Stack**: FastAPI, Pydantic v2 schemas, PyTorch `HybridFusedClassifier`, Hugging Face `DistilBertTokenizerFast`, artifacts under a single directory (e.g. `artifacts/models/phase6_fused/` per README).
- **Inference path**: JSON → `JobPostInput` → `combined_text()` → `FusedScamPredictor.predict_proba` → float probabilities → boolean labels vs threshold from `fused_meta` or settings; parallel `compute_warnings` on raw combined strings.

## Historical Context (from `cursor/project/`)

- [`cursor/project/implementation/2026-03-22-NA-phase6-fused-production-inference.md`](../implementation/2026-03-22-NA-phase6-fused-production-inference.md) — Records the phase6 fused FastAPI wiring (`lifespan`, `/predict`, tests).
- [`cursor/project/research/2026-03-22-phase6-fused-production-inference-next-steps.md`](2026-03-22-phase6-fused-production-inference-next-steps.md) — Post-implementation verification notes for the fused path.
- [`cursor/project/implementation/2026-03-22-NA-add-rate-field-job-post-body.md`](../implementation/2026-03-22-NA-add-rate-field-job-post-body.md) — Documents adding optional `rate` to the request body without changing `combined_text()` for the model.
- Older docs under `cursor/project/research/` and `cursor/project/implementation/` describe prior layouts (e.g. `app/services/prediction.py`, `/batch-predict`); the **current** committed tree centers on `app/main.py` and fused modules as in this research.

## Related Research

- [`2026-03-22-phase6-fused-production-inference-next-steps.md`](2026-03-22-phase6-fused-production-inference-next-steps.md) — Manual verification and configuration behavior for the fused API.

## Open Questions

- Whether TICKET-008’s three-class **`legit` / `warning` / `fraud`** contract and probability breakdown are planned as a future API revision; the present codebase implements **two-class** softmax output plus separate heuristic **warning code** lists.
