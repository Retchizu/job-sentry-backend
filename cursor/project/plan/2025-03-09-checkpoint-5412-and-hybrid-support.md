# Checkpoint 5412 Default and Hybrid Model Support — Implementation Plan

## Overview

This plan does two things: (1) switch the default DistilBERT checkpoint from 1566 to **5412** (numerically last training step) and document it; (2) add support for the Implementation Plan 1 hybrid ensemble (traditional ML + optional Bi-LSTM + DistilBERT) with configurable paths and a default combination method (soft voting). The backend remains backward-compatible: when no hybrid artifacts are configured, behavior is unchanged (single DistilBERT).

## Current State Analysis

- **Checkpoint**: Default is `checkpoint-1566` in `app/config.py` and `.env.example`. Five checkpoints are documented (783, 1566, 2349, 3608, 5412). No "best" or "final" was defined in this repo; the decision is to use **5412** as the default.
- **Model**: Single `ScamDetectionModel` (DistilBERT) loaded at startup (`app/main.py`), used by `app/services/prediction.py`. Config explicitly states phase6_merged and bilstm_checkpoints are not supported (`app/config.py:8`).
- **Preprocessing**: `app/preprocessing.py` builds `combined_text` and structural features; there is no TF-IDF vectorization or traditional-ML feature path in the backend.
- **No ensemble**: No second model, no voting or stacking logic.

### Key Discoveries

- `app/config.py:12` — `model_artifact_path` default; env `JOBSENTRY_MODEL_ARTIFACT_PATH`.
- `app/model.py:55-105` — `load_model()` returns a single `ScamDetectionModel`; expects `model.safetensors` in artifact directory.
- `app/services/prediction.py` — `predict_single` / `predict_batch` call `preprocess_job_post` then `model.predict` / `model.predict_batch`; no ensemble.
- `app/preprocessing.py` — `build_combined_text` and `preprocess_job_post` produce `combined_text`; no TF-IDF.
- Implementation summary: DistilBERT only because BiLSTM had no saved tokenizer and no sklearn models were exported from thesis; hybrid support was out of scope.

## Desired End State

1. **Checkpoint**: Default artifact path is `thesis-scam-job-post/models/distilbert_run/checkpoint-5412`. Docs (README, .env.example) state that 5412 is the chosen default (final training step).
2. **Hybrid**: When hybrid is enabled via config (paths set), the app loads (a) DistilBERT (unchanged), (b) optional traditional ML artifact (e.g. phase6_merged joblib pipeline), (c) optional Bi-LSTM (Keras). Prediction runs each enabled model and combines outputs (default: soft voting — average of scam probabilities). When hybrid paths are not set, behavior is unchanged (DistilBERT only).
3. **Verification**: All existing tests pass; new tests cover ensemble combination and optional loaders; manual test with real artifacts when available.

### Prerequisite for Hybrid (No Thesis Details Yet)

Exact path and format of `phase6_merged` and Bi-LSTM in the thesis repo are not specified. The plan therefore:

- Uses **configurable paths** (env) for traditional ML and Bi-LSTM artifacts; either path empty/unset means that branch is disabled.
- Assumes **phase6_merged** is a joblib (or pickle) artifact containing a fitted pipeline that accepts text (e.g. `TfidfVectorizer` + classifier); input is `combined_text` from existing preprocessing. If the thesis uses a different format (e.g. separate vectorizer + model), we load accordingly once the format is documented.
- Assumes **Bi-LSTM** is optional; format Keras `.h5` or `.keras`; input is tokenized sequence (we may need to load thesis tokenizer or align with a fixed vocab). If not available, hybrid runs with DistilBERT + traditional ML only.
- Uses **soft voting** (average of scam probabilities) as the default combination method; config can be extended later for stacking or fixed weights if the thesis specifies.

Once thesis artifact paths and formats are known, update config defaults or add a short "thesis artifact spec" doc and adjust loaders if needed.

## What We're NOT Doing

- Adding `model.safetensors` or thesis artifacts to this repo (they remain in thesis project or artifact store).
- Changing the thesis-scam-job-post repo.
- Implementing stacking or custom weights in the first version (default is soft voting; extensible later).
- Breaking existing API: `/predict` and `/batch-predict` request/response shapes stay the same; only the internal model(s) and combination logic change when hybrid is enabled.

## Implementation Approach

First switch default checkpoint to 5412 and document (small, low-risk). Then add hybrid in phases: config and types, traditional ML loader, optional Bi-LSTM loader, ensemble combiner, wire prediction service, tests and docs. Backward compatibility: no hybrid paths → single DistilBERT as today.

---

## Phase 1: Checkpoint 5412 as Default

### Overview

Change the default DistilBERT checkpoint from 1566 to 5412 and update documentation to state that 5412 is the chosen default (final training step).

### Changes Required

#### 1. Config default

**File**: `app/config.py`

**Changes**: Set default `model_artifact_path` to `Path("thesis-scam-job-post/models/distilbert_run/checkpoint-5412")`. Update the comment for supported paths to list 5412 as the default.

#### 2. Env example and README

**File**: `.env.example`

**Changes**: Set `JOBSENTRY_MODEL_ARTIFACT_PATH=thesis-scam-job-post/models/distilbert_run/checkpoint-5412`.

**File**: `README.md`

**Changes**: In "Running with the real model" and "Checkpoint", replace references to checkpoint-1566 with checkpoint-5412. Add one line that the default checkpoint is 5412 (final training step).

### Success Criteria

#### Automated Verification

- [x] Existing tests pass: `python -m pytest tests/ -v`
- [x] No new lint errors in modified files

#### Manual Verification

- [x] With no env override, app loads from `.../checkpoint-5412` when that path exists; logs show "Loading fine-tuned model from ... checkpoint-5412"
- [x] README and .env.example clearly state default is checkpoint-5412

**Implementation Note**: After Phase 1 and automated verification, pause for manual confirmation before proceeding to Phase 2.

---

## Phase 2: Config and Types for Hybrid

### Overview

Add configuration and types for optional traditional ML and Bi-LSTM artifact paths and for ensemble mode. When these paths are unset, behavior remains DistilBERT-only.

### Changes Required

#### 1. Config

**File**: `app/config.py`

**Changes**:

- Add optional `phase6_merged_path: Optional[Path] = None` (env: `JOBSENTRY_PHASE6_MERGED_PATH`). If set, points to joblib (or pickle) artifact for traditional ML pipeline.
- Add optional `bilstm_artifact_path: Optional[Path] = None` (env: `JOBSENTRY_BILSTM_ARTIFACT_PATH`). If set, points to Keras model directory or file (.h5/.keras).
- Add `hybrid_combination: str = "soft_voting"` (env: `JOBSENTRY_HYBRID_COMBINATION`) with allowed values e.g. `soft_voting` only for now; reserved for future `stacking` / `weighted`.
- Update comment: when `phase6_merged_path` or `bilstm_artifact_path` is set, hybrid mode is enabled (DistilBERT + those components).

#### 2. Types / interface for ensemble

**File**: `app/model.py` (or new `app/ensemble.py` if preferred)

**Changes**: Introduce a small interface or protocol for "something that can predict scam probability from text (or combined text)" so that:

- `ScamDetectionModel` (DistilBERT) implements it (e.g. returns scam probability).
- Future traditional ML wrapper and Bi-LSTM wrapper implement the same (single text → float probability).
- Ensemble combiner takes a list of such predictors and combines with soft voting (average of probabilities). If only one predictor is present, no averaging.

No change yet to `load_model()` return type in Phase 2; just add the types and a placeholder or stub for the combiner used in later phases.

### Success Criteria

#### Automated Verification

- [x] Tests pass: `python -m pytest tests/ -v`
- [x] Config loads with new optional env vars; unset leaves them None

#### Manual Verification

- [ ] Setting `JOBSENTRY_PHASE6_MERGED_PATH` in .env does not break startup when path is missing (loader can fail gracefully in Phase 3)

**Implementation Note**: After Phase 2, pause if desired before implementing loaders.

---

## Phase 3: Traditional ML Loader (phase6_merged)

### Overview

Load an optional traditional ML artifact (joblib pipeline) from `phase6_merged_path`. Pipeline is assumed to accept a list of raw text strings and return predictions or probabilities; input is `combined_text` from existing preprocessing. If path is unset or load fails, hybrid runs without this component (DistilBERT only, or DistilBERT + Bi-LSTM if that is added).

### Changes Required

#### 1. Dependencies

**File**: `requirements.txt`

**Changes**: Add `joblib` (and `scikit-learn` if not already present) for loading the pipeline.

#### 2. Loader

**File**: `app/model.py` or new `app/traditional_ml.py`

**Changes**:

- Implement `load_phase6_merged(path: Path)` that:
  - Loads artifact with `joblib.load(path)` (or pickle if format differs; document assumption).
  - Wraps it in a small object that exposes `predict_proba(texts: List[str]) -> List[float]` (scam probability per text), using `combined_text` as input. If the pipeline expects a different input (e.g. pre-vectorized), adapt once thesis format is known.
  - On file-not-found or load error, log warning and return None (hybrid without traditional ML).
- Call this from startup only when `settings.phase6_merged_path` is set and path exists.

#### 3. Startup wiring

**File**: `app/main.py`

**Changes**: In lifespan, after loading DistilBERT, if `settings.phase6_merged_path` is set, call `load_phase6_merged(settings.phase6_merged_path)` and attach result to `app.state` (e.g. `app.state.phase6_model`). If None, hybrid will use only DistilBERT (and optional Bi-LSTM from Phase 4).

### Success Criteria

#### Automated Verification

- [x] Unit test: with a tiny joblib pipeline (e.g. CountVectorizer + LogisticRegression) saved to a temp file, loader returns a callable that returns probabilities for a list of strings.
- [x] When path is unset, loader is not called; app starts as today.
- [x] `python -m pytest tests/ -v` passes

#### Manual Verification

- [ ] With a real phase6_merged artifact at a path and `JOBSENTRY_PHASE6_MERGED_PATH` set, app starts and does not crash (combination used in Phase 5).

**Implementation Note**: If thesis artifact format differs (e.g. separate vectorizer + model), document in code and adjust loader; plan assumes a single pipeline for now.

---

## Phase 4: Optional Bi-LSTM Loader

### Overview

Load an optional Bi-LSTM model from `bilstm_artifact_path` (Keras .h5 or .keras). Input to the model is tokenized text; we need to align with thesis tokenization (e.g. load tokenizer from artifact dir or use a fixed vocab). If path is unset or load fails, hybrid runs without Bi-LSTM.

### Changes Required

#### 1. Dependencies

**File**: `requirements.txt`

**Changes**: Add `tensorflow` or `keras` (per project preference) for loading Keras models.

#### 2. Loader

**File**: `app/model.py` or new `app/bilstm.py`

**Changes**:

- Implement `load_bilstm(path: Path)` that:
  - Loads Keras model from path (e.g. `keras.models.load_model(path)`).
  - Wraps it so it exposes `predict_proba(texts: List[str]) -> List[float]` (scam probability). Tokenization: use a tokenizer from the artifact directory if present, else document that we need thesis tokenizer/vocab and stub or use a simple tokenizer for tests.
  - On load error, log warning and return None.
- Call from startup when `settings.bilstm_artifact_path` is set.

#### 3. Startup wiring

**File**: `app/main.py`

**Changes**: In lifespan, if `settings.bilstm_artifact_path` is set, call `load_bilstm(settings.bilstm_artifact_path)` and attach to `app.state` (e.g. `app.state.bilstm_model`).

### Success Criteria

#### Automated Verification

- [ ] Unit test: with a minimal Keras model saved to temp path, loader returns a wrapper that accepts list of strings and returns list of floats (may use mock tokenizer).
- [x] When path is unset, loader not called.
- [x] `python -m pytest tests/ -v` passes

#### Manual Verification

- [ ] With real Bi-LSTM artifact and tokenizer at path, app starts and Bi-LSTM is used in combination (Phase 5).

**Implementation Note**: If thesis does not provide a Bi-LSTM artifact or tokenizer, this phase can be skipped or stubbed; hybrid will still work with DistilBERT + phase6_merged.

---

## Phase 5: Ensemble Combination and Prediction API

### Overview

Implement soft-voting combination: for each job post, collect scam probabilities from DistilBERT and from any loaded traditional ML and Bi-LSTM models; average them; threshold with `confidence_threshold` to get final prediction. Wire the prediction service to use this when at least one extra model is loaded (hybrid mode), else keep current single-model behavior.

### Changes Required

#### 1. Combiner

**File**: `app/model.py` or `app/ensemble.py`

**Changes**:

- Implement `combine_soft_voting(probabilities_per_model: List[List[float]]) -> List[float]`: for each item, average the probabilities across models, return list of averaged scam probabilities.
- Ensure all model outputs are aligned (same length list, same order as input texts).

#### 2. Prediction service

**File**: `app/services/prediction.py`

**Changes**:

- When `app.state` has only DistilBERT (no phase6, no bilstm): keep current flow (preprocess → `model.predict` / `model.predict_batch`).
- When hybrid is enabled (phase6 and/or bilstm loaded):
  - Preprocess as today to get `combined_text` (and optional structural features if traditional ML needs them).
  - Run DistilBERT, and if present phase6 and bilstm, on the same inputs.
  - Collect scam probabilities from each; run `combine_soft_voting`; build `PredictResponse` from averaged probability and threshold. Keep `warning_signals` from preprocessing unchanged.

#### 3. Request context

**File**: `app/main.py`

**Changes**: Pass request/app state into prediction service so it can see which models are loaded (or inject a small "predictor" object that encapsulates single vs hybrid).

### Success Criteria

#### Automated Verification

- [x] Unit tests: `combine_soft_voting` with 1, 2, 3 model outputs; correctness of average and thresholding.
- [x] Integration test: with dummy_model only, predict response unchanged; with dummy_model + dummy phase6 (fixed probabilities), response reflects averaging.
- [x] `python -m pytest tests/ -v` passes

#### Manual Verification

- [ ] With only DistilBERT configured, behavior matches current app (no regression).
- [ ] With DistilBERT + phase6_merged (and optionally Bi-LSTM), predictions differ from DistilBERT-only in a plausible way (e.g. more conservative or different edge cases).

**Implementation Note**: After Phase 5, full hybrid path is in place; Phase 6 adds tests and docs.

---

## Phase 6: Health, Tests, and Documentation

### Overview

Expose hybrid status in health (e.g. which components are loaded), add tests for hybrid and checkpoint default, and document hybrid setup and checkpoint 5412.

### Changes Required

#### 1. Health response

**File**: `app/schemas.py`

**Changes**: Extend `HealthResponse` with optional fields, e.g. `hybrid_enabled: bool`, `models_loaded: List[str]` (e.g. `["distilbert", "phase6_merged"]`), or keep minimal (e.g. only `model_loaded: bool` and `model_name`). Prefer not breaking existing clients; add optional fields only.

**File**: `app/main.py`

**Changes**: In `GET /health`, set new fields from `app.state` (which models are non-None).

#### 2. Tests

**Files**: `tests/test_ensemble.py` (or under existing test files)

**Changes**:

- Test soft-voting combiner with 1, 2, 3 inputs.
- Test prediction service with mock hybrid (DistilBERT + mock phase6) and assert response shape and that probability is averaged.
- Test that when phase6 path is set but file missing, app still starts and runs DistilBERT-only (no crash).

#### 3. Documentation

**File**: `README.md`

**Changes**:

- Under "Running with the real model" and "Checkpoint", state that default checkpoint is **5412** (final training step).
- Add section "Hybrid model (optional)": describe `JOBSENTRY_PHASE6_MERGED_PATH` and `JOBSENTRY_BILSTM_ARTIFACT_PATH`; when both unset, app uses DistilBERT only; when set, app loads those artifacts and combines predictions via soft voting. Note that artifact format must match (joblib pipeline for phase6, Keras for Bi-LSTM) and that thesis repo may document exact paths and formats.

**File**: `.env.example`

**Changes**: Add commented-out lines for `JOBSENTRY_PHASE6_MERGED_PATH`, `JOBSENTRY_BILSTM_ARTIFACT_PATH`, and `JOBSENTRY_HYBRID_COMBINATION=soft_voting`.

### Success Criteria

#### Automated Verification

- [x] All tests pass: `python -m pytest tests/ -v`
- [x] Lint passes on new/edited files

#### Manual Verification

- [ ] GET /health shows hybrid status when hybrid artifacts are loaded.
- [ ] README and .env.example give enough guidance to enable hybrid when artifacts are available.

---

## Testing Strategy

### Unit Tests

- Config: default checkpoint path is 5412; optional hybrid paths load from env.
- Traditional ML loader: load from temp joblib file; predict_proba returns list of floats.
- Bi-LSTM loader: load from temp Keras file; predict_proba with mock tokenizer.
- Soft voting: one model (identity); two/three models (average); threshold boundary.

### Integration Tests

- Predict and batch-predict with only DistilBERT (fixture): same behavior as current tests.
- Predict with DistilBERT + mock phase6: response has averaged probability and correct prediction label.
- Startup: phase6 path set but file missing → app starts, hybrid disabled for that component.

### Manual Testing Steps

1. Run with default config (no env): verify checkpoint-5412 is used when path exists.
2. Run with `JOBSENTRY_PHASE6_MERGED_PATH` pointing to a real joblib pipeline: verify no crash and predictions use ensemble.
3. Run with only DistilBERT: verify no regression in prediction or latency.

## Performance Considerations

- Hybrid adds one or two extra model forwards per request; monitor latency (target remains &lt;2 s per prediction). Batch endpoint should still benefit from batching DistilBERT; phase6 and Bi-LSTM can be run in sequence or parallel as needed.
- Lazy-loading of optional artifacts is acceptable; prefer loading at startup so first request is not slow.

## Migration Notes

- **Config**: Existing deployments using default will start using checkpoint-5412 after deploy; ensure that path exists or set `JOBSENTRY_MODEL_ARTIFACT_PATH` to a valid checkpoint directory.
- **Hybrid**: No migration; hybrid is opt-in via new env vars. Unset = current behavior.

## References

- Research: `cursor/project/research/2025-03-09-hybrid-model-and-checkpoint.md`
- Implementation Plan 1 (hybrid definition): `cursor/project/notes/Implementation Plan 1.md` §3.3
- Current config and loader: `app/config.py`, `app/model.py`
- Thesis-trained model plan: `cursor/project/plan/2025-03-08-thesis-trained-model-remaining-work.md`
