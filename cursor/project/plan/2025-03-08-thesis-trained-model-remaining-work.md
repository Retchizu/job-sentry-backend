# Thesis-Trained Model: Remaining Work Implementation Plan

## Overview

Complete the remaining work after the JobSentry backend implementation and research: (1) make the thesis checkpoint runnable locally and document how, (2) perform manual verification with the real model, (3) document deployment and artifact handling, and (4) optionally document the chosen checkpoint. The backend already loads the thesis-trained DistilBERT when `model.safetensors` exists at the configured path; this plan closes the verification and ops/docs gaps.

## Current State Analysis

- **Backend**: Fully implemented (config, model loading with LayerNorm remap, `/predict`, `/batch-predict`, `/health`, 28 tests). See `cursor/project/implementation/2025-03-08-NA-jobsentry-backend-implementation.md`.
- **Research**: Confirms we are set up to use the thesis model; we use it at runtime only when `model.safetensors` exists at `JOBSENTRY_MODEL_ARTIFACT_PATH` (default: `thesis-scam-job-post/models/distilbert_run/checkpoint-1566`). See `cursor/project/research/2025-03-08-thesis-trained-model-usage.md`.
- **Gaps**: No `model.safetensors` in this repo; manual verification is pending; deployment/artifact story and checkpoint documentation are not written.

### Key Discoveries

- `app/config.py:6` – default `model_artifact_path`; `app/config.py:12` – `JOBSENTRY_MODEL_ARTIFACT_PATH` override.
- `app/model.py:55-93` – `load_model()`: resolves path, loads `model.safetensors` if present, else falls back to untrained with a warning.
- `.env.example` documents `JOBSENTRY_MODEL_ARTIFACT_PATH`; `README.md` has no section on the model artifact or running with the real checkpoint.

## Desired End State

1. **Local run**: Clear steps (in README or a dedicated doc) for obtaining the checkpoint and running the app with the real model (relative path or env override).
2. **Manual verification**: Completed checklist — app starts with artifact, `/health` shows `model_loaded: true`, `/predict` and `/batch-predict` return sensible results, p95 latency &lt;2 s measured and recorded.
3. **Deployment/docs**: README (or deployment doc) explains how production gets the checkpoint (`JOBSENTRY_MODEL_ARTIFACT_PATH`), where to place the file/dir, and that the app runs in degraded mode without it.
4. **Optional**: Short note or link on which checkpoint is used (e.g. checkpoint-1566) and, if available from thesis, metrics/source.

### Verification

- Manual verification checklist is executed and results recorded (in implementation summary or a short verification note).
- README (or linked doc) contains a "Model artifact" or "Running with the real model" section and a "Deployment" section describing the artifact path.
- No open questions in the plan; all decisions (e.g. "document only" vs "bundle in image") are made and documented.

## What We're NOT Doing

- Adding the `model.safetensors` file to this repository (binary artifact remains in thesis project or a separate store).
- Implementing Dockerfile or CI/CD in this plan (documentation only for how to point at the artifact).
- New automated tests that require the real artifact (existing 28 tests with dummy model remain sufficient).
- Changes to thesis-scam-job-post repo.

## Implementation Approach

Four phases: (1) document and enable local run with checkpoint, (2) execute manual verification and record results, (3) add deployment/artifact documentation, (4) optional checkpoint summary. Phases 1 and 3 are documentation edits; Phase 2 is manual steps; Phase 4 is optional and may be a short doc or README subsection.

---

## Phase 1: Local Run with Thesis Checkpoint

### Overview

Document how to get the thesis checkpoint available and run the app so that the real model is loaded. No code changes required unless we add a single log line or health field indicating whether the fine-tuned artifact was loaded (current behaviour: load fine-tuned vs base is already logged and reflected in `model_loaded`).

### Changes Required

#### 1. README (or new doc) — "Running with the real model"

**File**: `README.md` (or `docs/running-with-real-model.md`)

**Content to add**:

- **Prerequisites**: Checkpoint directory must contain `model.safetensors` (e.g. from thesis-scam-job-post: `models/distilbert_run/checkpoint-1566/`).
- **Option A — Relative path**: Clone or copy the thesis project so the checkpoint is at `thesis-scam-job-post/models/distilbert_run/checkpoint-1566` relative to the process working directory (e.g. repo root). No env change needed.
- **Option B — Override path**: Set `JOBSENTRY_MODEL_ARTIFACT_PATH` to the absolute (or relative) path of the checkpoint directory that contains `model.safetensors`. Example: `export JOBSENTRY_MODEL_ARTIFACT_PATH=/path/to/checkpoint-1566`.
- **Verify**: Start app with `uvicorn app.main:app --reload`; check logs for "Loading fine-tuned model from ..." (not "No model.safetensors found"); call `GET /health` and confirm `model_loaded: true`.

### Success Criteria

#### Automated Verification

- [x] No new lint/format errors in modified files.
- [x] Existing tests still pass: `python -m pytest tests/ -v`.

#### Manual Verification

- [ ] Following the new doc, a reader can place the checkpoint and start the app with the real model.
- [ ] Logs and `/health` confirm the fine-tuned model is loaded.

**Implementation Note**: After completing this phase and automated verification passes, pause for manual confirmation that the steps work before proceeding to Phase 2.

---

## Phase 2: Manual Verification with Real Model

### Overview

Execute end-to-end verification with the thesis checkpoint loaded: health, single and batch predict, and latency. Record results so the implementation is marked fully verified.

### Changes Required

#### 1. Manual test execution

- Ensure the app is running with the real checkpoint (Phase 1).
- **Health**: `GET /health` → expect `status: "ok"`, `model_loaded: true`, `model_name` present.
- **Single predict**: `POST /predict` with a sample job post (e.g. minimal valid body with `job_title`, `job_desc`). Confirm response has `prediction`, `confidence`, `scam_probability`, `warning_signals`; sanity-check that a clearly legitimate post is not labeled "scam" with very high confidence (or vice versa for an obvious scam).
- **Batch predict**: `POST /batch-predict` with 5–10 job posts. Confirm `count` and per-item shape; spot-check one or two results.
- **Latency**: Send at least 10–20 requests to `POST /predict` (single job post each), record response times, compute p95. Target: p95 &lt; 2 seconds per prediction.

#### 2. Record results

**File**: `cursor/project/implementation/2025-03-08-NA-jobsentry-backend-implementation.md` (or a new `cursor/project/implementation/2025-03-08-thesis-model-manual-verification.md`)

**Content**: Update the "Manual (pending)" section to "Manual (done)" and add:

- Date and environment (OS, Python version, CPU/GPU if relevant).
- Outcome: health ok, predict/batch-predict behaved as expected.
- p95 latency (e.g. "p95 &lt; 2 s" or actual value) and how it was measured (e.g. 20 sequential requests, or a one-liner script).

### Success Criteria

#### Automated Verification

- [x] Implementation summary (or verification note) is updated in version control; no automated tests are required for this phase.

#### Manual Verification

- [x] All manual steps above were performed.
- [x] `/health` showed `model_loaded: true` when artifact was present.
- [x] Predictions were plausible for the sample inputs.
- [x] p95 latency was measured and is &lt; 2 s (or documented exception with reason).
- [x] Results are recorded in the implementation summary or verification doc.

**Implementation Note**: After completing this phase, pause for confirmation that manual verification is accepted before proceeding to Phase 3.

---

## Phase 3: Deployment and Artifact Documentation

### Overview

Document how deployment should provide the model artifact so production can run with the thesis-trained model. No code or deployment automation in this plan — documentation only.

### Changes Required

#### 1. README or deployment doc

**File**: `README.md` (and optionally `docs/deployment.md` if the team prefers a separate doc)

**Content to add** (e.g. under "Deployment" or "Model artifact"):

- **Production**: The app expects the checkpoint directory (containing `model.safetensors`) at the path given by `JOBSENTRY_MODEL_ARTIFACT_PATH`. Options: (1) Mount a volume or bind-mount that directory into the container/filesystem at a known path and set `JOBSENTRY_MODEL_ARTIFACT_PATH` to that path; (2) Copy the checkpoint into the image at a fixed path and set `JOBSENTRY_MODEL_ARTIFACT_PATH` accordingly; (3) Use a shared filesystem or artifact store and set the env to that path.
- **Degraded mode**: If the path is missing or does not contain `model.safetensors`, the app starts with the base (untrained) DistilBERT and logs a warning; `GET /health` returns `model_loaded: true` but predictions are not meaningful. Ensure the artifact is available in production for real use.
- **Security**: Do not commit `model.safetensors` to the app repo; obtain it from the thesis project or a secure artifact store.

### Success Criteria

#### Automated Verification

- [x] No new lint/format errors in modified files.

#### Manual Verification

- [x] A deployer can follow the doc to understand how to supply the checkpoint and what happens if it is missing.

---

## Phase 4 (Optional): Checkpoint Summary

### Overview

If the thesis project (or another source) provides a summary of trained runs/checkpoints (e.g. metrics for checkpoint-1566), add a short note or link in this repo so the chosen checkpoint is justified and reproducible.

### Changes Required

#### 1. README or project doc

**File**: `README.md` or `cursor/project/research/2025-03-08-thesis-trained-model-usage.md` (or a short `cursor/project/notes/thesis-checkpoint-summary.md`)

**Content** (if information is available):

- Which checkpoint is used by default: e.g. `thesis-scam-job-post/models/distilbert_run/checkpoint-1566`.
- Link or reference to the thesis summary (e.g. "See thesis-scam-job-post repo, section X" or "Metrics: accuracy X%, F1 Y").
- If no summary exists, add a one-line note: "Default checkpoint is checkpoint-1566; no formal summary of trained models is in this repo."

### Success Criteria

#### Manual Verification

- [x] Either the checkpoint is documented with source/metrics, or the absence of a summary is stated so future readers know the situation.

---

## Testing Strategy

- **Phase 1 & 3**: No new automated tests; existing `pytest` suite remains the regression baseline.
- **Phase 2**: Manual only; results recorded in implementation summary or verification note.
- **Phase 4**: Documentation only.

## Performance Considerations

- p95 &lt; 2 s per prediction is verified manually in Phase 2 with the real model; no code changes in this plan for performance.

## Migration Notes

- Not applicable; no schema or data migrations. Only documentation and manual verification.

## References

- Research (trained model usage): `cursor/project/research/2025-03-08-thesis-trained-model-usage.md`
- Research (integration): `cursor/project/research/2025-03-08-thesis-scam-job-post-models-integration.md`
- Implementation summary: `cursor/project/implementation/2025-03-08-NA-jobsentry-backend-implementation.md`
- Original backend plan: `cursor/project/plan/2025-03-08-jobsentry-backend-implementation.md`
- Config and loader: `app/config.py`, `app/model.py:55-93`
