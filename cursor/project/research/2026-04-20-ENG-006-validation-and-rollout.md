---
date: "2026-04-20T12:00:00-07:00"
researcher: unspecified
git_commit: 26c01727e996da4fcc64221713a2f75fad464f18
branch: main
repository: job-sentry-backend
topic: "TICKET-006: validation, comparison, and rollout (as implemented in codebase)"
tags: [research, codebase, TICKET-006, validation, rollout, multiclass, inference, phase6_fused]
status: complete
last_updated: "2026-04-20"
last_updated_by: unspecified
metadata_note: "Repository has no hack/spec_metadata.sh; git_commit, branch, and timestamps were collected via git and date commands at research time."
---

# Research: TICKET-006 — validation, comparison, and rollout

**Date**: 2026-04-20T12:00:00-07:00  
**Researcher**: unspecified (no `researcher` field in `cursor/project/notes/`)  
**Git Commit**: `26c01727e996da4fcc64221713a2f75fad464f18`  
**Branch**: `main`  
**Repository**: `job-sentry-backend` (remote: `Retchizu/job-sentry-backend`)

## Research Question

What exists in the codebase today that corresponds to [`cursor/project/tickets/TICKET-006-validation-and-rollout.md`](../tickets/TICKET-006-validation-and-rollout.md): validating model quality, comparing binary vs multiclass behavior, error analysis, artifact versioning, and documenting impact on downstream consumers of predictions?

## Summary

The ticket defines a **process-oriented** gate: benchmark comparison, inspection of false positives/negatives for **warning** and **fraud**, notebook or markdown deliverables, versioned artifacts, and migration notes for clients that assumed **binary** output.

In the **application code**, inference is implemented as a **native 3-class softmax** path end-to-end (`POST /predict` → `FusedScamPredictor` → `class_from_softmax_triple`). A separate **`map_binary_to_risk`** helper exists for mapping a scalar **P(scam)** plus heuristics to three-way labels; it is **not** used by the default `/predict` handler but is documented in `README.md` as an optional policy for binary-only workflows.

**Model-quality validation** in-repo is represented by **training/evaluation notebooks** under `artifacts/ipynb/` (especially `phase6_hybrid_fused.ipynb` for val/test reports and confusion matrices) and **exported artifacts** under `artifacts/models/phase6_fused/` (for example `fused_metrics.csv`, `confusion_matrices.png`, `training_curves.png`). **Automated tests** assert the HTTP contract and predictor mapping behavior; they do not embed a full benchmark comparing an old binary stack to the new multiclass model on a fixed subset.

**Downstream integration** is documented primarily in **`README.md`**: response field semantics, softmax vs one-hot expectations, deterministic HTTP errors, health behavior, environment variables, and a short **migration** paragraph for legacy field names and probability interpretation. A **FastAPI checklist** section lists local setup steps. The ticket index places TICKET-006 as the **last step** on the critical path before production rollout.

## Detailed Findings

### Ticket scope and roadmap position

[`cursor/project/tickets/TICKET-006-validation-and-rollout.md`](../tickets/TICKET-006-validation-and-rollout.md) states objectives: validate quality and safe rollout from binary to multiclass; scope includes shared-benchmark comparison, FP/FN inspection for warning and fraud, error-analysis examples in a notebook, finalized artifacts/version naming, and migration notes for binary consumers. Deliverables include a final validation section (notebook or markdown) and a release checklist for deployment/inference consumers.

[`cursor/project/tickets/README.md`](../tickets/README.md) lists TICKET-006 as step 8 (“final gate before production rollout”) and orders the critical path: `TICKET-007` → `TICKET-001` → `TICKET-002` → `TICKET-003` → `TICKET-004` → `TICKET-005` → `TICKET-006`.

### Inference contract and multiclass vs binary bridging

- **`PredictResponse`** documents parallel arrays per post: integer `predicted_class` (0/1/2), string `predicted_label`, three softmax probabilities, `confidence` (max of the three), and heuristic `warnings`. See [`app/schemas.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L62-L92).
- **`predict`** loads texts, runs `compute_warnings`, enforces `max_batch_size`, calls `predictor.predict_full`, and builds `PredictResponse`. See [`app/main.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L97-L153).
- **`class_from_softmax_triple`** maps `(p_legit, p_warning, p_fraud)` with argmax and `confidence = max(...)`. **`map_binary_to_risk`** maps scalar `p_scam` and thresholds to a one-hot triple (binary-era bridge); both live in [`app/risk_labels.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py#L14-L64).
- **`README.md`** describes default **argmax** on the 3-class softmax, documents **`map_binary_to_risk`** for scalar P(scam) workflows, lists **`JOBSENTRY_WARN_THRESHOLD` / `JOBSENTRY_FRAUD_THRESHOLD`** as used by those helpers (not by default softmax class selection), and includes a **Migration** subsection for legacy response fields and softmax vs one-hot expectations. See [`README.md`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/README.md#L62-L108) and the FastAPI checklist at [lines 134–140](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/README.md#L134-L140).

### Loader, artifact compatibility, and versioning

- [`app/fused_loader.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L1-L47) documents that **legacy 2-class parallel-fusion** checkpoints (word BiLSTM branch + `word_index.json`) are **not** compatible with the current sequential 3-class loader. **`_validate_risk_class_labels`** logs warnings if `fused_meta.json`’s `risk_class_labels` order disagrees with the expected `("legit", "warning", "fraud")` order.
- Weight resolution prefers `model.safetensors`, then optional env checkpoint override, then highest `epoch_NN.pt` under `checkpoints/` (implementation continues in the same module after line 50).

### Heuristic warnings (complements model scores)

[`app/predict_warnings.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/predict_warnings.py#L1-L71) defines regex-based codes (for example `upfront_payment`, `off_platform_contact`) returned in parallel with model outputs; these codes are also inputs to **`map_binary_to_risk`**’s warning path when that function is used.

### Evaluation artifacts and notebooks (offline validation)

- Under **`artifacts/models/phase6_fused/`** (present in the workspace): `fused_metrics.csv`, `confusion_matrices.png`, `training_curves.png`, `fused_meta.json`, `config.json`, tokenizer files, `model.safetensors`, and `checkpoints/epoch_01.pt` … `epoch_10.pt`.
- **`artifacts/ipynb/phase6_hybrid_fused.ipynb`** is the primary Phase 6 fused training notebook referenced in prior research; it contains sections for training curves, load-best evaluation, classification reports, confusion matrices, save/load verification, inference sanity, and baseline comparison (cell-level line numbers vary with notebook edits; the notebook file is the source of truth).
- Other notebooks under **`artifacts/ipynb/`** (`phase6_deep_learning.ipynb`, `main_scam_detection.ipynb`, `dataset2_scam_detection.ipynb`, `phase6_scam_detection.ipynb`) contain additional validation, confusion matrix, ROC, and error-analysis style sections; **`main_scam_detection.ipynb`** and **`dataset2_scam_detection.ipynb`** include explicit error-analysis headings in the exploratory pipeline narratives.

### Automated tests (contract, not full benchmark suite)

- [`tests/test_api.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_api.py) exercises `/health`, `/predict` 503 when no model, and `/predict` 200 with an injected fake predictor using 3-class softmax triples.
- [`tests/test_fused_predictor.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_fused_predictor.py), [`tests/test_fused_loader.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_fused_loader.py), [`tests/test_risk_labels.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_risk_labels.py) (if present on branch), and related tests cover predictor math, loading, and label helpers.

## Code References

- `cursor/project/tickets/TICKET-006-validation-and-rollout.md` — Ticket acceptance criteria and deliverables for validation and rollout.
- `app/main.py:97-153` — `predict` handler: batching, warnings, `predict_full`, response assembly.
- `app/schemas.py:62-92` — `PredictResponse` field definitions (3-class contract).
- `app/risk_labels.py:14-64` — `class_from_softmax_triple` and `map_binary_to_risk`.
- `app/fused_loader.py:1-47` — Incompatibility note for 2-class parallel fusion; `risk_class_labels` validation.
- `README.md:26-108` — Phase 6 fused setup, response semantics, thresholds, migration, errors.
- `README.md:134-140` — FastAPI checklist (local venv and artifact path unchecked by default).
- `artifacts/models/phase6_fused/fused_metrics.csv` — Per-epoch or run metrics (CSV artifact).
- `artifacts/models/phase6_fused/confusion_matrices.png` — Saved confusion matrix visualization.

## Architecture Documentation

- **Serving path**: HTTP → Pydantic `PredictRequest` → combined text per post → `FusedScamPredictor` (DistilBERT tokenizer + `HybridFusedClassifier` + softmax) → `RiskPrediction` rows → `PredictResponse`.
- **Binary-to-multiclass documentation**: README explains softmax outputs and optional `map_binary_to_risk`; loader docstring explains architectural incompatibility with older 2-class fusion checkpoints.
- **Rollout-related configuration**: `JOBSENTRY_PHASE6_FUSED_DIR`, optional `JOBSENTRY_PHASE6_FUSED_CHECKPOINT`, `JOBSENTRY_DEVICE`, `JOBSENTRY_MAX_BATCH_SIZE`; degraded mode when the fused dir is unset (`README.md`).

## Historical Context (from `cursor/project/`)

- `cursor/project/tickets/README.md` — TICKET-006 placement on the critical path and “backward compatibility by documenting label mapping and output contract.”
- `cursor/project/research/2026-04-19-TICKET-004-training-and-evaluation-updates.md` — Training/evaluation behavior (notebook-driven) related to prior evaluation tickets.
- `cursor/project/research/2026-04-19-TICKET-005-inference-contract-and-serving.md` — Inference API contract research.
- `cursor/project/research/2026-03-22-phase6-fused-production-inference-next-steps.md` — Earlier notes on production inference and rollout topics.
- `cursor/project/research/2026-03-23-phase6-fused-only-artifact-usage.md` — Artifact usage for fused inference.
- `cursor/project/notes/dataset2-linguistic-feature-set.md`, `Implementation Plan 1.md`, `Implementation Plan 2.md` — Broader project notes (not TICKET-006-specific).

## Related Research

- [`2026-04-19-TICKET-004-training-and-evaluation-updates.md`](2026-04-19-TICKET-004-training-and-evaluation-updates.md)
- [`2026-04-19-TICKET-005-inference-contract-and-serving.md`](2026-04-19-TICKET-005-inference-contract-and-serving.md)
- [`2026-03-22-phase6-fused-production-inference-next-steps.md`](2026-03-22-phase6-fused-production-inference-next-steps.md)
- [`2026-03-23-phase6-fused-only-artifact-usage.md`](2026-03-23-phase6-fused-only-artifact-usage.md)

## Open Questions

- Whether a **single automated script or test** compares “old binary” vs “new multiclass” on a **shared benchmark subset** is not present in the surveyed `app/` and `tests/` trees; that work appears to live in **notebooks and manual artifact review** per ticket wording.
- The ticket’s “evaluation summary with key strengths and risks” as a **standalone deliverable** may be represented by notebook sections and/or markdown not uniquely named `TICKET-006` in-repo; locating a dedicated companion markdown file would require searching for filenames or headings introduced after this research date.
