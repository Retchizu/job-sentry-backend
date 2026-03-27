---
date: 2026-03-23T16:55:00Z
researcher: job-sentry-backend research
git_commit: 174bbed692bc5932c7f73f2de7e01a9f4a0972ea
branch: main
repository: job-sentry-backend
topic: "Are we only using what is in artifacts/models/phase6_fused?"
tags: [research, codebase, phase6_fused, FusedScamPredictor, inference, artifacts]
status: complete
last_updated: 2026-03-23
last_updated_by: job-sentry-backend research
---

# Research: Are we only using what is in artifacts/models/phase6_fused?

**Date**: 2026-03-23T16:55:00Z  
**Researcher**: job-sentry-backend research  
**Git Commit**: `174bbed692bc5932c7f73f2de7e01a9f4a0972ea`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

Are we only using what is in `artifacts/models/phase6_fused`?

## Summary

The **FastAPI service** loads inference from **a single configurable artifact directory** set by `JOBSENTRY_PHASE6_FUSED_DIR`. The repository and `.env.example` use **`artifacts/models/phase6_fused` as the documented example path**, but the application does **not** hardcode that string; any directory with the expected fused layout works.

**Application Python code** does **not** load separate legacy stacks (for example `phase32_*`, `phase41_*`, `distilbert_run`, or `phase6_merged`) for `/predict`. Settings in `app/config.py` expose only `phase6_fused_dir`, `phase6_fused_checkpoint`, device, batch size, and threshold.

**Everything read for weights, tokenizer files, fused metadata, and word index** is resolved **under that one directory** (plus optional explicit checkpoint path). One nuance: `HybridFusedClassifier` constructs `DistilBertModel.from_pretrained(distilbert_name)` using the model id from `fused_meta.json` (default `distilbert-base-uncased`) **before** fused weights are loaded from disk; the Transformers library may use the **Hugging Face cache** for that base id during module construction, then `load_state_dict` applies the **fused** weights from `model.safetensors` or a `.pt` checkpoint in the artifact directory.

## Detailed Findings

### Configuration and entrypoint

- **`app/config.py`** defines `phase6_fused_dir` and optional `phase6_fused_checkpoint` with env prefix `JOBSENTRY_`. No other model artifact paths are defined for the current app.

### Startup loading

- **`app/main.py`** `_load_predictor` returns `None` if `phase6_fused_dir` is unset; otherwise it calls `FusedScamPredictor.from_artifact_dir(settings.phase6_fused_dir, checkpoint_override=settings.phase6_fused_checkpoint, ...)`. Health and `/predict` use only `app.state.predictor` as a `FusedScamPredictor`.

### What the fused loader reads from the artifact directory

- **`app/fused_loader.py`** `load_fused_artifacts`:
  - Requires `fused_meta.json` in the artifact directory.
  - Loads the word index from the file named by `word_index_file` in meta (default `word_index.json`) in the same directory.
  - Resolves weights via `resolve_weight_source`: prefers `model.safetensors` in the artifact dir; else `JOBSENTRY_PHASE6_FUSED_CHECKPOINT` if set; else the highest-numbered `checkpoints/epoch_*.pt` under the artifact dir.
  - Builds `HybridFusedClassifier` from numeric fields in `fused_meta.json` and `distilbert_model` (default `distilbert-base-uncased`).
  - Loads tokenizer with `DistilBertTokenizerFast.from_pretrained(str(artifact_dir))`, so tokenizer JSON/config under **that** directory are used.

### Model class and DistilBERT id

- **`app/hybrid_fused_model.py`** `HybridFusedClassifier.__init__` calls `DistilBertModel.from_pretrained(distilbert_name)`. The id comes from fused meta, not from a second on-disk model tree under `artifacts/models/` besides the fused artifact folder.

### Documentation vs other artifact trees

- **`.env.example`** states that legacy DistilBERT-only / hybrid ensemble variables are **not used by the current app** when the fused dir is set, and lists commented examples only.

### Tests

- **`tests/test_api.py`** optionally points `JOBSENTRY_PHASE6_FUSED_DIR` at `artifacts/models/phase6_fused` when those files exist for integration-style coverage.

## Code References

- `app/config.py:21-35` — `phase6_fused_dir`, `phase6_fused_checkpoint`, no other model dirs.
- `app/main.py:21-34` — Single loader path: `FusedScamPredictor.from_artifact_dir`.
- `app/fused_loader.py:34-58` — Weight resolution: safetensors, optional checkpoint override, or best `epoch_*.pt`.
- `app/fused_loader.py:87-149` — `load_fused_artifacts`: meta, word index, weights, tokenizer from `artifact_dir`.
- `app/hybrid_fused_model.py:57-69` — `DistilBertModel.from_pretrained(distilbert_name)` before fused `load_state_dict` in the loader.
- `.env.example:1-35` — Example fused path; legacy vars documented as unused by current app.

## Architecture Documentation

- **Single-artifact-dir inference**: One env-configured directory holds fused metadata, vocab, tokenizer files, optional `config.json` for Transformers, and fused weights (`model.safetensors` or PyTorch checkpoints).
- **No multi-model ensemble in `app/`**: Grep over Python sources shows no references to `phase32`, `phase41`, `phase42`, or `phase6_merged` in application modules; only fused-related settings and tests.

## Historical Context (from cursor/project/)

- `cursor/project/research/2026-03-22-phase6-fused-production-inference-next-steps.md` — Notes production inference via `JOBSENTRY_PHASE6_FUSED_DIR` and `app/main.py` lifespan.
- `cursor/project/research/2025-03-22-phase6-fused-vs-codebase-gaps.md` — Older snapshot notes; live codebase now includes `fused_loader`, `FusedScamPredictor`, and `model.safetensors` under `phase6_fused` in the workspace listing.

## Related Research

- `cursor/project/research/2026-03-22-salary-hours-structured-fields-phase6-fused.md` — Fused meta fields and text-only inputs.

## Open Questions

- **`hack/spec_metadata.sh`** was not present in the repository at research time; metadata was collected with `git` and `date` instead.

## Metadata note

Step 5 of the research command referenced `hack/spec_metadata.sh`; that path does not exist in this repository, so frontmatter was filled from `git rev-parse HEAD`, `git branch --show-current`, and filesystem inspection.
