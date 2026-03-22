---
date: 2025-03-22T12:00:00+00:00
researcher: Cursor Agent
git_commit: a41aea8ccf37a2dc4a249e6d66f96daf13046f52
branch: main
repository: job-sentry-backend
topic: "What is missing in the codebase after fusing BiLSTM and DistilBERT (phase6_hybrid_fused + phase6_fused artifacts)"
tags: [research, codebase, phase6_fused, HybridFusedClassifier, BiLSTM, DistilBERT, FastAPI, inference]
status: complete
last_updated: 2025-03-22
last_updated_by: Cursor Agent
---

# Research: Phase 6 fused model vs. current codebase

**Date**: 2025-03-22T12:00:00+00:00  
**Researcher**: Cursor Agent  
**Git Commit**: a41aea8ccf37a2dc4a249e6d66f96daf13046f52  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

What is missing in our codebase, given that BiLSTM and DistilBERT were fused in `artifacts/ipynb/phase6_hybrid_fused.ipynb` and artifacts live under `artifacts/models/phase6_fused`?

## Summary

- **Training and export are defined in the notebook** (`HybridFusedClassifier`, dual tokenization, `fused_meta.json`, tokenizer, `word_index.json`, `config.json` with custom fused keys). Internal project notes mark **production inference (Phase 5) as not done** and name a concrete follow-up (`app/fused_model.py` loader pattern).

- **The runnable Python application layer is absent in the working tree**: the only tracked `.py` file is `app/main.py`, and that file is **empty**. Git status at research time shows **deleted** modules that previously implemented the API and DistilBERT loading (`app/model.py`, `app/config.py`, `app/schemas.py`, `app/ensemble.py`, and others).

- **Under `artifacts/models/phase6_fused/` there is no `model.safetensors` file** in this workspace, while the notebook’s export section writes that file. The directory **does** contain PyTorch checkpoints (`checkpoints/epoch_01.pt`, `epoch_03.pt`, `epoch_06.pt`) whose structure (per the notebook) includes a `model_state` key suitable for `load_state_dict`. Other fused artifacts present: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `word_index.json`, `fused_meta.json`, `fused_metrics.csv`, and plot PNGs.

- **Configuration for serving the fused model is not reflected in `.env.example`**, which still documents a **DistilBERT-only** artifact path plus optional **separate** hybrid pieces (merged joblib, Keras BiLSTM), not a single fused checkpoint directory.

- **No `tests/` Python tests** were found under the repository root at research time.

- **`hack/spec_metadata.sh`** referenced by the research workflow command **does not exist** in this repository; metadata in this document was collected via `git` and filesystem inspection.

## Detailed Findings

### Notebook contract (`artifacts/ipynb/phase6_hybrid_fused.ipynb`)

The notebook documents and implements:

- **`HybridFusedClassifier`**: DistilBERT branch (mean-pool over non-pad tokens → 768-d), BiLSTM branch (word-level vocab → embedding → bidirectional LSTM → 128-d from last hidden states), fusion MLP to 2-class logits.
- **Inputs at inference**: `input_ids`, `attention_mask` (DistilBERT, `max_len_bert` = 256), `lstm_ids` (word ids padded to `max_len_bilstm` = 400), with word tokenization `re.findall(r"\w+", text.lower())` and `word2idx` / OOV handling matching training.
- **Export targets** under `phase6_fused/`: `model.safetensors`, extended `config.json`, tokenizer files, `word_index.json`, `fused_meta.json`, `fused_metrics.csv`, plus optional `checkpoints/epoch_XX.pt`.

### Artifact directory state (`artifacts/models/phase6_fused/`)

Present (non-exhaustive): `config.json`, `tokenizer.json`, `tokenizer_config.json`, `word_index.json`, `fused_meta.json`, `fused_metrics.csv`, `training_curves.png`, `confusion_matrices.png`, `checkpoints/*.pt`.

Not present in workspace (relative to notebook §12): **`model.safetensors`**. A repository-wide search finds `model.safetensors` under other model trees (e.g. `artifacts/models/distilbert_run/checkpoint-*`, `phase32_distilbert`, `phase42_distilbert`) but **not** under `phase6_fused/`.

`fused_meta.json` records hyperparameters and metrics (e.g. `max_len_bert` 256, `max_len_bilstm` 400, `vocab_size` 20000, `threshold` 0.5, `distilbert_model` `distilbert-base-uncased`).

### Application code (`app/`)

- **`app/main.py`**: empty file (no FastAPI app, routes, or imports).
- **Git status (snapshot at conversation start)**: `D` (deleted) on `app/__init__.py`, `app/bilstm.py`, `app/config.py`, `app/ensemble.py`, `app/model.py`, `app/preprocessing.py`, `app/schemas.py`, `app/traditional_ml.py` — so the previously documented DistilBERT loader, settings, schemas, and hybrid ensemble hooks are **not** in the tree as of that snapshot.

### Dependencies and environment

- **`requirements.txt`**: includes `torch`, `transformers`, `safetensors`, `fastapi`, `uvicorn`, etc., which are compatible with implementing a PyTorch fused loader; also lists `tensorflow` for optional Keras BiLSTM hybrid mode, which is a **different** integration path than the single PyTorch fused module.
- **`.env.example`**: documents `JOBSENTRY_MODEL_ARTIFACT_PATH` for a DistilBERT checkpoint directory and optional `JOBSENTRY_PHASE6_MERGED_PATH` / `JOBSENTRY_BILSTM_ARTIFACT_PATH` for a **multi-model** hybrid — **no variable** for `artifacts/models/phase6_fused` or a fused safetensors path.

### Project notes (historical / planning)

- **`cursor/project/implementation/2025-03-21-NA-hybrid-bilstm-distilbert-fused-training.md`**: explicitly lists Phase 5 **“Production inference path”** as unchecked and describes a follow-up loader: tokenizer from `phase6_fused/`, `word_index.json`, reconstruct from `fused_meta.json`, load `model.safetensors` via `safetensors.torch.load_file`, and a `predict_proba`-style API referencing `app/ensemble.py` and `ScamPredictor` — those referenced app files are **deleted** in the current snapshot.

- Older research docs under `cursor/project/research/` describe **`app/model.py`** DistilBERT-only loading; they reflect a **prior** codebase layout and are **not** aligned with an empty `app/main.py` and absent `app/model.py`.

## Code References

- `artifacts/ipynb/phase6_hybrid_fused.ipynb` — `HybridFusedClassifier`, training loop, export §12, collate/tokenization behavior.
- `artifacts/models/phase6_fused/config.json` — DistilBERT base config plus fused keys (`max_len_bert`, `max_len_bilstm`, `vocab_size_bilstm`, `embed_dim`, `lstm_hidden`, `fusion_hidden`, `architectures`: `HybridFusedClassifier`).
- `artifacts/models/phase6_fused/fused_meta.json` — runtime hyperparameters and threshold for mirroring training-time preprocessing.
- `app/main.py` — empty (no HTTP API).
- `requirements.txt` — ML + web stack dependencies as listed.
- `.env.example` — DistilBERT and legacy hybrid env vars only.
- `cursor/project/implementation/2025-03-21-NA-hybrid-bilstm-distilbert-fused-training.md` — Phase 5 inference follow-up described; Phase 5 checkbox unchecked.

## Architecture Documentation (as-is)

- **Fused model**: single PyTorch module combining DistilBERT and BiLSTM; weights exportable as full `state_dict` in `model.safetensors` or inside epoch `.pt` checkpoints.
- **Serving**: not implemented in the current Python tree; prior design docs assumed DistilBERT-only or separate Keras/joblib components, not this fused checkpoint layout.

## Historical Context (from `cursor/project/`)

- `cursor/project/implementation/2025-03-21-NA-hybrid-bilstm-distilbert-fused-training.md` — training notebook delivered; production inference deferred; proposed loader steps and `ScamPredictor` protocol reference.
- `cursor/project/plan/2025-03-21-hybrid-bilstm-distilbert-fused-training.md` — plan for fused training and serving-compatible artifacts.
- `cursor/project/research/2025-03-09-hybrid-model-and-checkpoint.md` — documents earlier backend behavior (DistilBERT-only, no ensemble); codebase has since diverged (app modules removed).

## Related Research

- [2025-03-09-hybrid-model-and-checkpoint.md](2025-03-09-hybrid-model-and-checkpoint.md)
- [2025-03-08-thesis-trained-model-usage.md](2025-03-08-thesis-trained-model-usage.md)

## Open Questions

- Whether `model.safetensors` for `phase6_fused` will be placed under `artifacts/models/phase6_fused/` (or another path) in deployments, or whether inference will load **`model_state` from a `.pt` checkpoint** instead.
- Intended restoration of `app/*` modules vs. a new minimal app surface for fused inference only.

## Note on metadata script

The research command specified `hack/spec_metadata.sh`; that path **does not exist** in this repository. Frontmatter fields were populated from `git rev-parse HEAD`, `git branch --show-current`, and filesystem listing of `artifacts/models/phase6_fused/`.
