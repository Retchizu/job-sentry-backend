---
date: 2026-04-18T04:10:13Z
researcher: riche
git_commit: 26c01727e996da4fcc64221713a2f75fad464f18
branch: main
repository: job-sentry-backend
topic: "TICKET-003: Sequential DistilBERT → BiLSTM fused model — codebase as-is"
tags: [research, codebase, HybridFusedClassifier, phase6_fused, DistilBERT, BiLSTM, TICKET-003]
status: complete
last_updated: 2026-04-18
last_updated_by: riche
metadata_note: "hack/spec_metadata.sh was not present in the repository; git hash, branch, and timestamps were gathered manually."
---

# Research: TICKET-003 — Sequential fused model update

**Date**: 2026-04-18T04:10:13Z  
**Researcher**: riche  
**Git Commit**: `26c01727e996da4fcc64221713a2f75fad464f18`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

What does the repository contain today with respect to [`cursor/project/tickets/TICKET-003-sequential-fused-model-update.md`](../tickets/TICKET-003-sequential-fused-model-update.md): sequential flow `Text → Tokenization → DistilBERT → contextual embeddings → BiLSTM → classifier`, masking-aware pooling after BiLSTM, `num_labels = 3`, freeze/unfreeze for DistilBERT, checkpoint behavior, and notebook alignment?

## Summary

The ticket describes a **sequential** architecture in which **BiLSTM consumes DistilBERT token embeddings** (`last_hidden_state`), followed by **masking-aware pooling after the BiLSTM** and a classifier with **three output logits**.

As of the referenced commit, the importable implementation in [`app/hybrid_fused_model.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py) defines `HybridFusedClassifier` as a **two-branch parallel** model: DistilBERT produces a **sentence vector** via **mean pooling** over `last_hidden_state`; a **separate** word-ID sequence is embedded with `nn.Embedding` and passed through **BiLSTM**; the two vectors are **concatenated** and fed to an MLP head. **Output logits have width 2** by default (`num_labels: int = 2`), matching [`artifacts/models/phase6_fused/fused_meta.json`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/models/phase6_fused/fused_meta.json) (`"num_labels": 2`). The training notebook [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb) mirrors the same parallel fusion pattern and **binary** `NUM_LABELS`, and implements a **freeze-then-unfreeze** schedule for DistilBERT via `FREEZE_BERT_EPOCHS` and `model.freeze_bert()` / `model.unfreeze_bert()`.

Loading and inference paths ([`app/fused_loader.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py), [`app/fused_predictor.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py)) reconstruct the module from `fused_meta.json`, load weights from `model.safetensors` or `checkpoints/epoch_*.pt`, and run forward passes that yield **`[batch_size, num_labels]`** with **`num_labels` from meta (default 2)**.

## Detailed Findings

### TICKET-003 scope (reference)

The ticket file specifies:

- Forward path: tokenization → DistilBERT → **contextual embeddings into BiLSTM** → classifier → output.
- BiLSTM input: DistilBERT **`last_hidden_state`** sequence.
- Pooling: **after BiLSTM**, masking-aware.
- Classifier: **`num_labels = 3`**.
- DistilBERT: configurable **freeze/unfreeze** schedule.
- Deliverables include **updated notebook** cells and **notes on tensor shapes**.

### `HybridFusedClassifier` in `app/hybrid_fused_model.py`

- **Class docstring** states binary fusion and documents inputs `input_ids`, `attention_mask`, `lstm_ids` and output logits `[B, NUM_LABELS=2]` ([`app/hybrid_fused_model.py:42-55`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L42-L55)).
- **`__init__`** builds `DistilBertModel`, `nn.Embedding(vocab_size, embed_dim)`, bidirectional `nn.LSTM` with `input_size=embed_dim` (not BERT hidden size), and a classifier whose first linear expects `bert_dim + lstm_out_dim` ([`app/hybrid_fused_model.py:57-88`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L57-L88)).
- **Freeze helpers**: `freeze_bert` and `unfreeze_bert` toggle `requires_grad` on all `self.bert` parameters ([`app/hybrid_fused_model.py:90-96`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L90-L96)).
- **`forward`**:
  - Runs `self.bert`, then **`_mean_pool` on `last_hidden_state`** using `attention_mask` → `bert_vec` shape `[B, 768]` for base DistilBERT ([`app/hybrid_fused_model.py:104-111`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L104-L111)).
  - Runs **`self.embedding(lstm_ids)`** then **`self.lstm`** on those embeddings; **final LSTM hidden states** are concatenated to `lstm_vec` `[B, 128]` (for default `lstm_hidden=64` bidirectional) ([`app/hybrid_fused_model.py:113-115`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L113-L115)).
  - **Concatenates** `bert_vec` and `lstm_vec`, then `self.classifier` → logits `[B, num_labels]` ([`app/hybrid_fused_model.py:117-119`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L117-L119)).
- **LSTM preprocessing** (`tokenize_words`, `_text_to_lstm_ids`, `texts_to_lstm_batch`) builds **word-level** ID tensors from `word2idx`; this path is independent of DistilBERT subword tokenization ([`app/hybrid_fused_model.py:13-39`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L13-L39)).

### Artifact loading and hyperparameters

- [`load_fused_artifacts`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L87-L149) reads `fused_meta.json`, loads `word_index.json`, resolves weights (`model.safetensors` preferred, else checkpoint override or highest `epoch_*.pt`), constructs `HybridFusedClassifier` with `num_labels=int(fused_meta.get("num_labels", 2))`, loads weights with `load_state_dict(..., strict=True)`, and loads `DistilBertTokenizerFast` from the artifact directory.
- On-disk [`artifacts/models/phase6_fused/fused_meta.json`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/models/phase6_fused/fused_meta.json) records **`num_labels`: 2** and standard fused hyperparameters (`max_len_bert` / `max_len_bilstm` are used by predictors from this meta in the inference stack; the JSON file itself lists vocab and model dims).

### Inference API path

- [`FusedScamPredictor._predict_proba_batch`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py) tokenizes texts with the DistilBERT tokenizer and builds `lstm_ids` via `texts_to_lstm_batch`, then calls `self.model(input_ids, attention_mask, lstm_ids)` and applies `softmax` on the last dimension. Callers treat the model as **binary** (e.g. probability of class index 1 as “scam”); this research does not expand on API mapping policies.

### Training notebook `artifacts/ipynb/phase6_hybrid_fused.ipynb`

- Defines **`FREEZE_BERT_EPOCHS`** and prints freeze configuration; after instantiation calls **`model.freeze_bert()`**; in the training loop, on **`epoch == FREEZE_BERT_EPOCHS + 1`** calls **`model.unfreeze_bert()`** and rebuilds the optimizer scheduler for remaining steps (grep-visible in the notebook JSON source).
- Embeds **`HybridFusedClassifier`** in the notebook with **`num_labels` tied to `NUM_LABELS` (2)** and the same parallel forward structure as `app/hybrid_fused_model.py` (see notebook cells under "## 6 · HybridFusedClassifier Module").

### Tests

- [`tests/test_hybrid_fused_model.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_hybrid_fused_model.py) patches `DistilBertModel.from_pretrained` with a fake BERT returning random `last_hidden_state`, runs `HybridFusedClassifier` with `num_labels=2`, and asserts output shape **`(b, 2)`** ([`tests/test_hybrid_fused_model.py:41-59`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_hybrid_fused_model.py#L41-L59)).
- [`tests/test_fused_loader.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_fused_loader.py) covers weight resolution priority for fused artifacts.

## Code References

- [`app/hybrid_fused_model.py:42-119`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L42-L119) — `HybridFusedClassifier` definition, freeze helpers, mean-pool, forward (parallel BERT + word-embedding BiLSTM, concat, MLP).
- [`app/fused_loader.py:124-136`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L124-L136) — `HybridFusedClassifier(...)` construction from `fused_meta.json` fields including `num_labels` default 2.
- [`artifacts/models/phase6_fused/fused_meta.json`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/models/phase6_fused/fused_meta.json) — Saved `num_labels: 2` and fused hyperparameters.
- [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb) — Training notebook: `HybridFusedClassifier`, `FREEZE_BERT_EPOCHS`, freeze/unfreeze calls, export cells.

## Architecture Documentation (as implemented today)

- **Fusion pattern**: **Parallel** encoders — DistilBERT sequence → **masked mean pool** → fixed-size vector; **independent** word-ID sequence → **embedding table** → BiLSTM → last hidden **concatenation** → MLP → logits.
- **Tensor shapes (typical defaults from meta)**: `last_hidden_state` `[B, 256, 768]`; `bert_vec` `[B, 768]`; `lstm_ids` `[B, 400]`; embedded `[B, 400, 100]`; BiLSTM output hidden states used as `[B, 128]`; fused `[B, 896]`; logits `[B, 2]`.
- **DistilBERT freeze schedule**: Exposed on the module via `freeze_bert` / `unfreeze_bert`; the **notebook** implements epoch-based freezing using those methods together with `FREEZE_BERT_EPOCHS`.

## Historical Context (from `cursor/project/`)

- [`cursor/project/tickets/TICKET-003-sequential-fused-model-update.md`](../tickets/TICKET-003-sequential-fused-model-update.md) — Ticket text for the sequential 3-class architecture and deliverables.
- [`cursor/project/research/2025-03-22-phase6-fused-vs-codebase-gaps.md`](2025-03-22-phase6-fused-vs-codebase-gaps.md) — Earlier research describing the fused notebook vs codebase; documents parallel DistilBERT + BiLSTM fusion and 2-class logits at time of writing.
- [`cursor/project/implementation/2025-03-21-NA-hybrid-bilstm-distilbert-fused-training.md`](../implementation/2025-03-21-NA-hybrid-bilstm-distilbert-fused-training.md) — Notes on notebook `HybridFusedClassifier` forward and `[B, 2]` logits.
- [`cursor/project/plan/2026-03-22-phase6-fused-production-inference.md`](../plan/2026-03-22-phase6-fused-production-inference.md) — Plan describing lifting `HybridFusedClassifier` from the notebook into `app/hybrid_fused_model.py` with structure aligned to the notebook (`self.bert`, `self.embedding`, `self.lstm`, `self.classifier`).

## Related Research

- [`cursor/project/research/2026-04-18-TICKET-008-backend-predict-deployment.md`](2026-04-18-TICKET-008-backend-predict-deployment.md) — Backend `/predict` deployment and fused predictor behavior.
- [`cursor/project/research/2026-03-22-salary-hours-structured-fields-phase6-fused.md`](2026-03-22-salary-hours-structured-fields-phase6-fused.md) — Phase6 fused model and schema touchpoints.

## Open Questions

- None required for “as-is” documentation; the ticket’s sequential 3-class design is **not** reflected in the current `HybridFusedClassifier` implementation and shipped `fused_meta.json` **num_labels** value at this commit.
