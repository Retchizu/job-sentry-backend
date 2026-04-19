---
date: 2026-04-19T15:29:50+08:00
researcher: riche
git_commit: 26c01727e996da4fcc64221713a2f75fad464f18
branch: main
repository: job-sentry-backend
topic: "TICKET-005: Inference contract and prediction outputs (as implemented vs ticket)"
tags: [research, codebase, TICKET-005, POST /predict, FusedScamPredictor, PredictResponse, risk_labels, phase6_fused]
status: complete
last_updated: 2026-04-19
last_updated_by: riche
metadata_note: "hack/spec_metadata.sh was not present in the repository; git hash, branch, and timestamps were gathered manually. The working tree had uncommitted modifications at research time; paths below reflect the workspace on disk."
---

# Research: TICKET-005 — Inference contract and prediction outputs

**Date**: 2026-04-19T15:29:50+08:00  
**Researcher**: riche  
**Git Commit**: `26c01727e996da4fcc64221713a2f75fad464f18`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

What exists in the repository today with respect to [`cursor/project/tickets/TICKET-005-inference-contract-and-serving.md`](../tickets/TICKET-005-inference-contract-and-serving.md): inference helper (raw text → tokenizer → forward pass), stable outputs (`predicted_class`, `predicted_label`, class probabilities), response contract (`legit_probability`, `warning_probability`, `fraud_probability`, `confidence`), one-text and batch inference, serialized class/label metadata, threshold policy notes, and notebook sanity checks?

## Summary

The **inference helper** is [`FusedScamPredictor`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py) in [`app/fused_predictor.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py): it accepts a **`list[str]`** (including length 1), tokenizes with **`DistilBertTokenizerFast`**, runs **`HybridFusedClassifier.forward`**, and returns **`(P(legit), P(warning), P(fraud))`** per row from a **3-class softmax**.

The **HTTP contract** for **`POST /predict`** is [`PredictResponse`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py) in [`app/schemas.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py): parallel lists **`predicted_class`**, **`predicted_label`**, **`legit_probability`**, **`warning_probability`**, **`fraud_probability`**, **`confidence`**, plus **`warnings`** (heuristic regex codes). The handler is [`predict()`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py) in [`app/main.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py). **`confidence`** is documented as **`max`** of the three softmax probabilities (winner probability) via [`class_from_softmax_triple`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py) in [`app/risk_labels.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py).

**Class id → string label** mapping for the native 3-class head is **0 → `"legit"`**, **1 → `"warning"`**, **2 → `"fraud"`** (constants and softmax mapping in `risk_labels.py`). **Serialized training metadata** for the fused stack lives in **`fused_meta.json`** under the artifact directory (see example under [Serialized metadata](#serialized-metadata)); it includes **`num_labels`** and tokenizer length **`max_len_bert`**, not a separate `id2label` JSON for the three strings (labels are enforced in application code).

**Threshold / business-rule hooks:** [`Settings`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/config.py) in [`app/config.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/config.py) defines **`JOBSENTRY_WARN_THRESHOLD`**, **`JOBSENTRY_FRAUD_THRESHOLD`**, and **`JOBSENTRY_CONFIDENCE_THRESHOLD`**. The docstring on [`map_binary_to_risk`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py) in `risk_labels.py` describes **threshold-based 3-way mapping from scalar `P(scam)`**; that function is **not** invoked on the **`POST /predict`** path in `main.py` (the live path uses the **3-class softmax** and `class_from_softmax_triple`).

**Notebook:** [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb) contains training **`evaluate()`** (softmax, argmax, metrics) and dataset/collate patterns aligned with production tokenization; it does not duplicate the **`FusedScamPredictor`** API verbatim inside the notebook cells surveyed for this research.

**README:** The **Predict** section in [`README.md`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/README.md) still describes **binary `P(scam)`** mapping to three classes and **one-hot** probabilities in prose; the **application code** path described above uses the **3-class softmax** outputs directly. Both files exist; readers comparing them should rely on **`app/`** and OpenAPI **`/docs`** for the implemented contract.

## Detailed Findings

### Ticket scope (TICKET-005)

The ticket file lists: inference helper with tokenizer + forward pass; return **`predicted_class`**, **`predicted_label`**, class probabilities; response fields **`legit_probability`**, **`warning_probability`**, **`fraud_probability`**, **`confidence`**; threshold policy notes; one-text and batch; documented schema; notebook samples; serialized metadata mapping class ids to labels [`cursor/project/tickets/TICKET-005-inference-contract-and-serving.md`](../tickets/TICKET-005-inference-contract-and-serving.md).

### `POST /predict` serving

- Route registration: [`create_app`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L166-L182) adds **`POST /predict`** with **`response_model=PredictResponse`**.
- Startup: [`lifespan`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L41-L51) calls **`_load_predictor`**, which constructs **`FusedScamPredictor.from_artifact_dir`** when **`JOBSENTRY_PHASE6_FUSED_DIR`** is set [`app/main.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py).
- Handler [`predict`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L98-L163): builds **`texts`** from **`JobPostInput.combined_text()`**, runs **`compute_warnings`** per text, enforces **`max_batch_size`**, calls **`predictor.predict_risk_distribution(texts)`**, then for each softmax triple calls **`class_from_softmax_triple`**, and returns **`PredictResponse`** with parallel lists.

### Request body: text assembly

- [`PredictRequest.posts`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L58-L59) requires at least one **`JobPostInput`**.
- [`JobPostInput.combined_text`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L41-L55) returns non-empty **`text`** or joins **`job_title`**, **`job_desc`**, **`skills_desc`**, **`company_profile`**. **`rate`** is validated on the model but **not** included in **`combined_text()`** (see docstring on [`RateInput`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L10-L11)).

### Inference helper (`FusedScamPredictor`)

- [`predict_risk_distribution`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py#L59-L73) chunks by **`max_batch_size`**, delegates to **`_predict_risk_batch`**.
- [`_predict_risk_batch`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py#L75-L97): **`tokenizer(..., padding=True, truncation=True, max_length=max_len_bert)`**, **`model(input_ids, attention_mask)`**, **`F.softmax(logits, dim=-1)`**, returns three floats per row.

### Model load path

- [`load_fused_artifacts`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L91-L143) reads **`fused_meta.json`**, resolves weights (**`model.safetensors`**, optional checkpoint override, or highest **`epoch_*.pt`**), loads **`HybridFusedClassifier`**, loads **`DistilBertTokenizerFast.from_pretrained(artifact_dir)`**.

### Serialized metadata

Example [`artifacts/models/phase6_fused/fused_meta.json`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/models/phase6_fused/fused_meta.json) (workspace copy):

```json
{
  "arch": "sequential_distilbert_bilstm_v1",
  "max_len_bert": 256,
  "lstm_hidden": 64,
  "fusion_hidden": 256,
  "num_labels": 3,
  "dropout": 0.3,
  "distilbert_model": "distilbert-base-uncased",
  "threshold": 0.5,
  "best_val_f1": 0.8351518090859225,
  "test_f1": 0.8333002534378262,
  "test_auc": 0.9502052694308722
}
```

### Heuristic `warnings`

- [`app/predict_warnings.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/predict_warnings.py) **`compute_warnings(text)`** returns a list of string codes from regex patterns; this runs **in parallel** to the neural softmax path in **`predict()`** (same order as **`posts`**).

### Tests (mirror API and predictor)

- [`tests/test_api.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_api.py), [`tests/test_schemas.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_schemas.py), [`tests/test_fused_predictor.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_fused_predictor.py), [`tests/test_risk_labels.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_risk_labels.py) exercise request/response shapes and inference helpers.

### Notebook patterns

- [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb): **`evaluate()`** uses **`torch.softmax(logits, dim=-1)`** and **`argmax`** over logits; **`HybridFusedDataset`** + **`build_collate_fn`** match training-time tokenization (**`padding=True`**, **`truncation=True`**, **`max_length`**). This aligns with **`_predict_risk_batch`** tokenization parameters sourced from **`fused_meta["max_len_bert"]`**.

## Code References

- [`app/main.py:98-163`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L98-L163) — **`predict`**: **`combined_text`**, **`compute_warnings`**, **`predict_risk_distribution`**, **`class_from_softmax_triple`**, **`PredictResponse`**
- [`app/schemas.py:31-84`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/schemas.py#L31-L84) — **`JobPostInput`**, **`PredictRequest`**, **`PredictResponse`** field definitions and descriptions
- [`app/fused_predictor.py:59-97`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py#L59-L97) — **`predict_risk_distribution`**, **`_predict_risk_batch`**
- [`app/risk_labels.py:14-28`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py#L14-L28) — **`class_from_softmax_triple`** (argmax + **`confidence` = max of triple)
- [`app/risk_labels.py:31-64`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py#L31-L64) — **`map_binary_to_risk`** (threshold + heuristic policy; not used in **`main.predict`**)
- [`app/fused_loader.py:91-143`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L91-L143) — **`load_fused_artifacts`**, **`fused_meta.json`**
- [`app/config.py:34-47`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/config.py#L34-L47) — **`max_batch_size`**, **`confidence_threshold`**, **`warn_threshold`**, **`fraud_threshold`**

## Architecture Documentation

- **Batching:** Single-post inference is **`posts` with one element**; larger batches are **`posts` with multiple entries**, chunked inside **`FusedScamPredictor`** and capped by **`JOBSENTRY_MAX_BATCH_SIZE`** in **`predict()`**.
- **Probability semantics:** **`legit_probability`**, **`warning_probability`**, **`fraud_probability`** are the three **softmax** components from the model for each post (not described as one-hot in **`schemas.py`** field descriptions).
- **Dual policy surface:** **Softmax + argmax** drives **`predicted_*`** and parallel probability lists; **`map_binary_to_risk`** exists for **scalar `P(scam)`** workflows and documents threshold ordering expectations via **`Settings`** validators.

## Historical Context (from cursor/project/)

- [`cursor/project/tickets/README.md`](../tickets/README.md) — Places **TICKET-005** after **TICKET-004** in the ticket sequence and notes **TICKET-008** can be finalized after **TICKET-005**.
- [`cursor/project/research/2026-04-18-TICKET-008-backend-predict-deployment.md`](2026-04-18-TICKET-008-backend-predict-deployment.md) — Older research snapshot; the live **`app/`** implementation at research time used the **3-class fused** response fields described in this document.
- [`cursor/project/plan/2026-04-18-TICKET-008-backend-predict-deployment.md`](../plan/2026-04-18-TICKET-008-backend-predict-deployment.md) — Planning text that referenced **binary** **`P(scam)`** mapping in places; **`main.py`** + **`fused_predictor.py`** in the workspace followed **3-class softmax** instead.
- [`cursor/project/implementation/2026-03-22-NA-phase6-fused-production-inference.md`](../implementation/2026-03-22-NA-phase6-fused-production-inference.md) — Notes **`GET /`**, **`GET /health`**, **`POST /predict`** on **`create_app()`**.

## Related Research

- [`cursor/project/research/2026-04-19-TICKET-004-training-and-evaluation-updates.md`](2026-04-19-TICKET-004-training-and-evaluation-updates.md) — Training metrics, **`CrossEntropyLoss`**, class order **`[0,1,2]`**, notebook evaluation loop.
- [`cursor/project/research/2026-04-18-TICKET-008-backend-predict-deployment.md`](2026-04-18-TICKET-008-backend-predict-deployment.md) — Prior **`POST /predict`** / deployment research (supplement; verify against current **`app/`**).

## Open Questions

- Whether **`README.md`** **Predict** prose will be updated to match the **3-class softmax** implementation everywhere, or the README is intentionally describing a **different** deployment contract — only both documents exist today.
