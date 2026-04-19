---
date: 2026-04-19T07:08:49Z
researcher: riche
git_commit: 26c01727e996da4fcc64221713a2f75fad464f18
branch: main
repository: job-sentry-backend
topic: "TICKET-004: Training and evaluation for multiclass (3-class) fused model"
tags: [research, codebase, TICKET-004, phase6, HybridFusedClassifier, metrics, CrossEntropyLoss]
status: complete
last_updated: 2026-04-19
last_updated_by: riche
metadata_note: "hack/spec_metadata.sh was not present in the repository; git hash, branch, and timestamps were gathered manually. The working tree had uncommitted modifications at research time; on-disk Python/notebook sources described below reflect the workspace state."
---

# Research: TICKET-004 — Training and evaluation for multiclass

**Date**: 2026-04-19T07:08:49Z  
**Researcher**: riche  
**Git Commit**: `26c01727e996da4fcc64221713a2f75fad464f18`  
**Branch**: main  
**Repository**: job-sentry-backend  

## Research Question

What exists in the repository today with respect to [`cursor/project/tickets/TICKET-004-training-and-evaluation-updates.md`](../tickets/TICKET-004-training-and-evaluation-updates.md): multiclass training/evaluation (3 classes), class-aware metrics, class weights `[0, 1, 2]`, `CrossEntropyLoss`, confusion matrix labels (`legit`, `warning`, `fraud`), and saved metrics artifacts?

## Summary

The **phase 6 fused training path** lives primarily in [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb). It loads CSV splits with integer `risk_class` in `{0, 1, 2}`, builds weighted **`nn.CrossEntropyLoss`** using **`sklearn.utils.class_weight.compute_class_weight`** with **`classes=np.array([0, 1, 2])`**, trains **`HybridFusedClassifier`** from [`app/hybrid_fused_model.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py), and evaluates with **accuracy**, **macro F1**, and **multiclass ROC-AUC (one-vs-rest)** inside a notebook-defined **`evaluate()`** function. **Per-class precision, recall, and F1** and a **weighted-average** row appear in printed **`classification_report(..., target_names=["legit", "warning", "fraud"])`** output. **Confusion matrices** use **`labels=[0, 1, 2]`** and axis labels **`legit` / `warning` / `fraud`**. The exported **`fused_metrics.csv`** (built in the notebook) stores **`split`, `loss`, `acc`, `f1`, `auc`** per row; the **`f1`** column is populated from the same **macro F1** returned by **`evaluate()`**, not a separately named weighted-F1 column.

The **FastAPI app** ([`app/main.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py)) does not run training; it loads fused artifacts and maps softmax triples to labels via [`app/risk_labels.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py) **`class_from_softmax_triple`**.

A maintenance script [`scripts/patch_phase6_hybrid_fused_notebook.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/scripts/patch_phase6_hybrid_fused_notebook.py) exists to align notebook markdown/code with the **sequential 3-class** stack and the same loss/metrics patterns.

## Detailed Findings

### TICKET-004 scope (reference)

The ticket [`cursor/project/tickets/TICKET-004-training-and-evaluation-updates.md`](../tickets/TICKET-004-training-and-evaluation-updates.md) specifies:

- Replace binary-only assumptions in data loaders and metrics.
- Class weights for classes `[0, 1, 2]`, **`CrossEntropyLoss`** for multiclass.
- Evaluation metrics: accuracy; macro-F1; weighted-F1; per-class precision/recall/F1; confusion matrix labels **`legit`**, **`warning`**, **`fraud`**.
- Acceptance: end-to-end train/val/test; classification reports for 3 classes; metrics artifact saved with multiclass values.

### `HybridFusedClassifier` — 3-class head

[`app/hybrid_fused_model.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py) defines **`HybridFusedClassifier`**: DistilBERT → packed BiLSTM on token sequence → masked mean pool → MLP. Default **`num_labels=3`**; docstring states logits **`[B, num_labels]`** with **legit / warning / fraud** semantics ([`app/hybrid_fused_model.py:11-27`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L11-L27), [`app/hybrid_fused_model.py:52-57`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L52-L57)).

### Notebook `phase6_hybrid_fused.ipynb` — data, loss, metrics, artifacts

- **Imports**: `compute_class_weight`, `accuracy_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report` (grep-visible in notebook JSON).
- **Class weights**: `compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)` → tensor on device; **`criterion = nn.CrossEntropyLoss(weight=class_weights_t)`** (e.g. [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb) source lines around the hyperparameter/criterion cells).
- **`evaluate()`**: aggregates loss; **`accuracy_score`**, **`f1_score(..., average="macro", zero_division=0)`**, **`roc_auc_score(..., multi_class="ovr")`** on softmax probabilities; returns preds for downstream reporting (pattern matches [`scripts/patch_phase6_hybrid_fused_notebook.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/scripts/patch_phase6_hybrid_fused_notebook.py) around lines 221–290).
- **Post-training**: **`classification_report(y_val, val_preds, target_names=["legit", "warning", "fraud"])`** and the same for test; **`confusion_matrix(..., labels=[0, 1, 2])`** with heatmaps using string labels **`["legit", "warning", "fraud"]`**.
- **`fused_metrics.csv`**: notebook builds a **`DataFrame`** with rows for **`val`** and **`test`**, columns **`split`, `loss`, `acc`, `f1`, `auc`** ([`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb) — `metrics_df` cell). Example on-disk CSV:

```text
split,loss,acc,f1,auc
val,...
test,...
```

- **`fused_meta.json`**: written in the same export cell block with keys such as **`best_val_f1`, `test_f1`, `test_auc`** (see notebook export section); **`num_labels`** is part of the fused metadata consumed by the loader ([`app/fused_loader.py:119-130`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L119-L130)).

### Inference stack — labels and probabilities

- [`app/fused_predictor.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py): **`predict_risk_distribution`** returns **`(P(legit), P(warning), P(fraud))`** per row; documents alignment with **`app.risk_labels`** indices ([`app/fused_predictor.py:59-66`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py#L59-L66), [`app/fused_predictor.py:86-96`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py#L86-L96)).
- [`app/risk_labels.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py): **`class_from_softmax_triple`** maps the triple to class index and string label **`legit` / `warning` / `fraud`** ([`app/risk_labels.py:14-27`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py#L14-L27)); **`map_binary_to_risk`** exists for binary-head compatibility ([`app/risk_labels.py:31-64`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py#L31-L64)).
- [`app/main.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py): **`predict`** uses **`class_from_softmax_triple`** on fused softmax outputs ([`app/main.py:129-147`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L129-L147)).

### `app/config.py`

Application settings control artifact paths, device, batch size, and **binary-mapping thresholds** for **`map_binary_to_risk`**; there is **no** training-time **`class_weights`** field in settings ([`app/config.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/config.py)).

### Tests

- [`tests/test_hybrid_fused_model.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_hybrid_fused_model.py): asserts forward output shape **`(b, 3)`** for **`num_labels=3`** ([`tests/test_hybrid_fused_model.py:25-39`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_hybrid_fused_model.py#L25-L39)).
- [`tests/test_risk_labels.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/tests/test_risk_labels.py) (if present in workspace) exercises label helpers.

### Other notebooks (binary vs multiclass)

- [`artifacts/ipynb/phase6_deep_learning.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_deep_learning.ipynb), [`main_scam_detection.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/main_scam_detection.ipynb), [`dataset2_scam_detection.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/dataset2_scam_detection.ipynb): **binary** Keras/sklearn flows with **Legitimate/Scam** style labels — separate from the **3-class** phase6 hybrid fused notebook.

## Code References

- [`app/hybrid_fused_model.py:11-96`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py#L11-L96) — Sequential **3-logit** fused classifier.
- [`app/fused_loader.py:1-143`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L1-L143) — Load **`fused_meta.json`**, resolve weights, construct **`HybridFusedClassifier`**, load tokenizer.
- [`app/fused_predictor.py:59-97`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_predictor.py#L59-L97) — Softmax triple inference.
- [`app/risk_labels.py:9-27`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/risk_labels.py#L9-L27) — **0/1/2** ↔ **`legit`/`warning`/`fraud`** for native softmax.
- [`app/main.py:98-163`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/main.py#L98-L163) — **`/predict`** wiring.
- [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/artifacts/ipynb/phase6_hybrid_fused.ipynb) — Training loop, **`evaluate()`**, **`classification_report`**, confusion matrices, **`fused_metrics.csv`** export.
- [`scripts/patch_phase6_hybrid_fused_notebook.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/scripts/patch_phase6_hybrid_fused_notebook.py) — Programmatic notebook alignment with sequential 3-class content.

## Architecture Documentation

- **Training**: Jupyter notebook drives PyTorch training; class weights feed **`CrossEntropyLoss`**; validation uses **macro F1** as the primary scalar F1 in **`evaluate()`** and for **best checkpoint** selection (as described in subagent synthesis from notebook cells).
- **Metrics**: **Accuracy** and **macro F1** are computed inside **`evaluate()`**; **weighted F1** appears as part of sklearn’s **`classification_report`** text output (weighted avg row), not as a separate column in **`fused_metrics.csv`** in the current notebook export pattern.
- **Inference**: Single **DistilBERT** tokenizer path; no separate **`word_index`** / parallel branch in [`app/fused_loader.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py) docstring ([`app/fused_loader.py:1-5`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/fused_loader.py#L1-L5)).

## Historical Context (from `cursor/project/notes/` and `cursor/project/research/`)

- [`cursor/project/research/2026-04-18-TICKET-003-sequential-fused-model-update.md`](2026-04-18-TICKET-003-sequential-fused-model-update.md) — Documents an **older** **`HybridFusedClassifier`** layout (parallel fusion, **`num_labels=2`**) and binary notebook patterns at the **cited commit**. The **current** workspace copies of [`app/hybrid_fused_model.py`](https://github.com/Retchizu/job-sentry-backend/blob/26c01727e996da4fcc64221713a2f75fad464f18/app/hybrid_fused_model.py) and the phase6 notebook **differ** from that historical description (sequential **3-class** stack in the files read for this research).
- [`cursor/project/research/2026-04-18-TICKET-002-dataset-build-and-splits.md`](2026-04-18-TICKET-002-dataset-build-and-splits.md) — Row-level **`risk_class`** labeling upstream of fused training.
- [`cursor/project/research/2026-04-18-TICKET-007-combine-row-datasets.md`](2026-04-18-TICKET-007-combine-row-datasets.md) — Combined datasets and **`risk_class`** distribution reporting, cross-linked to fused training.
- [`cursor/project/notes/Implementation Plan 1.md`](../notes/Implementation%20Plan%201.md), [`cursor/project/notes/Implementation Plan 2.md`](../notes/Implementation%20Plan%202.md) — High-level evaluation phases for earlier dataset pipelines.

## Related Research

- [`cursor/project/research/2026-04-18-TICKET-003-sequential-fused-model-update.md`](2026-04-18-TICKET-003-sequential-fused-model-update.md)
- [`cursor/project/research/2026-04-18-TICKET-002-dataset-build-and-splits.md`](2026-04-18-TICKET-002-dataset-build-and-splits.md)
- [`cursor/project/research/2026-04-18-TICKET-007-combine-row-datasets.md`](2026-04-18-TICKET-007-combine-row-datasets.md)
- [`cursor/project/research/2026-03-23-phase6-fused-only-artifact-usage.md`](2026-03-23-phase6-fused-only-artifact-usage.md)

## Open Questions

- None required for “as-is” documentation. Historical doc **TICKET-003** research may not match the **current** sequential **3-class** implementation; use live files as the source of truth for audits that span multiple commits.
