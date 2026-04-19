# TICKET-004: Update training and evaluation for multiclass

## Objective

Train and evaluate the updated 3-class model with robust class-aware metrics.

## Scope

- Replace binary-only assumptions in data loaders and metrics.
- Compute class weights for classes `[0, 1, 2]`.
- Keep `CrossEntropyLoss` for multiclass objective.
- Update evaluation metrics:
  - Accuracy
  - Macro-F1
  - Weighted-F1
  - Per-class precision/recall/F1
- Update confusion matrix labels (`legit`, `warning`, `fraud`).

## Acceptance Criteria

- Training loop runs end-to-end on train/val/test.
- Classification reports correctly show 3 target classes.
- Metrics artifact is saved and includes multiclass values.

## Deliverables

- Updated training/eval cells in notebook.
- Revised plots and confusion matrix outputs.
