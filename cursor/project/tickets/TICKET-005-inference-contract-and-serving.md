# TICKET-005: Define inference contract and prediction outputs

## Objective

Produce stable inference outputs for application consumption.

## Scope

- Add an inference helper function that:
  - Accepts raw text input.
  - Runs tokenizer and model forward pass.
  - Returns `predicted_class`, `predicted_label`, and class probabilities.
- Define response contract:
  - `legit_probability`
  - `warning_probability`
  - `fraud_probability`
  - `confidence`
- Add threshold policy notes for optional business rules (for example, escalating borderline warnings).

## Acceptance Criteria

- One-text and batch inference both work.
- Output dictionary schema is consistent and documented.
- Sample predictions are shown in notebook for sanity checking.

## Deliverables

- Inference cell(s) in notebook.
- Serialized metadata mapping class ids to labels.
- Response contract that can be reused directly by backend `POST /predict`.
