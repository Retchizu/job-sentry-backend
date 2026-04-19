# TICKET-003: Update model to sequential DistilBERT -> BiLSTM

## Objective

Implement the exact architecture flow:
`Text -> Tokenization -> DistilBERT -> Contextual embeddings -> BiLSTM -> Classifier -> Output`.

## Scope

- Update `HybridFusedClassifier` forward pass so BiLSTM consumes DistilBERT token embeddings.
- Use DistilBERT `last_hidden_state` as BiLSTM input sequence.
- Apply masking-aware pooling after BiLSTM output.
- Update classifier head to `num_labels = 3`.
- Keep configurable freeze/unfreeze schedule for DistilBERT layers.

## Acceptance Criteria

- Forward pass works on a sample batch without shape errors.
- Output logits shape is `[batch_size, 3]`.
- Model still supports checkpoint save/load.

## Deliverables

- Updated notebook model class and forward pass cells.
- Notes on tensor shapes at key stages for maintainability.
