# Warning Detection Roadmap Tickets

These tickets break down the work to add warning detection to the fused model workflow.

## Priority Order (importance + non-blockers first)

1. `TICKET-007-combine-row-datasets.md` (high impact data expansion, non-blocking)
2. `TICKET-001-label-schema-and-mapping.md` (core label contract, unblocks training/inference)
3. `TICKET-008-backend-predict-deployment.md` (start API scaffold early, can run in parallel)
4. `TICKET-002-dataset-build-and-splits.md` (depends on dataset + label strategy)
5. `TICKET-003-sequential-fused-model-update.md` (depends on final training inputs)
6. `TICKET-004-training-and-evaluation-updates.md` (depends on updated model and splits)
7. `TICKET-005-inference-contract-and-serving.md` (finalize contract from trained model outputs)
8. `TICKET-006-validation-and-rollout.md` (final gate before production rollout)

## Dependency-aware critical path

`TICKET-007` -> `TICKET-001` -> `TICKET-002` -> `TICKET-003` -> `TICKET-004` -> `TICKET-005` -> `TICKET-006`

`TICKET-008` can begin after `TICKET-001` and be finalized after `TICKET-005`.

## Notes

- Goal pipeline: `Text -> Tokenization -> DistilBERT -> Contextual embeddings -> BiLSTM -> Classifier -> Output`.
- Use `artifacts/datasets/job_postings_rows.csv` as source.
- Keep backward compatibility by documenting label mapping and output contract.
