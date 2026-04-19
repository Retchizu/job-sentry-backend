# TICKET-006: Validate, compare, and roll out

## Objective

Validate model quality and ensure safe rollout from binary to multiclass predictions.

## Scope

- Compare old binary model vs new multiclass model on shared benchmark subset.
- Inspect false positives and false negatives for warning and fraud classes.
- Add error analysis examples to notebook.
- Finalize model artifacts and version naming.
- Document migration notes for any downstream service expecting binary output.

## Acceptance Criteria

- Evaluation summary is written with key strengths and risks.
- Artifacts are saved with clear version tag.
- Downstream integration impact is documented.

## Deliverables

- Final validation section in notebook or companion markdown.
- Release checklist for deployment/inference consumers.
