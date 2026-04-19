# TICKET-006: Validation and rollout — implementation plan

## Overview

Close [`cursor/project/tickets/TICKET-006-validation-and-rollout.md`](../tickets/TICKET-006-validation-and-rollout.md) by producing **evidence-backed validation** of the 3-class fused model, a **transparent comparison** to legacy **binary** baselines on a **shared row-level test split**, **qualitative error analysis** for **warning** and **fraud**, **clear artifact versioning**, and **consumer-facing** documentation (evaluation summary, release checklist, downstream migration impact). Training/inference mechanics are largely covered by TICKET-004 (metrics) and TICKET-005 (contract); this ticket is the **release gate** before production rollout.

## Current State Analysis

- **Inference** (`app/`, tests): `POST /predict` serves **native 3-class softmax** outputs via `FusedScamPredictor` + `class_from_softmax_triple` (`app/fused_predictor.py`, `app/risk_labels.py`). `map_binary_to_risk` exists for **scalar P(scam)** workflows and is **not** the default path (`README.md`).
- **Row-level benchmark split**: [`artifacts/data/processed/merged_test.csv`](../../artifacts/data/processed/merged_test.csv) (3104 rows) with `risk_class` ∈ {0,1,2}, `fraudulent`, `combined_text`; summary in [`merged_splits.summary.json`](../../artifacts/data/processed/merged_splits.summary.json).
- **Multiclass training/eval artifact**: [`artifacts/models/phase6_fused/`](../../artifacts/models/phase6_fused/) includes `fused_meta.json` (e.g. `test_f1`, `test_auc`, `risk_class_labels`), `fused_metrics.csv`, `confusion_matrices.png`, `training_curves.png`, weights, tokenizers — produced from [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](../../artifacts/ipynb/phase6_hybrid_fused.ipynb).
- **Legacy binary ensemble artifacts**: [`artifacts/models/phase6_merged/merged_test_metrics.csv`](../../artifacts/models/phase6_merged/merged_test_metrics.csv) holds **binary** test metrics (LR/RF/XGB) on a merged test evaluation; **label space differs** from 3-class `risk_class` (comparison requires an explicit **collapse / mapping rule** and confirmation the **same rows** were used).
- **Documentation**: [`README.md`](../../README.md) already describes softmax fields, migration from legacy response names, env vars, and a short FastAPI checklist; it does **not** yet contain a **standalone evaluation narrative** (strengths/risks) or a **full release checklist** for TICKET-006 acceptance.
- **Prior research**: [`cursor/project/research/2026-04-20-ENG-006-validation-and-rollout.md`](../research/2026-04-20-ENG-006-validation-and-rollout.md) maps what exists; this plan defines **what to add** to satisfy the ticket.

### Key Discoveries

- **Shared subset for comparison**: `merged_test.csv` is the natural **fixed benchmark** (stratified on `risk_class`, seed 42); use it for fused **on-text** evaluation and any **binary vs multiclass** table.
- **Apples-to-apples caveat**: Binary baselines optimize **fraudulent** (or similar); multiclass optimizes **legit / warning / fraud**. The plan must **document** the mapping used when collapsing 3-class predictions to binary (e.g. “scam-positive” = `predicted_class in {1,2}` vs ground-truth `fraudulent == 1`) and report **per-class** metrics for warning and fraud separately.
- **`fused_meta.json`** already records `best_val_f1`, `test_f1`, `test_auc`, `arch`, `risk_class_labels`; it lacks a **human-facing release tag** (semver or date stamp) required by the ticket’s “clear version tag.”

## Desired End State

1. **Evaluation summary** (markdown): Written narrative with **key strengths** (e.g. macro-F1, AUC, per-class behavior on `merged_test`) and **risks** (class imbalance, confusion between warning vs fraud, domain shift, latency).
2. **Benchmark comparison**: A **single table or notebook section** comparing **multiclass fused** results on **`merged_test`** to **published binary baseline numbers** from `phase6_merged/merged_test_metrics.csv`, with **explicit footnotes** on label alignment and any limitations if baselines were not re-run on the identical file version.
3. **Error analysis**: **False positives and false negatives** for **warning (class 1)** and **fraud (class 2)** with **example snippets** (notebook cells and/or copied into the companion markdown appendix).
4. **Artifact versioning**: **Version tag** recorded in **`fused_meta.json`** (and mirrored in export notebook) plus a one-line **naming convention** (e.g. directory name or git tag pattern) documented next to artifacts.
5. **Downstream integration impact**: Short doc covering **who breaks** (clients expecting binary `P(scam)` only), **field semantics** (softmax triple vs one-hot), **`map_binary_to_risk`**, batch limits, 503 behavior — cross-linked from [`README.md`](../../README.md).
6. **Release checklist**: Standalone checklist for **deployers and API consumers** (env vars, health checks, rollback via checkpoint, tests to run).

### Verification

- **Automated**: `pytest -q` passes (no regressions); any new **script** under `scripts/` has **smoke** or **unit** tests if it contains non-trivial logic (optional: pure re-export of sklearn calls may rely on manual run + committed outputs).
- **Manual**: Reviewer can open the **validation markdown**, **notebook section**, and **tagged `fused_meta.json`** and trace numbers back to **`merged_test`** row counts in `merged_splits.summary.json`.

## What We're NOT Doing

- Retraining `HybridFusedClassifier` or changing architecture/hyperparameters as part of this ticket.
- Replacing the FastAPI stack or adding new endpoints (unless a tiny **internal** script is added for batch eval only).
- Guaranteeing **numerical identity** between notebook re-run and previously committed PNG/CSV if upstream data or torch versions drift (document **repro** prerequisites instead).
- Running full **load/perf** production tests (mention as optional follow-up in risks).

## Implementation Approach

1. **Freeze the benchmark protocol** in prose (data path, columns, metrics, collapse rules for binary comparison).
2. **Add evaluation artifacts** via **repeatable notebook cells** and/or a **small `scripts/` utility** that loads `merged_test.csv`, runs the **same** `FusedScamPredictor` path as production (or loads weights + tokenizer like `from_artifact_dir`), and writes **predictions + sklearn reports** (optional committed JSON/CSV under `artifacts/models/phase6_fused/` or `artifacts/reports/`).
3. **Write two markdown deliverables** under `cursor/project/notes/` (or one combined file if preferred): **evaluation summary** + **release checklist**; link them from `README.md` in a **minimal** new subsection (one paragraph + links).
4. **Extend metadata**: add **`artifact_version`** (or `release_tag`) to **`fused_meta.json`** and to the notebook export cell that writes it; align with TICKET-004/005 conventions for `risk_class_labels`.
5. **Notebook**: add a **“Final validation (TICKET-006)”** section — benchmark table, confusion matrix recap, FP/FN examples for classes 1 and 2.

---

## Phase 1: Benchmark protocol and quantitative comparison

### Overview

Define and document how **multiclass fused** is compared to **binary** baselines on the **shared** `merged_test` split.

### Changes Required

#### 1. Protocol document (section inside evaluation summary or short appendix)

**File**: `cursor/project/notes/TICKET-006-evaluation-summary.md` (new; see Phase 4 for full content)

**Changes**: Include:

- **Dataset**: `artifacts/data/processed/merged_test.csv`, `n=3104`, class counts from `merged_splits.summary.json`.
- **Multiclass metrics**: sklearn `classification_report`, confusion matrix on `risk_class` vs `argmax` prediction; macro-F1, weighted-F1, OVR AUC (match TICKET-004 list).
- **Binary comparison**: Reference [`artifacts/models/phase6_merged/merged_test_metrics.csv`](../../artifacts/models/phase6_merged/merged_test_metrics.csv) for **Merged LR / RF / XGB**; state the **collapse rule** for multiclass→binary (e.g. ground truth positive = `fraudulent==1`, predicted positive = `predicted_class==2` **or** include warning — **pick one** and apply consistently). If the saved binary metrics were produced on a **different** `merged_test` revision, **state that** and treat numbers as **indicative** unless re-run.

#### 2. Notebook and/or script — run fused on `merged_test`

**Preferred**: New markdown section + cells in [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](../../artifacts/ipynb/phase6_hybrid_fused.ipynb) **or** a script [`scripts/evaluate_fused_on_merged_test.py`](../../scripts/evaluate_fused_on_merged_test.py) (new) that:

- Reads `merged_test.csv` (`combined_text`, `risk_class`).
- Instantiates predictor from `JOBSENTRY_PHASE6_FUSED_DIR` or default `artifacts/models/phase6_fused`.
- Batches `predict_full` / softmax triples, builds `y_pred`.
- Prints/saves `classification_report`, confusion matrix, and optional **binary collapsed** metrics.

**Note**: If GPU memory is a concern, batch size should mirror `JOBSENTRY_MAX_BATCH_SIZE` defaults.

#### 3. Optional committed outputs

**Path**: e.g. `artifacts/models/phase6_fused/validation_merged_test.json` (new) — small JSON with metrics + git hash + `artifact_version` for auditability.

### Success Criteria

#### Automated Verification

- [x] `pytest -q` passes repository-wide.
- [x] If a new script is added: `python scripts/evaluate_fused_on_merged_test.py` exits 0 in dev (CPU) with `--help` or dry-run if artifacts missing (graceful skip **or** documented requirement to set `JOBSENTRY_PHASE6_FUSED_DIR`).

#### Manual Verification

- [ ] Numbers in the evaluation summary **match** script/notebook output for the same artifact checkout.
- [ ] Comparison table **footnotes** explain binary vs 3-class label spaces.

**Implementation Note**: Pause after Phase 1 for human confirmation that the **collapse rule** for binary comparison matches product intent before locking Phase 2 examples.

---

## Phase 2: Error analysis (warning and fraud FP/FN)

### Overview

Surface **actionable** misclassification examples for **class 1 (warning)** and **class 2 (fraud)** from the `merged_test` run.

### Changes Required

#### 1. Notebook cells or script output

**File**: [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](../../artifacts/ipynb/phase6_hybrid_fused.ipynb) (or the new script)

**Changes**:

- From per-row `(y_true, y_pred, combined_text)`, filter:
  - **Fraud FN**: `y_true==2`, `y_pred!=2` (and variants for FP).
  - **Warning FN/FP**: analogous for `y_true==1` / `y_pred==1`.
- Display **5–15** short excerpts (truncate `combined_text` to e.g. 400 chars); **do not** log PII beyond what is already in the dataset.

#### 2. Optional appendix in `TICKET-006-evaluation-summary.md`

**Changes**: Paste or summarize patterns observed (e.g. ambiguous wording, label noise).

### Success Criteria

#### Automated Verification

- [x] N/A beyond tests still passing (qualitative deliverable).

#### Manual Verification

- [ ] At least **two** fraud-error and **two** warning-error examples are visible in notebook or markdown.
- [ ] Confusion matrix off-diagonal for classes 1 and 2 is **referenced** in the evaluation summary.

---

## Phase 3: Artifact versioning and naming

### Overview

Satisfy “artifacts are saved with clear version tag” by **embedding** a tag in **`fused_meta.json`** and documenting **how** releases are named.

### Changes Required

#### 1. Notebook export cell (writes `fused_meta.json`)

**File**: [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](../../artifacts/ipynb/phase6_hybrid_fused.ipynb)

**Changes**: When serializing `fused_meta`, add fields such as:

- `artifact_version`: e.g. `"1.0.0"` or `"2026-04-20"` (pick **one** scheme and use consistently).
- `git_commit`: optional, if running in a dev environment that has git (else omit).

#### 2. Committed `fused_meta.json`

**File**: [`artifacts/models/phase6_fused/fused_meta.json`](../../artifacts/models/phase6_fused/fused_meta.json)

**Changes**: Update to include the new fields after notebook re-export (or hand-edit once with the same values the notebook will produce).

#### 3. Release checklist / evaluation summary

**Changes**: Document **directory layout** expectation: e.g. deploy `artifacts/models/phase6_fused/` as a versioned bundle `phase6_fused-<artifact_version>/` in object storage (operational convention, not necessarily a code change).

### Success Criteria

#### Automated Verification

- [x] `tests/test_fused_loader.py` still passes (loader **ignores unknown** extra keys or documents them — current loader reads only known keys; JSON may grow without code change).

#### Manual Verification

- [ ] `fused_meta.json` contains **`artifact_version`** readable by humans.
- [ ] Release checklist mentions **which** field to verify when promoting a build.

---

## Phase 4: Documentation deliverables (summary, checklist, downstream impact)

### Overview

Produce the **ticket’s written deliverables** and **link** them from the main README.

### Changes Required

#### 1. Evaluation summary

**File**: `cursor/project/notes/TICKET-006-evaluation-summary.md` (new)

**Content** (minimum):

- **Strengths**: headline metrics + per-class behavior; reference `fused_meta.json` / validation run.
- **Risks**: imbalance (rare fraud), warning vs fraud confusion, operational (latency, GPU), data drift.
- **Benchmark comparison** table + limitations (Phase 1).
- **Error analysis** pointer (Phase 2).

#### 2. Release checklist for deployers and consumers

**File**: `cursor/project/notes/TICKET-006-release-checklist.md` (new)

**Content** (checklist items):

- Env: `JOBSENTRY_PHASE6_FUSED_DIR`, optional `JOBSENTRY_PHASE6_FUSED_CHECKPOINT`, `JOBSENTRY_DEVICE`, `JOBSENTRY_MAX_BATCH_SIZE`.
- Health: `GET /health` shows `model_loaded`, `mode`, `device`.
- Smoke: `POST /predict` sample body; expect 200 vs 503 when unset.
- Rollback: point to `checkpoints/epoch_XX.pt` vs `model.safetensors` per [`README.md`](../../README.md) / [`app/fused_loader.py`](../../app/fused_loader.py).
- Tests: `pytest -q`.

#### 3. Downstream integration impact

**File**: Same as (1) under a dedicated **“Downstream integration”** section **or** split if long.

**Content**: Binary-only clients; softmax vs one-hot; `map_binary_to_risk`; field renames from legacy API (`README.md` migration paragraph — **link**, avoid duplicating verbatim).

#### 4. README pointer

**File**: [`README.md`](../../README.md)

**Changes**: Add a **short** subsection (e.g. “Validation & rollout (TICKET-006)”) with **two links** to the new notes files — **no** large paste.

### Success Criteria

#### Automated Verification

- [x] Markdown paths exist and are valid relative links from repo root.
- [x] `pytest -q` still passes.

#### Manual Verification

- [ ] A new engineer can follow **release checklist** to deploy and smoke-test.
- [ ] Evaluation summary satisfies ticket **acceptance criteria** (strengths, risks, downstream impact, version tag reference).

---

## Testing Strategy

### Unit / integration

- Existing **`tests/test_api.py`**, **`tests/test_fused_predictor.py`**, **`tests/test_fused_loader.py`**, **`tests/test_risk_labels.py`**: run full suite after doc/script additions.
- New script: if it includes **pure functions** (e.g. collapse labels), add **small tests** in `tests/test_validation_metrics.py` (optional).

### Manual

1. Run notebook section or script on `merged_test`; spot-check counts vs `merged_splits.summary.json`.
2. Open `TICKET-006-evaluation-summary.md` and confirm metrics match.
3. Follow release checklist on a clean env with/without `JOBSENTRY_PHASE6_FUSED_DIR`.

## Performance Considerations

- Batch evaluation on 3104 rows: use batched `predict_full`; CPU may take minutes — document expected runtime or provide `--limit` for dev.

## Migration Notes

- **Consumers**: See `TICKET-006-evaluation-summary.md` + [`README.md`](../../README.md) migration lines for API field changes.
- **Artifacts**: Deploy a **versioned** copy of `phase6_fused` with **`artifact_version`** in `fused_meta.json`.

## References

- Ticket: [`cursor/project/tickets/TICKET-006-validation-and-rollout.md`](../tickets/TICKET-006-validation-and-rollout.md)
- Upstream tickets: [`TICKET-004`](./2026-04-19-TICKET-004-training-and-evaluation-updates.md), [`TICKET-005`](./2026-04-19-TICKET-005-inference-contract-and-serving.md)
- Research: [`cursor/project/research/2026-04-20-ENG-006-validation-and-rollout.md`](../research/2026-04-20-ENG-006-validation-and-rollout.md)
- Data split summary: [`artifacts/data/processed/merged_splits.summary.json`](../../artifacts/data/processed/merged_splits.summary.json)
- Binary baseline metrics: [`artifacts/models/phase6_merged/merged_test_metrics.csv`](../../artifacts/models/phase6_merged/merged_test_metrics.csv)
