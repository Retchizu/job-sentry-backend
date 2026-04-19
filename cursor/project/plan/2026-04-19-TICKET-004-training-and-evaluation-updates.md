# TICKET-004: Multiclass training and evaluation — implementation plan

## Overview

Bring the Phase 6 **sequential fused** training notebook (`artifacts/ipynb/phase6_hybrid_fused.ipynb`) fully in line with TICKET-004: explicit **multiclass** metrics (accuracy, macro-F1, weighted-F1, per-class precision/recall/F1), **class weights** for `[0, 1, 2]` with **`CrossEntropyLoss`**, **confusion matrix** axis labels **`legit` / `warning` / `fraud`**, and a **saved metrics artifact** that records those values (not only macro-F1 and AUC).

## Current State Analysis

- **Model and inference** (`app/hybrid_fused_model.py`, `app/fused_loader.py`, `app/fused_predictor.py`, `app/risk_labels.py`): Already assume a **3-logit** head and `num_labels` from `fused_meta.json`. No code change required for TICKET-004 unless you choose to **read** extended metrics files at runtime (out of scope for this ticket).

- **Notebook** (`phase6_hybrid_fused.ipynb`):
  - **Data**: `merged_{train,val,test}.csv` with `combined_text` and integer **`risk_class`** — already multiclass.
  - **Loss**: `sklearn.utils.class_weight.compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)` → `nn.CrossEntropyLoss(weight=...)` — already satisfies class-weight + CE requirement.
  - **`evaluate()`** (notebook): Returns **loss, accuracy, macro F1, OVR AUC**, predictions; docstring says “macro-F1” only — **weighted F1 is not computed** in this function.
  - **Reporting**: `classification_report(..., target_names=["legit", "warning", "fraud"])` and confusion matrices with **`labels=[0, 1, 2]`** — already show 3-class names in stdout/plots.
  - **`fused_metrics.csv`**: Columns `split`, `loss`, `acc`, `f1`, `auc` where **`f1` is macro F1** from `evaluate()`. **Weighted F1 and per-class metrics are not persisted** in structured form (only visible in printed `classification_report`).

- **Maintenance script** `scripts/patch_phase6_hybrid_fused_notebook.py`: Embeds parallel copies of notebook sections; should be **updated in lockstep** if `evaluate()`, export, or training-loop metric keys change.

- **README.md** (`README.md` ~64–72): Still describes **binary** `P(scam)` mapping in places — **documentation drift** vs current softmax triple. Fixing that is **not** required to close TICKET-004 but is listed under “not doing” vs optional follow-up.

### Key Discoveries

- `evaluate()` returns **macro** F1 only; checkpoint selection uses **`val_f1`** = macro F1 (`phase6_hybrid_fused.ipynb` training loop and `history` rows).
- `fused_metrics.csv` is **produced only by the notebook**; **`app/` does not consume it** — safe to add columns or a sidecar JSON without breaking the API.

## Desired End State

1. **Training** runs end-to-end on **train / val / test** with **`CrossEntropyLoss`** and class weights for **`[0, 1, 2]`** (already true; verify after edits).

2. **Evaluation** computes and surfaces:
   - **Accuracy**
   - **Macro-F1**
   - **Weighted-F1**
   - **Per-class** precision, recall, F1 for `legit`, `warning`, `fraud`

3. **Confusion matrices** use semantic labels **`legit`, `warning`, `fraud`** (and `labels=[0, 1, 2]` where applicable) — already the pattern; confirm plots/titles stay consistent.

4. **Metrics artifact**: At minimum **`fused_metrics.csv`** (under `FUSED_OUT_DIR`, typically `artifacts/models/phase6_fused/`) includes **multiclass scalars** (explicit **`f1_macro`** and **`f1_weighted`**; keep legacy **`f1`** column as **alias of macro F1** for backward compatibility unless you explicitly want a breaking rename). **Per-class** metrics are **persisted** in structured form — recommended: a second file **`fused_per_class_metrics.json`** (or equivalent) with val/test sections and per-label precision/recall/F1, to avoid an overly wide CSV.

5. **Classification reports** in the notebook clearly show **three** target classes (already via `target_names`).

### Verification

- Re-run notebook cells from data load through export; confirm **no binary-only** code paths remain in this notebook’s metric/dataset sections.
- Open **`fused_metrics.csv`** and **`fused_per_class_metrics.json`** and confirm values match sklearn recomputation on saved predictions (spot-check).

## What We're NOT Doing

- Changing **`HybridFusedClassifier`** architecture or **`app/`** inference logic **for this ticket** (unless a tiny shared helper is extracted — optional and usually unnecessary).
- Retraining production weights as a **CI** step (manual/local GPU run remains the norm).
- Updating **`README.md`** API description (binary vs 3-class) — **optional** follow-up; not part of TICKET-004 deliverables as written.
- **Replacing** macro-F1 as the **checkpoint-selection** metric unless product asks — default plan keeps **macro F1** for `best_val_f1` / `epoch_*.pt` selection; weighted F1 is **reported** alongside.

## Implementation Approach

1. **Extend `evaluate()`** to compute **weighted F1** (`sklearn.metrics.f1_score(..., average="weighted", zero_division=0)`) and return it alongside existing return values. Update all call sites (training loop prints, final val/test aggregation, export).

2. **Optional but recommended**: Add **`precision_recall_fscore_support`** (labels `[0,1,2]`) or parse **`classification_report`** output into a **dict** for **structured per-class** metrics on val and test.

3. **Persist metrics**:
   - Update **`metrics_df`** / CSV: add **`f1_weighted`**; add **`f1_macro`** or document that **`f1`** = macro; include **`acc`**, **`auc`** as today.
   - Write **`fused_per_class_metrics.json`** with val/test nested metrics (class-index or label keys consistent with `app/risk_labels.py`: 0=legit, 1=warning, 2=fraud).

4. **Plots**: Ensure confusion-matrix plotting uses **xtick/ytick** labels `["legit", "warning", "fraud"]` and **`labels=[0, 1, 2]`** — align with existing cells; only adjust if any cell still assumes 2×2.

5. **Sync `scripts/patch_phase6_hybrid_fused_notebook.py`** so programmatic regeneration of the notebook does not revert multiclass metric work.

6. **Tests**: Repository tests do not execute the notebook; **manual** verification is primary. Optional: add a small **`tests/test_multiclass_metrics_helpers.py`** only if you extract pure functions (e.g. building JSON from numpy arrays) into `app/` or `scripts/` — **skip** if all logic stays in the notebook.

---

## Phase 1: Extend `evaluate()` and training loop

### Overview

Return **weighted F1** from `evaluate()`, thread it through **epoch logging** and **`history`** rows, and print **macro vs weighted** clearly. Keep **`val_f1`** in checkpoints as **macro F1** unless the team decides otherwise (document in notebook markdown).

### Changes Required

#### 1. Notebook — `evaluate()` and training loop

**File**: `artifacts/ipynb/phase6_hybrid_fused.ipynb`  
**Changes**:

- Change `evaluate()` to compute:
  - `f1_macro = f1_score(..., average="macro", zero_division=0)`
  - `f1_weighted = f1_score(..., average="weighted", zero_division=0)`
- Return tuple including both (e.g. `loss, acc, f1_macro, f1_weighted, auc, probs, preds`) — update **every** unpack of `evaluate()`.
- In the per-epoch `row = dict(...)`, add **`val_f1_weighted`** (and keep **`val_f1`** as macro for backward compatibility with resume/best-checkpoint logic).
- Update **`print`** lines for epochs to show both F1s or label them explicitly.

```python
# Illustrative shape (adapt to actual cell)
def evaluate(model, loader, criterion, device):
    ...
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, preds, average="weighted", zero_division=0)
    ...
    return loss, acc, f1_macro, f1_weighted, auc, probs, preds
```

### Success Criteria

#### Automated Verification

- [x] `pytest -q` passes from repo root (no Python regressions if only notebook changed — **notebook-only** changes may not touch tests).

#### Manual Verification

- [ ] Run `evaluate()` cell and one training epoch: stdout shows **both** macro and weighted F1.
- [ ] Resume-from-checkpoint path still loads when **`val_f1`** key remains macro in **`history`** (if you keep key names stable).

**Implementation Note**: After Phase 1 manual checks, proceed to Phase 2.

---

## Phase 2: Final val/test reporting and structured per-class metrics

### Overview

After training, recompute or reuse **`evaluate()`** outputs for **val** and **test**. Build **per-class** precision/recall/F1 (e.g. via `precision_recall_fscore_support` with `labels=[0,1,2]`, `zero_division=0`) and map indices to **`legit` / `warning` / `fraud`**.

### Changes Required

#### 1. Notebook — post-training cells

**File**: `artifacts/ipynb/phase6_hybrid_fused.ipynb`  
**Changes**:

- Ensure **`classification_report`** calls use **`target_names=["legit", "warning", "fraud"]`** and **`labels=[0, 1, 2]`** if using explicit labels.
- Build a **JSON-serializable** dict, e.g.:

```python
# Conceptual structure
{
  "val": {
    "legit":   {"precision": ..., "recall": ..., "f1": ...},
    "warning": {...},
    "fraud":   {...}
  },
  "test": { ... }
}
```

- Write to **`os.path.join(FUSED_OUT_DIR, "fused_per_class_metrics.json")`** with `json.dump(..., indent=2)`.

### Success Criteria

#### Automated Verification

- [ ] File **`fused_per_class_metrics.json`** appears under `FUSED_OUT_DIR` after export cell runs.

#### Manual Verification

- [ ] JSON **keys** match the three TICKET-004 labels and **numeric** ranges are sensible ([0,1] for precision/recall/F1).

---

## Phase 3: `fused_metrics.csv` and `fused_meta.json` alignment

### Overview

Extend **`fused_metrics.csv`** so it **explicitly** stores **macro** and **weighted** F1. Preserve **`f1`** as **macro F1** for backward compatibility (same numeric meaning as today).

### Changes Required

#### 1. Notebook — metrics DataFrame

**File**: `artifacts/ipynb/phase6_hybrid_fused.ipynb`  
**Changes**:

- Build **`metrics_df`** with columns such as:  
  `split`, `loss`, `acc`, `f1`, `f1_macro`, `f1_weighted`, `auc`  
  where **`f1` == `f1_macro`** (duplicate column) **or** omit duplicate and document breaking change — **prefer duplicate for clarity + compatibility**.

#### 2. Notebook — `fused_meta.json` (optional)

If **`fused_meta.json`** currently stores **`test_f1`**, document in a markdown cell that it refers to **macro F1** (or add **`test_f1_macro`** / **`test_f1_weighted`** keys — optional; avoid breaking consumers that read **`test_f1`** only).

### Success Criteria

#### Manual Verification

- [ ] **`fused_metrics.csv`** contains **multiclass** scalar columns including **weighted F1**.
- [ ] Values align with sklearn metrics computed from the same predictions.

---

## Phase 4: Confusion matrices and training curves

### Overview

Confirm heatmaps use **3×3** confusion matrices and **string tick labels** **`legit`, `warning`, `fraud`**. Update titles/legends if they still say “binary” or “scam”.

### Changes Required

#### 1. Notebook — plotting cells

**File**: `artifacts/ipynb/phase6_hybrid_fused.ipynb`  
**Changes**:

- Grep for **`confusion_matrix`**, **`sns.heatmap`**, **`labels=`** — set **`labels=[0, 1, 2]`** and **`xticklabels` / `yticklabels`** to **`["legit", "warning", "fraud"]`**.
- Regenerate **`confusion_matrices.png`** / **`training_curves.png`** as part of the export section if those files are checked in for documentation.

### Success Criteria

#### Manual Verification

- [ ] Saved confusion matrix figure shows **3×3** with correct axis labels.
- [ ] No notebook section asserts **`NUM_LABELS == 2`** or binary-only metrics.

---

## Phase 5: Sync `patch_phase6_hybrid_fused_notebook.py`

### Overview

Update **`scripts/patch_phase6_hybrid_fused_notebook.py`** so any **string literals** it injects for `evaluate()`, metrics export, or markdown **match** the notebook after Phases 1–4.

### Changes Required

#### 1. Script

**File**: `scripts/patch_phase6_hybrid_fused_notebook.py`  
**Changes**: Align patched cells with new `evaluate()` signature, **`metrics_df`** columns, and **`fused_per_class_metrics.json`** export if the script contains those blocks.

### Success Criteria

#### Automated Verification

- [x] `python scripts/patch_phase6_hybrid_fused_notebook.py` runs without error (if that is the project’s intended usage).

#### Manual Verification

- [ ] Re-run script and **diff** notebook — no unintended reversion of multiclass metrics.

---

## Testing Strategy

### Unit Tests

- **Optional** only if shared helpers are extracted to importable modules.

### Integration Tests

- Not required for notebook-only work.

### Manual Testing Steps

1. Clear or backup old `phase6_fused` metrics files; run notebook end-to-end (or from loaded checkpoint through eval/export only).
2. Inspect **`fused_metrics.csv`**, **`fused_per_class_metrics.json`**, **`fused_meta.json`**, confusion matrix PNG.
3. Confirm **train/val/test** paths all execute without shape/label errors.

## Performance Considerations

- Extra sklearn metrics add negligible time vs forward passes.

## Migration Notes

- **Downstream** consumers of **`fused_metrics.csv`**: If they assumed **`f1`** was the only F1 column, behavior **unchanged** if **`f1`** remains macro. New columns are **additive**.
- **Checkpoint files**: If **`history`** rows gain new keys, old checkpoints remain loadable if **`val_f1`** is still present and resume logic unchanged.

## References

- Ticket: `cursor/project/tickets/TICKET-004-training-and-evaluation-updates.md`
- Research: `cursor/project/research/2026-04-19-TICKET-004-training-and-evaluation-updates.md`
- Notebook: `artifacts/ipynb/phase6_hybrid_fused.ipynb`
- Model: `app/hybrid_fused_model.py`
- Labels: `app/risk_labels.py`
- Patch script: `scripts/patch_phase6_hybrid_fused_notebook.py`
