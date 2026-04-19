# TICKET-005: Inference contract and prediction outputs — implementation plan

## Overview

Close [`cursor/project/tickets/TICKET-005-inference-contract-and-serving.md`](../tickets/TICKET-005-inference-contract-and-serving.md) by (1) exposing a **single inference helper** that goes from **raw text → tokenizer → forward → `predicted_class` / `predicted_label` / per-class probabilities / `confidence`**, (2) persisting **serialized class-id → label metadata** alongside `fused_meta.json`, (3) making the **HTTP and prose documentation** match the **3-class softmax** behavior already implemented in `app/`, (4) adding **threshold policy notes** for optional business rules (including when `map_binary_to_risk` and env thresholds apply), and (5) adding **notebook inference sanity cells** that print sample outputs aligned with `POST /predict`.

## Current State Analysis

- **`FusedScamPredictor`** (`app/fused_predictor.py`): Implements **`predict_risk_distribution(texts) → list[tuple[float,float,float]]`** (softmax triples). No single method returns class id, string label, and confidence together.

- **`POST /predict`** (`app/main.py`): Calls **`predict_risk_distribution`**, then **`class_from_softmax_triple`** per row (`app/risk_labels.py`), then builds **`PredictResponse`**. One-text and batch both work via **`posts: [ ... ]`**.

- **`PredictResponse`** (`app/schemas.py`): Already defines **`predicted_class`**, **`predicted_label`**, **`legit_probability`**, **`warning_probability`**, **`fraud_probability`**, **`confidence`**, **`warnings`**. Field descriptions for **`confidence`** already match **`max` of the three softmax probabilities**.

- **Thresholds** (`app/config.py`): **`JOBSENTRY_WARN_THRESHOLD`**, **`JOBSENTRY_FRAUD_THRESHOLD`**, **`JOBSENTRY_CONFIDENCE_THRESHOLD`** exist. **`map_binary_to_risk`** (`app/risk_labels.py`) implements binary→3-way mapping using warn/fraud thresholds; it is **not** used in **`main.predict`**. README still describes a **binary `P(scam)`** API path and **one-hot** probabilities — **documentation drift** vs code.

- **`fused_meta.json`**: Contains **`num_labels`**, hyperparameters, and metrics; **no** explicit **`class id → label`** map in JSON.

- **Notebook** (`artifacts/ipynb/phase6_hybrid_fused.ipynb`): Training/eval/export exist; **no** dedicated cell that runs **sample raw texts** through the same contract as production and prints **`predicted_*` + probabilities** for sanity checking.

### Key Discoveries

- [`README.md`](../../README.md) lines 68–72 and example lines 95–104 describe **binary head + threshold mapping** and **one-hot** probabilities; implementation uses **`HybridFusedClassifier`** **3-logit softmax** and raw **three probabilities** per class (`tests/test_api.py` fake triple asserts this shape).

- [`scripts/patch_phase6_hybrid_fused_notebook.py`](../../scripts/patch_phase6_hybrid_fused_notebook.py) (if present) must stay in sync with notebook export blocks that write **`fused_meta.json`**.

## Desired End State

1. **Inference helper**: One callable (preferably on **`FusedScamPredictor`** or a thin wrapper in `app/fused_predictor.py`) that accepts **`list[str]`** (including length 1), runs **tokenizer + `model` forward + softmax**, and returns a structured row per input: **`predicted_class`**, **`predicted_label`**, **`legit_probability`**, **`warning_probability`**, **`fraud_probability`**, **`confidence`** — using the same mapping rules as **`POST /predict`** (i.e. **`class_from_softmax_triple`**).

2. **`POST /predict`**: Implemented **via** that helper (or shared builder) so the HTTP path and helper cannot drift.

3. **Serialized metadata**: **`fused_meta.json`** (written by the notebook export path and checked in under `artifacts/models/phase6_fused/` when regenerated) includes a stable **class-id → label** list or map (e.g. index `0..num_labels-1` → **`"legit"`**, **`"warning"`**, **`"fraud"`**). **`load_fused_artifacts`** may **log** or **validate** consistency with **`app.risk_labels`** expectations when **`num_labels == 3`**.

4. **Documentation**: **`README.md`** Predict section and **`PredictResponse`** field descriptions describe **3-class softmax outputs**; example JSON uses **probabilities that sum to ~1** per post. A short **“Threshold / optional business rules”** subsection explains **`map_binary_to_risk`**, **`JOBSENTRY_*_THRESHOLD`**, and example **borderline** policies (documentation-first; no change to default inference unless explicitly chosen).

5. **Notebook**: New section with **2–3 sample strings** (e.g. clearly legit vs scam-like), printing the same fields clients see from **`POST /predict`** (or calling **`FusedScamPredictor`** loaded from **`FUSED_OUT_DIR`**).

### Verification

- **`pytest -q`** passes.
- Manual: **`uvicorn app.main:app`**, **`POST /predict`** in **`/docs`** matches documented schema; notebook cells run top-to-bottom after export (or from a defined checkpoint) and show sample outputs.

## What We're NOT Doing

- Switching **`POST /predict`** to **`map_binary_to_risk`** by default (would require a **binary** head or extra score; out of scope unless product redefines the model).

- Changing **`HybridFusedClassifier`** architecture or retraining weights as part of this ticket.

- Full **frontend** or **OpenAPI client** codegen beyond what FastAPI already exposes.

- Running **`humanlayer thoughts sync`** — not present in this repository; skip.

## Implementation Approach

1. Introduce a small **immutable row type** (e.g. **`NamedTuple`** or **`dataclass`** in `app/fused_predictor.py` or `app/risk_labels.py`) for **one row** of inference output.

2. Add **`FusedScamPredictor.predict_full(texts) -> list[Row]`** (name adjustable) that internally calls **`predict_risk_distribution`** + **`class_from_softmax_triple`** and fills **probabilities** from the triple.

3. Refactor **`main.predict`** to assemble **`PredictResponse`** from **`predict_full`** (flatten to parallel lists).

4. Extend **notebook** `fused_meta` dict and **committed** **`fused_meta.json`** with **`risk_class_labels`** (or equivalent key name chosen once and used everywhere).

5. **README** + optional **`.env.example`** comments: align prose with softmax; document optional threshold-based policy for **`map_binary_to_risk`** and when env vars matter.

6. **Notebook**: add **§ Inference sanity** (or similar) with imports from **`app`** or duplicated **minimal** forward aligned with **`_predict_risk_batch`**.

---

## Phase 1: Inference helper + `main` refactor

### Overview

Single entry point for **text → full prediction row**; **`POST /predict`** uses it.

### Changes Required

#### 1. `app/fused_predictor.py` (and/or `app/risk_labels.py`)

**Changes**: Define **`RiskPrediction`** (or similar) with fields: **`predicted_class: int`**, **`predicted_label: str`**, **`legit_probability`**, **`warning_probability`**, **`fraud_probability`**, **`confidence: float`**. Add **`FusedScamPredictor.predict_full(self, texts: list[str]) -> list[RiskPrediction]`** that:
- Returns `[]` for empty **`texts`** (match **`predict_risk_distribution`** behavior).
- For non-empty, reuses **`predict_risk_distribution`** (or internal batching) and **`class_from_softmax_triple`** from **`app.risk_labels`**.

#### 2. `app/main.py`

**Changes**: Replace the manual loop over **`triples`** + **`class_from_softmax_triple`** with **`rows = predictor.predict_full(texts)`** and build lists from **`rows`**.

#### 3. `tests/test_fused_predictor.py`

**Changes**: Add tests for **`predict_full`** on a **mock** or **tiny** predictor fixture consistent with existing tests (inject fake **`predict_risk_distribution`** if needed, or use existing model tests).

#### 4. `tests/test_api.py`

**Changes**: Keep assertions on **`POST /predict`** behavior unchanged unless response shape changes (it should not).

### Success Criteria

#### Automated Verification

- [x] **`pytest -q`** passes from repo root.

#### Manual Verification

- [ ] **`POST /predict`** with injected fake predictor still returns expected parallel arrays (spot-check one case in **`/docs`** if desired).

**Implementation Note**: Pause after automated verification for optional human spot-check before Phase 2.

---

## Phase 2: Serialized `risk_class_labels` in `fused_meta.json`

### Overview

Persist **class index → label** strings next to **`num_labels`** so artifacts are self-describing for downstream tools.

### Changes Required

#### 1. `artifacts/ipynb/phase6_hybrid_fused.ipynb`

**Changes**: In the cell that builds **`fused_meta`** (section **12.3** / export), add a key such as **`"risk_class_labels": ["legit", "warning", "fraud"]`** (must match **`num_labels`** and **`app.risk_labels`** order).

#### 2. `artifacts/models/phase6_fused/fused_meta.json`

**Changes**: Regenerate or hand-merge the same key after notebook run so the committed artifact matches export.

#### 3. `app/fused_loader.py`

**Changes**: After loading JSON, if **`risk_class_labels`** is present and **`len` != **`num_labels`**, **raise** or **log warning** (choose **raise** in strict mode only if tests require; default recommendation: **`logger.warning`** to avoid breaking older artifacts without the key).

#### 4. `tests/test_fused_loader.py`

**Changes**: Fixture or test with **`risk_class_labels`** present; assert loader behavior.

#### 5. `scripts/patch_phase6_hybrid_fused_notebook.py`

**Changes**: If the script embeds the **`fused_meta`** export snippet, update it in lockstep with the notebook.

### Success Criteria

#### Automated Verification

- [x] **`pytest -q`** passes, including **`tests/test_fused_loader.py`**.

#### Manual Verification

- [ ] Open **`fused_meta.json`** and confirm **`risk_class_labels`** exists and order matches **`schemas.PredictResponse`** / **`risk_labels.py`**.

---

## Phase 3: Documentation and threshold policy notes

### Overview

Align **README** and optional **env** comments with **3-class softmax**; document **optional** **`map_binary_to_risk`** and **threshold** env vars without claiming they drive default **`POST /predict`**.

### Changes Required

#### 1. `README.md`

**Changes**:
- Replace table rows that say **binary** / **one-hot** with: per-class values are **softmax probabilities** (sum **~1** per post); **`predicted_*`** from **argmax**; **`confidence`** = **max** of the three.
- Remove or rewrite the **Mapping** paragraph that states the model outputs **P(scam)** only.
- Add **“Optional threshold policies”**: **`map_binary_to_risk`** in **`app/risk_labels`** for **scalar `P(scam)`** workflows; **`JOBSENTRY_WARN_THRESHOLD` / `JOBSENTRY_FRAUD_THRESHOLD`** apply there; **`JOBSENTRY_CONFIDENCE_THRESHOLD`** reserved for future policy (or document if unused). Example **borderline** rule: escalate when **top two softmax values** within **ε** (narrative only unless you implement — default **narrative only**).

- Fix **example JSON** so **`legit_probability` + `warning_probability` + `fraud_probability`** sum to **~1.0** (e.g. **0.91 / 0.06 / 0.03** with **`confidence`** **0.91**).

#### 2. `.env.example`

**Changes**: Comment block clarifying which thresholds affect **`map_binary_to_risk`** vs **future** policy, not the default **`POST /predict`** softmax path.

#### 3. `app/schemas.py`

**Changes**: Ensure **`Field(description=...)`** on probability lists states **softmax** (not one-hot) if not already explicit.

### Success Criteria

#### Automated Verification

- [x] **`pytest -q`** passes (docs-only changes should not break tests).

#### Manual Verification

- [ ] Read **`README`** Predict section end-to-end; confirm no statement contradicts **`app/main.py`** + **`fused_predictor.py`**.

---

## Phase 4: Notebook inference sanity cells

### Overview

Satisfy acceptance **“Sample predictions are shown in notebook for sanity checking.”**

### Changes Required

#### 1. `artifacts/ipynb/phase6_hybrid_fused.ipynb`

**Changes**: Add a short section **after** model + tokenizer are available (post-load or post-export):
- **Option A (preferred)**: **`sys.path`** insert repo root, **`from app.fused_predictor import FusedScamPredictor`**, **`FusedScamPredictor.from_artifact_dir(FUSED_OUT_DIR, ...)`**, then **`predict_full`** on **2–3** hard-coded **`combined_text`** strings; **`print`** or **`display`** rows.
- **Option B**: Inline **torch** forward matching **`_predict_risk_batch`**, then call **`class_from_softmax_triple`** — duplicates logic; use only if **importing `app`** from notebook is blocked.

#### 2. `scripts/patch_phase6_hybrid_fused_notebook.py`

**Changes**: Mirror new cells if the script regenerates the notebook.

### Success Criteria

#### Automated Verification

- [x] **`pytest -q`** (notebook not in CI by default).

#### Manual Verification

- [ ] Run the new cells locally; outputs show **non-degenerate** softmax triples and **labels** consistent with **argmax**.

---

## Testing Strategy

### Unit Tests

- **`predict_full`**: Known triple → expected **class** / **label** / **confidence** / probabilities.
- **`fused_loader`**: With **`risk_class_labels`** key present.

### Integration Tests

- Existing **`tests/test_api.py`** coverage for **`POST /predict`** remains the regression guard.

### Manual Testing Steps

1. Set **`JOBSENTRY_PHASE6_FUSED_DIR`** to **`artifacts/models/phase6_fused`**, start **`uvicorn`**, call **`POST /predict`** with **`{"posts":[{"text":"..."}]}`**.
2. Execute new notebook section; compare one sample’s probabilities to **`curl`** response for the same string.

## Performance Considerations

**`predict_full`** should not add extra forward passes; it should reuse **`predict_risk_distribution`** once per request batch.

## Migration Notes

- **Clients** that relied on README’s **one-hot** description should treat fields as **softmax** — document under **README** migration line if needed.
- Older **`fused_meta.json`** without **`risk_class_labels`**: loader should **warn** and continue (unless you choose strict mode).

## References

- Ticket: [`cursor/project/tickets/TICKET-005-inference-contract-and-serving.md`](../tickets/TICKET-005-inference-contract-and-serving.md)
- Related ticket: [`cursor/project/tickets/TICKET-004-training-and-evaluation-updates.md`](../tickets/TICKET-004-training-and-evaluation-updates.md)
- Prior research: [`cursor/project/research/2026-04-19-TICKET-005-inference-contract-and-serving.md`](../research/2026-04-19-TICKET-005-inference-contract-and-serving.md)
- Code: [`app/fused_predictor.py`](../../app/fused_predictor.py), [`app/main.py`](../../app/main.py), [`app/schemas.py`](../../app/schemas.py), [`app/risk_labels.py`](../../app/risk_labels.py), [`app/fused_loader.py`](../../app/fused_loader.py), [`README.md`](../../README.md)
