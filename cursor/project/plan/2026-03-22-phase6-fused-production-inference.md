# Phase 6 fused model — production inference and API restoration

## Overview

Restore a runnable **FastAPI** service and implement **production inference** for the **single PyTorch** `HybridFusedClassifier` trained in `artifacts/ipynb/phase6_hybrid_fused.ipynb`, using artifacts under a configurable directory (e.g. `artifacts/models/phase6_fused/`). This closes the gap documented in `cursor/project/research/2025-03-22-phase6-fused-vs-codebase-gaps.md`: empty `app/main.py`, deleted legacy `app/*` modules, missing env/docs for fused serving, and optional `model.safetensors` vs epoch checkpoints.

## Current State Analysis

- **`app/main.py`** is empty; there is **no** working HTTP API in the tree.
- **Legacy modules** (`app/model.py`, `app/config.py`, `app/schemas.py`, `app/ensemble.py`, etc.) are **deleted** in the working tree; README still describes `GET /health`, DistilBERT loading, and optional hybrid env vars — **not** the fused layout.
- **Fused artifacts** present locally include `config.json`, `fused_meta.json`, `word_index.json`, DistilBERT tokenizer files, `checkpoints/epoch_*.pt`, and metrics/plots. **`model.safetensors` may be absent** (as in this workspace); the notebook exports it in §12, and checkpoints contain `model_state` suitable for `load_state_dict` (see notebook §9–§11).
- **`HybridFusedClassifier`** exists **only inside the notebook** (not importable Python); production needs the same `nn.Module` in a **`.py` file** so the loader can instantiate and load weights.
- **Dependencies** (`requirements.txt`) already include `torch`, `transformers`, `safetensors`, `fastapi`, `pytest`, `httpx` — sufficient for fused inference **without** TensorFlow (TF remains for optional legacy Keras BiLSTM, not required for fused path).
- **No `Makefile`**; **no `tests/`** directory. Verification commands should use `pytest`, `uvicorn`, and `curl` directly.
- **`humanlayer thoughts sync`** is not available in this environment; plans live under `cursor/project/plan/` only unless you add that tooling later.

### Key Discoveries

- `artifacts/models/phase6_fused/config.json` — DistilBERT config plus fused keys: `max_len_bert` 256, `max_len_bilstm` 400, `vocab_size_bilstm` 20000, `embed_dim`, `lstm_hidden`, `fusion_hidden`, `num_labels` (see file in repo).
- `artifacts/models/phase6_fused/fused_meta.json` — Confirms preprocessing constants and `threshold` 0.5; `word_index_file` points at `word_index.json`.
- Notebook §7 / `build_collate_fn` — LSTM side uses **`re.findall(r"\w+", text.lower())`** with `word2idx`, OOV index **1**, pad **0**; BERT side uses `DistilBertTokenizerFast` with `max_length=256`. Inference **must match** this exactly.
- Checkpoint dict keys (epoch `.pt`): `epoch`, `model_state`, `optimizer_state`, `val_f1`, `history` (per notebook §9).

## Desired End State

1. **Importable model**: `HybridFusedClassifier` in a Python module (structure aligned with the notebook: `self.bert`, `self.embedding`, `self.lstm`, `self.classifier`).
2. **Loader**: Given a **fused artifact directory**, load tokenizer + `word_index.json`, construct the model from `fused_meta.json` / `config.json`, load weights from **`model.safetensors` when present**, otherwise from a **documented fallback** (see decisions below).
3. **Predict API**: Batch or single-text **scam probability** (class 1) using softmax on logits, with optional threshold from `fused_meta.json` / settings.
4. **FastAPI app**: At minimum `GET /`, `GET /health` (reports fused model loaded, device, artifact path), `POST /predict` (or equivalent) with Pydantic request/response models consistent with README expectations.
5. **Configuration**: New env vars for fused artifact directory and optional explicit weights file; `.env.example` and README updated so deployers are not misled by DistilBERT-only defaults.
6. **Automated tests**: Unit tests for tokenization alignment and forward shape; optional smoke test with tiny random state_dict or skipped-if-no-artifact integration test.

### Verification

- With `model.safetensors` (or a chosen `.pt`) and tokenizer files in place, `GET /health` shows the fused model loaded; `POST /predict` returns probabilities in `[0,1]`.
- Without weights file, app behavior is **explicit** (clear error at startup **or** documented degraded mode — pick one in implementation; **recommended: fail fast** if fused mode is requested but weights missing).

## Decisions (resolve research open questions)

| Question | Decision |
|----------|----------|
| `model.safetensors` missing in deployment | **Primary**: load from `model.safetensors` if present. **Fallback**: load `model_state` from a `.pt` file — either **`JOBSENTRY_PHASE6_FUSED_CHECKPOINT`** (path to one `epoch_XX.pt`) or, if unset, **highest `epoch_NN.pt`** under `artifact_dir/checkpoints/` (numeric sort by `NN`). |
| Restore full deleted `app/*` vs minimal surface | **Minimal**: implement fused inference + FastAPI + small `config`/`schemas` **without** restoring `app/ensemble.py`, Keras `bilstm`, or TF-IDF joblib stack **unless** a follow-up ticket explicitly requires backward-compatible multi-model serving in the same process. Document how to run **legacy** DistilBERT-only mode as a **separate** optional path only if you choose to reintroduce it later. |
| Fused vs DistilBERT-only routing | **Single primary path for this plan**: when **`JOBSENTRY_PHASE6_FUSED_DIR`** (name TBD in Phase 1) is set, the app uses **only** the fused predictor for `/predict`. When unset, behavior can be **stub/no model** or **future DistilBERT-only** — implementers should choose one and document it; **recommended**: unset → clear startup message that no model is configured (until DistilBERT loader is re-added) **or** optional second env to load old DistilBERT-only — **out of scope** unless explicitly added in a sub-phase. |

*Adjust env var names in code to match project naming; the table captures semantics.*

## What We're NOT Doing

- Re-adding the full **soft-voting ensemble** (DistilBERT + Keras BiLSTM + joblib) in this plan.
- Changing the **training notebook** or retraining the fused model.
- Committing large **weight files** to git (artifacts stay local/CI cache/object storage).
- Adding **Makefile** or **Docker** unless requested in a separate task.
- Running **`humanlayer thoughts sync`** (tool not available here).

## Implementation Approach

1. **Lift** `HybridFusedClassifier` from the notebook into a small module; keep hyperparameters **constructor-driven** from JSON on disk (same defaults as `fused_meta.json`).
2. **Implement** `FusedScamPredictor` (name flexible) that: tokenizes text for both branches, runs `model.eval()` forward, returns **numpy/torch** probabilities for the positive class.
3. **Wire** FastAPI with Pydantic settings (`pydantic-settings`) for paths and batch limits.
4. **Test** tokenization golden cases (short string → fixed LSTM id pattern with a **tiny** toy `word_index` in tests) and mock tensor forward.
5. **Document** env vars and operational notes (CPU vs GPU, memory).

---

## Phase 1: Extract `HybridFusedClassifier` and shared preprocessing

### Overview

Make the architecture importable and unit-testable without the notebook.

### Changes Required

#### 1. New module e.g. `app/hybrid_fused_model.py` (or `app/models/hybrid_fused.py`)

**Content**:

- `HybridFusedClassifier` matching notebook cell §6 (`DistilBertModel`, mean pool over non-pad tokens, `nn.Embedding`, `nn.LSTM` bidirectional, fusion MLP).
- Helper **`tokenize_words(text: str) -> list[str]`** using `re.findall(r"\w+", str(text).lower())`.
- **`texts_to_lstm_batch(texts, word2idx, max_len, oov_idx=1, pad_idx=0)`** → `LongTensor [B, max_len]` mirroring notebook §7.

**File**: `app/hybrid_fused_model.py`  
**Changes**: New file; no behavioral drift from notebook.

#### 2. Optional tiny `app/preprocessing.py`

If you prefer separation: move only regex word tokenization + LSTM padding here; keep imports minimal.

### Success Criteria

#### Automated Verification

- [x] `python -c "from app.hybrid_fused_model import HybridFusedClassifier; ..."` runs (instantiate with small vocab, dummy forward).
- [x] `pytest` tests pass for forward shape: `logits.shape == (B, 2)` with random integer tensors of correct lengths.

#### Manual Verification

- [ ] Diff notebook §6 vs module: attribute names and pooling logic match.

**Implementation Note**: Pause for a quick review of the extracted class against the notebook before wiring the loader.

---

## Phase 2: Artifact loading and weight resolution

### Overview

Load tokenizer, `word_index`, hyperparameters, and weights with **`model.safetensors` preferred**, `.pt` fallback.

### Changes Required

#### 1. New module e.g. `app/fused_loader.py`

**Responsibilities**:

- Resolve weight source:
  1. If `artifact_dir / "model.safetensors"` exists → `safetensors.torch.load_file` → `load_state_dict`.
  2. Else resolve `.pt`: use `JOBSENTRY_PHASE6_FUSED_CHECKPOINT` if set; else `max(checkpoints/epoch_*.pt)` by epoch number.
  3. `torch.load(..., map_location=device, weights_only=False)` → `state["model_state"]` → `load_state_dict`.
- Load `DistilBertTokenizerFast.from_pretrained(artifact_dir)` (tokenizer files present).
- Load JSON: `fused_meta.json` (required) for `max_len_bert`, `max_len_bilstm`, `vocab_size`, architecture dims; merge with `config.json` fused keys if needed.
- Build `word2idx` from `word_index.json` (format: `{"word": idx, ...}` per notebook — verify against actual file; handle `"<OOV>"` if present).

**File**: `app/fused_loader.py`  
**Changes**: New file.

#### 2. Device selection

Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` unless overridden by env (optional `JOBSENTRY_DEVICE`).

### Success Criteria

#### Automated Verification

- [x] Unit test with **temporary directory**: fake minimal `word_index.json`, copy or mock tiny tensors — or **mock** `load_state_dict` to assert correct file pick order (safetensors vs pt).
- [x] `pytest -q` passes.

#### Manual Verification

- [ ] On a machine with real `artifacts/models/phase6_fused/`, loader prints which weight source was used and loads without shape errors.

---

## Phase 3: Pydantic settings and predictor facade

### Overview

Centralize configuration and expose **`predict_proba(texts: list[str]) -> list[float]`** (scam probability).

### Changes Required

#### 1. `app/config.py`

- `phase6_fused_dir: Optional[Path] = None` (or required for fused-only deployments).
- Optional `phase6_fused_checkpoint: Optional[Path] = None` override.
- `max_batch_size`, `confidence_threshold` aligned with `fused_meta.json` default 0.5 where applicable.

#### 2. `app/fused_predictor.py` (or methods on loader class)

- Batch texts: BERT tokenizer batch encode with `max_length` from meta, truncation/padding True.
- LSTM ids from shared preprocessing.
- `torch.no_grad()`, `softmax` on logits, return **P(scam)** = prob of label 1.

### Success Criteria

#### Automated Verification

- [x] With mocked model, `predict_proba` returns correct list length and values in `[0,1]`.

#### Manual Verification

- [ ] Compare one string’s probability with notebook inference cell (if run side-by-side) within float tolerance.

---

## Phase 4: FastAPI application

### Overview

Restore HTTP surface expected by README: root, health, predict.

### Changes Required

#### 1. `app/schemas.py`

- Request: job post fields needed for `combined_text` (or a single `text` field) — **match** prior API if known from thesis/docs; otherwise minimal `{"text": str}` or `{"title", "description", ...}` merged into `combined_text` with same rule as training CSV (`combined_text` column). *Inspect `merged_train.csv` column convention if multiple fields.*

#### 2. `app/main.py`

- `FastAPI` app, lifespan or startup: load fused predictor when `phase6_fused_dir` set.
- `GET /` — service info.
- `GET /health` — `status`, `model_loaded`, `mode: "phase6_fused"`, artifact path.
- `POST /predict` — accept batch, return probabilities and optional binary labels using threshold.

#### 3. `app/__init__.py`

- Empty or package marker if needed for imports.

### Success Criteria

#### Automated Verification

- [x] `httpx` async tests against `TestClient` for `/health` and `/predict` with dependency overrides or temp artifacts.
- [x] `uvicorn app.main:app` starts without error when config is valid.

#### Manual Verification

- [ ] Swagger UI at `/docs` works; sample request returns 200.

---

## Phase 5: Documentation, environment, and tests layout

### Overview

Align `.env.example` and README with fused deployment; add `tests/` tree.

### Changes Required

#### 1. `.env.example`

- Document `JOBSENTRY_PHASE6_FUSED_DIR` (and checkpoint override if implemented).
- Note DistilBERT-only vars as **legacy / not used** when fused dir is set, or remove duplicate confusion with a short comment block.

#### 2. `README.md`

- Section: **Phase 6 fused model (recommended)** vs legacy DistilBERT-only.
- Update paths: `artifacts/models/phase6_fused` example.
- Health check instructions: `model_loaded`, fused-specific messages.

#### 3. `tests/test_hybrid_fused_model.py`, `tests/test_api.py`

- As outlined above; use `pytest`.

### Success Criteria

#### Automated Verification

- [x] `pytest -q` from repo root passes in CI/local.
- [x] No linter errors on touched files (if Ruff/flake8 exists; optional).

#### Manual Verification

- [ ] New developer can follow README to run the server against local `phase6_fused` artifacts.

---

## Testing Strategy

### Unit Tests

- `HybridFusedClassifier` forward shapes; word tokenization regex golden cases.
- Loader weight-resolution order (safetensors vs pt) with mocks.
- `predict_proba` list length and value range.

### Integration Tests

- `TestClient` `/health` and `/predict` with mocked predictor or tiny artifact fixture.

### Manual Testing Steps

1. Point env at real `artifacts/models/phase6_fused/` (with `model.safetensors` **or** a chosen `epoch_XX.pt`).
2. Run `uvicorn app.main:app --reload`, call `GET /health` and `POST /predict` with a known job description.
3. Confirm latency and memory acceptable on target hardware (CPU vs GPU).

## Performance Considerations

- Fused forward runs **DistilBERT + LSTM** each request; cap batch size via settings.
- Prefer **GPU** in production for DistilBERT; document CPU fallback latency.
- Use `torch.inference_mode()` and half precision only if validated (optional, not required initially).

## Migration Notes

- Deployments that relied on **deleted** `app/model.py` need a **fresh** deploy using fused env vars or a restored DistilBERT-only module — **out of scope** unless added explicitly.
- Ensure training export pipeline eventually writes **`model.safetensors`** to the artifact dir to avoid relying on epoch checkpoints long term.

## References

- Research: `cursor/project/research/2025-03-22-phase6-fused-vs-codebase-gaps.md`
- Implementation summary: `cursor/project/implementation/2025-03-21-NA-hybrid-bilstm-distilbert-fused-training.md`
- Prior training plan: `cursor/project/plan/2025-03-21-hybrid-bilstm-distilbert-fused-training.md`
- Notebook: `artifacts/ipynb/phase6_hybrid_fused.ipynb` (§6 module, §7 collate, §9 checkpoints, §12 export)
- Artifacts: `artifacts/models/phase6_fused/config.json`, `fused_meta.json`, `word_index.json`
