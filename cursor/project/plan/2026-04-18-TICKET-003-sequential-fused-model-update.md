# TICKET-003: Sequential DistilBERT → BiLSTM fused model — Implementation Plan

## Overview

Replace the current **parallel** `HybridFusedClassifier` (DistilBERT mean-pool **concat** word-embedding BiLSTM) with a **sequential** stack: DistilBERT `last_hidden_state` → **BiLSTM** → **masking-aware pooling** → MLP classifier with **`num_labels = 3`**, while keeping **`freeze_bert` / `unfreeze_bert`**. Update [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](../../artifacts/ipynb/phase6_hybrid_fused.ipynb) and the importable [`app/hybrid_fused_model.py`](../../app/hybrid_fused_model.py) in lockstep, add shape documentation, and wire loading/inference so checkpoints train-exported from the notebook load in the app.

**Label semantics (fixed):** `0 = legit`, `1 = warning`, `2 = fraud`, aligned with [`app/risk_labels.py`](../../app/risk_labels.py) and [`app/schemas.py`](../../app/schemas.py), and with `risk_class` in [`artifacts/data/processed/merged_train.csv`](../../artifacts/data/processed/merged_train.csv) (values `0`/`1`/`2`).

## Current State Analysis

- [`app/hybrid_fused_model.py`](../../app/hybrid_fused_model.py) (`HybridFusedClassifier`): DistilBERT → mean-pool → `bert_vec`; **separate** `lstm_ids` → `nn.Embedding` → BiLSTM → `lstm_vec`; **concat** → MLP; **`num_labels` default 2** (lines 42–119).
- [`app/fused_loader.py`](../../app/fused_loader.py): builds the model from `fused_meta.json`, **requires** `word_index.json`, passes `vocab_size`, `embed_dim`, etc. (lines 115–136).
- [`app/fused_predictor.py`](../../app/fused_predictor.py): tokenizes with DistilBERT **and** builds `lstm_ids` via `texts_to_lstm_batch`; forward `(input_ids, attention_mask, lstm_ids)`; **`predict_proba`** uses **`softmax(logits)[:, 1]`** as binary P(scam) (lines 72–92).
- [`app/main.py`](../../app/main.py): uses `map_binary_to_risk(p_scam, heuristics, …)` to fill the 3-way API (`predicted_class` 0/1/2) (lines 129–152).
- Research: [`cursor/project/research/2026-04-18-TICKET-003-sequential-fused-model-update.md`](../research/2026-04-18-TICKET-003-sequential-fused-model-update.md).

## Desired End State

1. **`HybridFusedClassifier.forward(self, input_ids, attention_mask)`** — no `lstm_ids`; output **`[batch_size, 3]`** logits.
2. **BiLSTM** `input_size = DistilBERT hidden size** (768 for `distilbert-base-uncased`), consuming the **token sequence** from DistilBERT outputs (not a separate word vocabulary).
3. **Masking-aware pooling** applied to the **BiLSTM sequence outputs** (not mean-pooling BERT alone for the final vector). Use **`torch.nn.utils.rnn.pack_padded_sequence`** / **`pad_packed_sequence`** (or an equivalent correct masking strategy) so LSTM state does not treat padded tokens as real content; then **masked mean** over the padded BiLSTM output using **`attention_mask`**.
4. **Classifier head** maps pooled BiLSTM vector → `fusion_hidden` → **`num_labels = 3`** (dropout/ReLU pattern can match the existing MLP style).
5. **`freeze_bert` / `unfreeze_bert`** unchanged in spirit; training notebook keeps a **freeze-then-unfreeze** epoch schedule.
6. **Checkpoint save/load:** training exports `model.safetensors` (and/or `epoch_*.pt` with `model_state`); [`load_fused_artifacts`](../../app/fused_loader.py) loads into the **new** module; **`load_state_dict(..., strict=True)`** succeeds for weights produced by the updated notebook.
7. **Deliverables:** updated notebook cells (model §6, data/collate §7, training §8+, export §12), plus **inline notes** (markdown or comments) documenting tensor shapes at: BERT `last_hidden_state`, packed LSTM I/O, pooled vector, logits.

### Key Discoveries

- `merged_train.csv` exposes **`risk_class`** with distribution suitable for 3-class training (0/1/2).
- The HTTP layer already exposes **3-way** `PredictResponse`; today the **model** is binary and [`map_binary_to_risk`](../../app/risk_labels.py) derives tiers from P(scam) + heuristics. A **native 3-head** model allows mapping **model softmax** directly to `legit_probability` / `warning_probability` / `fraud_probability` with an explicit policy for heuristics (see Phase 3).

## What We're NOT Doing

- **Guaranteeing** a particular validation F1 or production accuracy (training quality is iterative).
- **Loading old** `phase6_fused` **2-class** `model.safetensors` into the new architecture (**state dict layout and constructor args change** — treat as **breaking**; keep previous artifacts on an older tag/branch if rollback is needed).
- **Optional TensorFlow / Keras** BiLSTM checkpoints under `artifacts/models/phase6_merged/` — unrelated to this PyTorch fused path.
- **Automated full retrain** in CI (dataset + GPU time is out of scope for this plan’s execution).

## Implementation Approach

1. Implement the **new** `HybridFusedClassifier` and **unit tests** with a **mock DistilBERT** (no download in tests).
2. **Simplify fused artifacts**: `fused_meta.json` documents `num_labels: 3` and drops **word-index / vocab / `max_len_bilstm`** requirements for the sequential model; adjust **loader + predictor** so inference only needs DistilBERT tokenizer + weights.
3. **Notebook**: rewire dataset labels to `risk_class`, loss = `CrossEntropyLoss`, export updated `fused_meta.json` and weights.
4. **API**: when the loaded model is 3-class, fill response probabilities from **softmax** over three logits; define a single, documented rule for **heuristic warnings** vs model (e.g. still surface `warnings` strings; **predicted class** from `argmax` of model probs, or optionally bump to `warning` when heuristics fire — **pick one policy in Phase 3 and test it**).

---

## Phase 1: Model core + unit tests

### Overview

Replace parallel forward with sequential BERT → BiLSTM → masked pool → MLP; default **`num_labels=3`**; **`forward` signature** `(input_ids, attention_mask)` only; remove **`nn.Embedding` word branch** and **`lstm_ids`** from the module API.

### Changes Required

#### 1. `app/hybrid_fused_model.py`

**File**: [`app/hybrid_fused_model.py`](../../app/hybrid_fused_model.py)

**Changes**:

- **`__init__`**: Remove `vocab_size` and `embed_dim` (and any word embedding table). Set `self.lstm = nn.LSTM(bert_dim, lstm_hidden, batch_first=True, bidirectional=True)` (with `bert_dim` from `self.bert.config.hidden_size` or `.dim` as today). Set classifier input dimension to **`lstm_hidden * 2`** (bidirectional), not `bert_dim + lstm_hidden * 2`.
- **`forward`**:  
  `bert_out = self.bert(...)`  
  `seq = bert_out.last_hidden_state`  # `[B, L, bert_dim]`  
  Run BiLSTM with **packed padded** inputs derived from `attention_mask` (lengths = `mask.sum(dim=1)`), then **`pad_packed_sequence`** back to `[B, L, 2*lstm_hidden]` and apply **`_mean_pool(lstm_seq, attention_mask)`** to obtain `[B, 2*lstm_hidden]`.  
  `logits = self.classifier(pooled)` → `[B, 3]`.
- **Docstring**: Update inputs/output shapes; state **3-class** logits.
- **Optional helpers**: Keep `tokenize_words` / `texts_to_lstm_batch` **only if** still used elsewhere; if nothing imports them after predictor change, **remove** or move to a `legacy_` module to avoid dead code (prefer **delete** if unused).

```python
# Illustrative structure (exact names may match project style):
# pooled: Tensor [B, 2 * lstm_hidden]
# logits: Tensor [B, num_labels]  # num_labels default 3
```

#### 2. `tests/test_hybrid_fused_model.py`

**File**: [`tests/test_hybrid_fused_model.py`](../../tests/test_hybrid_fused_model.py)

**Changes**:

- Mock `DistilBertModel.from_pretrained` to return a small module whose `forward` yields deterministic `last_hidden_state` shape `[B, L, 768]`.
- Call **`model(input_ids, attention_mask)`** (two args only).
- Assert **`out.shape == (b, 3)`** when `num_labels=3`.
- Add a test that **mask** changes pooled output vs all-ones mask (sanity for masking path).
- Remove tests that only exist for **word** `lstm_ids` / `texts_to_lstm_batch` if those helpers are removed.

### Success Criteria

#### Automated Verification

- [x] `pytest tests/test_hybrid_fused_model.py -q` passes.
- [x] `pytest tests/ -q` may still fail until Phases 2–3 update mocks — after full plan, **entire** `pytest` suite passes.

#### Manual Verification

- [ ] Quick `python -c` instantiate (with patched or tiny BERT) runs forward without shape errors.

**Implementation Note:** After Phase 1 automated checks pass, pause for confirmation before merging dependent phases if your team gates on incremental PRs.

---

## Phase 2: Fused artifacts, loader, and predictor

### Overview

Align **disk format** and **Python loaders** with the sequential model: no `word_index.json` requirement; **`fused_meta.json`** carries `num_labels: 3` and any lengths needed (e.g. `max_len_bert` only). Update **`FusedScamPredictor`** to tokenize once and call the **two-arg** forward.

### Changes Required

#### 1. `app/fused_loader.py`

**File**: [`app/fused_loader.py`](../../app/fused_loader.py)

**Changes**:

- **`load_fused_artifacts`**: If `word_index` is **no longer required**, **stop requiring** `word_index.json` when `fused_meta` indicates sequential 3-class (e.g. `num_labels == 3` or explicit `"arch": "sequential_distilbert_bilstm_v1"`). Return **`word2idx` as `{}`** or **remove** `word2idx` from the return tuple — **prefer adjusting the return type** once all callers are updated to avoid dummy globals.
- **`HybridFusedClassifier(...)`** constructor call: pass only arguments the new `__init__` accepts (`lstm_hidden`, `fusion_hidden`, `num_labels`, `dropout`, `distilbert_name`, etc.).
- Document in code comments that **old 2-class fused artifacts** are **not** loadable by this module version without a migration shim (shim **out of scope** unless explicitly added).

#### 2. `app/fused_predictor.py`

**File**: [`app/fused_predictor.py`](../../app/fused_predictor.py)

**Changes**:

- Remove **`texts_to_lstm_batch`** and any **`word2idx` / `max_len_bilstm`** from the predictor **unless** kept for unrelated legacy code paths.
- **`_predict_proba_batch`**: `logits = model(input_ids, attention_mask)`; **`probs = softmax(logits, dim=-1)`** with shape `[B, 3]`.
- Replace **`predict_proba` → scalar P(scam)** with one of:
  - **`predict_risk_distribution(texts) -> list[tuple[float,float,float]]`** (legit, warning, fraud), **or**
  - Keep **`predict_proba`** as **P(fraud) = probs[:, 2]** for minimal churn, **and** add a method returning full triples for `main.py`.

Choose **one** public API and use it consistently in Phase 3.

#### 3. `tests/test_fused_loader.py` / `tests/test_fused_predictor.py`

**Files**: [`tests/test_fused_loader.py`](../../tests/test_fused_loader.py), [`tests/test_fused_predictor.py`](../../tests/test_fused_predictor.py)

**Changes**:

- Update fixtures: temporary artifact dirs with **`fused_meta.json`** matching the new schema; **drop** fake `word_index.json` where no longer read.
- Mock models: **`forward(self, input_ids, attention_mask)`** returning `[B, 3]`.

### Success Criteria

#### Automated Verification

- [x] `pytest tests/test_fused_loader.py tests/test_fused_predictor.py -q` passes.

#### Manual Verification

- [ ] Export a tiny checkpoint from the updated notebook (or scripted `state_dict`) into a temp dir and run `load_fused_artifacts` + one batch inference.

---

## Phase 3: FastAPI + risk mapping + schemas touchpoints

### Overview

Use **native 3-way softmax** for `legit_probability`, `warning_probability`, `fraud_probability` when `num_labels == 3`. Define **confidence** (e.g. `max_k softmax`) and how **heuristic** `compute_warnings` interacts with **predicted_class** (recommended default: **`predicted_class = argmax(model_probs)`**; keep **`warnings`** as explanatory strings only unless product requires heuristic-based tier bumps).

### Changes Required

#### 1. `app/main.py`

**File**: [`app/main.py`](../../app/main.py)

**Changes**:

- Branch on predictor/model metadata: if **3-class**, fill lists from **model probabilities**; if **legacy 2-class** still needed, keep `map_binary_to_risk` path — **if 2-class support is fully removed**, delete the binary branch and **`map_binary_to_risk`** usage for fused inference (heuristics may remain for `warnings` only).

#### 2. `app/schemas.py`

**File**: [`app/schemas.py`](../../app/schemas.py)

**Changes**:

- Update `PredictResponse.confidence` description if confidence is no longer `max(P(scam), 1-P(scam))`.

#### 3. `tests/test_api.py`

**File**: [`tests/test_api.py`](../../tests/test_api.py)

**Changes**:

- Update **`_FakePred`** and assertions to match the new predictor API and expected probabilities.

### Success Criteria

#### Automated Verification

- [x] `pytest tests/test_api.py -q` passes.
- [x] `pytest tests/ -q` passes repository-wide.

#### Manual Verification

- [ ] `uvicorn app.main:app` with `JOBSENTRY_PHASE6_FUSED_DIR` pointing at a **new** artifact dir returns coherent 3-way probabilities on `/predict`.

---

## Phase 4: Notebook + export + shape notes

### Overview

Update [`artifacts/ipynb/phase6_hybrid_fused.ipynb`](../../artifacts/ipynb/phase6_hybrid_fused.ipynb): **§6** `HybridFusedClassifier` matches **`app/hybrid_fused_model.py`**; **§7** dataset/collate uses **only** DistilBERT tokenizer (no parallel word-id tensor); labels from **`risk_class`**; **§8+** optimizer groups still split **BERT** vs **head**; **§12** export writes `fused_meta.json` with **`num_labels: 3`**, tokenizer files, `model.safetensors`, and checkpoint format compatible with [`load_weights_into_model`](../../app/fused_loader.py). Add a **markdown “Tensor shape cheat sheet”** cell (BERT `[B,L,768]` → packed LSTM → `[B,L,2H]` → pooled `[B,2H]` → logits `[B,3]`).

### Success Criteria

#### Automated Verification

- [ ] Notebook JSON executes through model forward and export cells in the team’s standard environment (may be marked **manual** if GPUs/colab-only).

#### Manual Verification

- [ ] Exported directory loads via `FusedScamPredictor.from_artifact_dir`.
- [ ] Shape notes are readable without running training.

---

## Testing Strategy

### Unit Tests

- Forward shape `[B, 3]`; masking/packing sanity; loader resolution order unchanged for weight files; predictor returns 3-way probs.

### Integration Tests

- FastAPI `/predict` with injected predictor; optional end-to-end with tiny local artifact (if CI size allows).

### Manual Testing Steps

1. Train or run a short notebook epoch; export artifacts.
2. Point `.env` `JOBSENTRY_PHASE6_FUSED_DIR` at export; restart app; POST sample posts.
3. Compare softmax triples sum to ~1.0 per row.

## Performance Considerations

- **Packed LSTM** reduces wasted compute vs full-length matmul on pads; still **O(B × L)** for DistilBERT.
- **Memory**: same order as current fused forward; no second 400-length embedding tensor.

## Migration Notes

- **Breaking:** Existing **`artifacts/models/phase6_fused/`** 2-class weights **cannot** be `strict=True` loaded after `HybridFusedClassifier` changes.
- **Deployment:** Ship **new** artifact directory alongside config switch; keep old service revision until rollback story is clear.

## References

- Ticket: [`cursor/project/tickets/TICKET-003-sequential-fused-model-update.md`](../tickets/TICKET-003-sequential-fused-model-update.md)
- Research: [`cursor/project/research/2026-04-18-TICKET-003-sequential-fused-model-update.md`](../research/2026-04-18-TICKET-003-sequential-fused-model-update.md)
- Current model: [`app/hybrid_fused_model.py`](../../app/hybrid_fused_model.py)
- Dataset column: `risk_class` in merged splits (see TICKET-002 / row-level merge work).

---

## Sync Note

`humanlayer thoughts sync` was not available in this environment; run it locally if your workflow indexes `cursor/project/plan/`.
