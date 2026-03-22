# Hybrid fused training: BiLSTM + DistilBERT as one model — Implementation Plan

## Overview

Train a **single end-to-end model** that combines a **DistilBERT** encoder and a **BiLSTM** branch over word-level sequences, fuses their representations, and optimizes **one classification loss**. This replaces the current two-stage approach in `artifacts/ipynb/phase6_deep_learning.ipynb` (§3 Bi-LSTM and §4 DistilBERT trained separately, then §5 ensemble on probabilities).

**Why PyTorch for the fused model:** The production API already serves DistilBERT via `transformers` + PyTorch (`app/model.py`). Reimplementing the BiLSTM branch in `torch.nn` keeps **one runtime**, one checkpoint format (`safetensors`), and one training loop. A TensorFlow/Keras-only fused model would duplicate DistilBERT serving or require TF inference for the transformer.

## Current State Analysis

- **`artifacts/ipynb/phase6_deep_learning.ipynb`**  
  - **§3 Bi-LSTM**: Keras `Sequential` — `Embedding` (vocab × 100, maxlen 400) → `Bidirectional(LSTM(64))` → `Dense(32, relu)` → `Dense(1, sigmoid)` (see model summary in notebook output cells). Tokenization: custom `tokenize_for_sequences` + `Tokenizer` / `pad_sequences` on `combined_text` from `merged_train.csv` / `merged_val.csv` / `merged_test.csv`.  
  - **§4 DistilBERT**: Hugging Face `AutoModelForSequenceClassification` + `Trainer`, `distilbert-base-uncased`, `MAX_LEN_DISTILBERT = 256` in notebook.  
  - **§5 Hybrid ensemble**: Soft stacking / recall-weighted combinations of **already trained** Bi-LSTM and DistilBERT (plus optional Phase 6 sklearn models) — **no shared weights**.

- **Production backend** (`app/main.py`, `app/ensemble.py`, `app/model.py`, `app/bilstm.py`)  
  - Loads **separate** `DistilBertPredictor`, optional `BiLSTMPredictor` (Keras), optional `Phase6MergedPredictor`; combines with **soft voting** (`combine_soft_voting`).  
  - `settings.max_sequence_length` defaults to **512** for DistilBERT inference (`app/config.py`); the Phase 6 notebook used **256** for DistilBERT training — any fused training run should **fix max length in config and training** to match.

### Key Discoveries

- `artifacts/ipynb/phase6_deep_learning.ipynb` — Separate frameworks (TensorFlow + PyTorch) for the two towers; ensemble only at prediction time (§5).  
- `app/model.py:31-37` — `ScamDetectionModel.predict_batch` tokenizes with `max_length=settings.max_sequence_length`.  
- `app/bilstm.py:12-37` — BiLSTM inference uses `split()` tokenization + `pad_sequences` with `MAX_SEQUENCE_LENGTH = 200`, while the notebook used **400** — fused training artifacts must document **max lengths** for both branches explicitly.

## Desired End State

1. **Model definition**: One `nn.Module` (e.g. `HybridFusedClassifier`) with:
   - **DistilBERT** backbone (`DistilBertModel` or `DistilBertForSequenceClassification` with head removed / ignored) producing a fixed-size vector (e.g. pooled output or `[CLS]`-based projection).  
   - **BiLSTM branch**: embedding table indexed by the same **word-level vocabulary** strategy as the notebook (build `word_index`, `pad_sequences` to `max_len_bilstm`, e.g. 400).  
   - **Fusion**: concatenate (or gated / attention fusion — start with **concat + MLP** or **concat + Linear + dropout**) → binary logits (2-class softmax or single logit + sigmoid).  
2. **Training**: Script or notebook that reads **`merged_train.csv` / `merged_val.csv`** (same as Phase 6 DL notebook), builds batches with **both** DistilBERT encodings and BiLSTM integer sequences, trains with **cross-entropy** (or BCE), optional **class weights**, **per-parameter-group learning rates** (lower LR for DistilBERT, higher for LSTM/embedding/head).  
3. **Artifacts**: Exported checkpoint(s) compatible with serving: at minimum `config.json`, `model.safetensors`, tokenizer for DistilBERT, and **serialized word_index** (JSON) + hyperparameters (max lengths) for the LSTM branch.  
4. **Evaluation**: Validation and test metrics comparable to §3.4 / §4.3 / §6 in the notebook; ablation optional (DistilBERT-only head vs LSTM-only vs fused).  
5. **Backend (optional follow-up)**: Either a new loader `load_fused_model()` returning one `ScamPredictor`, or document that fused model replaces DistilBERT-only path when configured — **only after** training stabilizes.

### Verification

- Training loss decreases; val accuracy / F1 / ROC-AUC meet or exceed a documented baseline (separate models or previous ensemble).  
- Single `forward` pass produces predictions; no separate ensemble step.  
- Inference test: load checkpoint and run a small batch through the fused module.

## What We're NOT Doing

- Changing the **existing** `phase6_deep_learning.ipynb` cells in place without a copy/branch (prefer a **new** notebook `phase6_fused_deep_learning.ipynb` or a **Python script** under `scripts/` to avoid breaking reproducibility of the old pipeline).  
- Joint training with **Phase 6 traditional ML** (TF-IDF + sklearn) inside the same neural graph — that remains a **separate** stacking/ensemble concern.  
- Committing multi-gigabyte checkpoints to git (artifacts stay local or in object storage).  
- Guaranteed improvement over soft-voting ensemble: fused training is an experiment; **fallback** remains separate models + `EnsemblePredictor`.

## Implementation Approach

1. **Unify in PyTorch**: Implement BiLSTM tower with `nn.Embedding`, `nn.LSTM(bidirectional=True)`, pooling, then fuse with DistilBERT hidden states.  
2. **Dataset**: For each row, precompute or on-the-fly compute `input_ids` / `attention_mask` (DistilBERT) and `lstm_ids` (LongTensor, padded). Use a **collate_fn** that pads LSTM sequences to batch max or fixed max.  
3. **Optimization**: Start with **frozen DistilBERT** for N epochs, then **unfreeze** with small LR; or use **differential learning rates** from epoch 0.  
4. **Alignment with production**: After training, set `JOBSENTRY_MAX_SEQUENCE_LENGTH` to match DistilBERT **training** `max_length` if you serve the fused DistilBERT tokenizer the same way.

---

## Phase 1: Specification and scaffolding

### Overview

Lock architecture choices, tensor shapes, and file layout so implementation does not thrash.

### Changes Required

#### 1. Design doc (short) in plan or `cursor/project/notes/`

**Content**:

- Branch A output dim (DistilBERT hidden size 768).  
- Branch B: `vocab_size`, `embed_dim` (100 to match notebook), `hidden` (64 per direction → 128 after bidirectional concat), `max_len` (400).  
- Fusion: `concat` → `Linear` / `MLP` → `num_labels`.  
- Loss: `CrossEntropyLoss` with class weights from `y_train` (same idea as notebook `compute_class_weight`).  
- **Decision**: Match notebook DistilBERT `max_length` **256** for training **or** explicitly move to **512** and re-baseline — pick one and document.

#### 2. Optional package layout

- `training/` or `scripts/` — e.g. `scripts/train_hybrid_fused.py`  
- `training/models/hybrid_fused.py` — `HybridFusedClassifier` (if you want importable tests)

### Success Criteria

#### Automated Verification

- [ ] New module(s) import without error: `python -c "from training.models.hybrid_fused import HybridFusedClassifier"` (adjust path after creation).  
- [ ] `python -m pytest tests/ -q` still passes (no regressions if only new files added).

#### Manual Verification

- [ ] Architecture diagram or bullet list reviewed: fusion point, dims, and two tokenizers agreed.

**Implementation Note**: Pause after Phase 1 for confirmation on **max_length** (256 vs 512) and whether to **freeze** DistilBERT initially.

---

## Phase 2: `HybridFusedClassifier` module

### Overview

Implement the PyTorch module and a small forward pass test on random tensors.

### Changes Required

#### 1. `HybridFusedClassifier`

**Responsibilities**:

- Load `DistilBertModel.from_pretrained("distilbert-base-uncased")`.  
- Pool sequence output (e.g. mean of last hidden layer over non-pad positions, or use DistilBERT’s default pooler if using `DistilBertForSequenceClassification` features — prefer explicit pooling for clarity).  
- LSTM: `nn.Embedding(vocab_size, embed_dim, padding_idx=0)` → `nn.LSTM(embed_dim, 64, batch_first=True, bidirectional=True)` → take last step or max-pool over time.  
- `torch.cat([bert_vec, lstm_vec], dim=-1)` → `nn.Sequential` with `Dropout`, `Linear`, `ReLU`, `Linear` to logits.

#### 2. Vocabulary

- Reuse the same **Tokenizer**/word index building as notebook §3.1 (`fit_on_texts` on train `combined_text`) so indices align with merged data.

### Success Criteria

#### Automated Verification

- [ ] Unit test: instantiate model with small `vocab_size`, run forward with dummy `input_ids`, `attention_mask`, `lstm_token_ids` → output shape `[batch, 2]` or `[batch, 1]` consistent with loss choice.  
- [ ] `pytest` passes for new test file e.g. `tests/test_hybrid_fused_model.py`.

#### Manual Verification

- [ ] Parameter count order-of-magnitude sensible vs separate DistilBERT + small LSTM.

---

## Phase 3: Dataset, collate, and training loop

### Overview

Load CSVs from `artifacts/data/processed/` (or configured path), match columns `combined_text` and `fraudulent`, build `DataLoader`.

### Changes Required

#### 1. `HybridFusedDataset`

- Reads paths from env/CLI: `merged_train.csv`, `merged_val.csv`.  
- Returns dict with raw text or pre-tokenized fields.

#### 2. Collate

- DistilBERT: `DistilBertTokenizerFast` batch encode with `padding=True`, `truncation=True`, `max_length=FUSED_MAX_LEN`.  
- LSTM: map text to indices using **fixed** `word_index`, `pad_sequence` to `MAX_LEN_BILSTM` (400).

#### 3. Training script

- `AdamW` with parameter groups:  
  - `distilbert.*` → lr ~2e-5 to 5e-5  
  - `lstm`, `embedding`, `classifier` → lr ~1e-3 (tune)  
- Mixed precision optional (`torch.cuda.amp`) if GPU.  
- Checkpoint every epoch: `save_pretrained` style folder with **custom** `head` weights + `lstm_state` in same `state_dict` or separate files (single `state_dict` preferred for one file).

### Success Criteria

#### Automated Verification

- [ ] Dry-run 1–2 steps on CPU with tiny subset: script exits 0.  
- [ ] `pytest` still passes globally.

#### Manual Verification

- [ ] Training loss decreases over first few hundred steps on real GPU.  
- [ ] No OOM at chosen batch size (reduce batch size or gradient accumulation).

---

## Phase 4: Export, evaluation, and comparison

### Overview

Save artifacts for inference; measure val/test metrics; compare to separate BiLSTM + DistilBERT baselines from the notebook.

### Changes Required

#### 1. Export format

- `model.safetensors` (full fused `state_dict`).  
- `config.json`: include custom keys for **fusion** (e.g. `architectures`, `max_len_bilstm`, `vocab_size`) — `transformers` may ignore unknown keys; alternatively a sidecar `fused_meta.json` with `word_index` path and hyperparams.  
- `word_index.json` next to checkpoint (same pattern as `app/bilstm.py` expectations).

#### 2. Evaluation notebook or script

- Confusion matrix, precision/recall/F1, ROC-AUC on val and test.  
- Optional: export metrics CSV next to `artifacts/models/phase6_merged/`.

### Success Criteria

#### Automated Verification

- [ ] Load saved `state_dict` into `HybridFusedClassifier` and match forward output before/after save (numerical tolerance).

#### Manual Verification

- [ ] Test metrics documented in a short table vs ensemble / single models.

---

## Phase 5 (optional): Production inference path

### Overview

Serve the fused model from the Job Sentry API as **one** predictor (replacing or alongside multi-model ensemble).

### Changes Required

#### 1. New loader e.g. `app/fused_model.py`

- Load DistilBERT tokenizer from checkpoint dir.  
- Load `word_index` for LSTM branch.  
- Rebuild `HybridFusedClassifier`, load weights, `model.eval()`.  
- Implement `predict_proba` matching `ScamPredictor` protocol (`app/ensemble.py`).

#### 2. `app/config.py` / `app/main.py`

- New optional setting: `fused_model_path: Optional[Path] = None`. If set, use fused predictor **instead of** or **in addition to** existing stack — **product decision**: simplest is **fused replaces DistilBERT-only** when path set and hybrid BiLSTM path unset.

### Success Criteria

#### Automated Verification

- [ ] Tests: mock or tiny fused checkpoint; `predict_proba` returns list of floats in [0,1].  
- [ ] `pytest tests/ -v` passes.

#### Manual Verification

- [ ] `POST /predict` latency acceptable vs current DistilBERT-only (fused is heavier than DistilBERT alone).

---

## Testing Strategy

### Unit Tests

- Forward shape test for `HybridFusedClassifier`.  
- Save/load roundtrip of `state_dict`.  
- Token alignment: one known string → fixed LSTM id sequence (golden test with small vocab).

### Integration Tests

- Optional: smoke test training script with `--max-steps 5` on CPU CI.

### Manual Testing Steps

1. Run full training on GPU; confirm val F1.  
2. Compare with `phase6_deep_learning.ipynb` baselines (same splits).  
3. If Phase 5 done: curl `/predict` with sample job JSON.

## Performance Considerations

- **Memory**: Two towers in one forward pass ≈ DistilBERT + embedding/LSTM overhead; use smaller batch size or gradient checkpointing if needed.  
- **Latency**: Fused inference runs both branches sequentially in one graph — likely **slower** than DistilBERT-only but **faster** than two separate HTTP/model loads if optimized as one GPU kernel batch.  
- **Sequence length**: Longer DistilBERT `max_length` increases cost quadratically in attention — align with product SLA.

## Migration Notes

- Existing **ensemble** deployment unchanged until Phase 5 is enabled.  
- Fused checkpoint is **not** interchangeable with plain `DistilBertForSequenceClassification` weights — `app/model.py` must not load fused weights into the current `ScamDetectionModel` without the new architecture class.

## References

- `artifacts/ipynb/phase6_deep_learning.ipynb` — §3 Bi-LSTM (Keras), §4 DistilBERT (`Trainer`), §5 ensemble, data from `merged_*.csv`.  
- `app/main.py:24-48` — ensemble wiring.  
- `app/model.py:17-53` — DistilBERT inference contract.  
- `app/bilstm.py:15-46` — legacy BiLSTM predictor (Keras + word_index).  
- `cursor/project/plan/2025-03-09-checkpoint-5412-and-hybrid-support.md` — separate-model hybrid serving.
