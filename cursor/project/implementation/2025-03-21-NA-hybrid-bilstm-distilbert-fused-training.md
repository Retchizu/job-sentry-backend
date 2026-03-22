# Hybrid Fused Training: BiLSTM + DistilBERT — Implementation Summary

**Date**: 2025-03-21  
**Plan**: `cursor/project/plan/2025-03-21-hybrid-bilstm-distilbert-fused-training.md`  
**Notebook**: `artifacts/ipynb/phase6_hybrid_fused.ipynb`

---

## What Was Built

A new Google Colab notebook (`phase6_hybrid_fused.ipynb`) that trains a **single end-to-end PyTorch model** (`HybridFusedClassifier`) combining DistilBERT and BiLSTM in one forward pass. This replaces the two-stage separate training approach in `phase6_deep_learning.ipynb`.

---

## Architecture Decisions Locked

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| DistilBERT `max_length` | **256** | Matches `phase6_deep_learning.ipynb` §4 baseline |
| BiLSTM `MAX_LEN_BILSTM` | **400** | Matches `phase6_deep_learning.ipynb` §3 baseline |
| Vocab size | **20,000** | Same as notebook §3.1 |
| Embedding dim | **100** | Same as notebook §3.2 |
| LSTM hidden | **64** per direction → **128** after BiLSTM concat | Same as notebook |
| Fusion input dim | **768 + 128 = 896** | DistilBERT mean-pool + LSTM last-step |
| Fusion hidden | **256** | MLP bottleneck |
| DistilBERT LR | **2e-5** | Standard fine-tune rate |
| LSTM/head LR | **1e-3** | Higher LR for randomly initialized weights |
| Freeze strategy | Freeze DistilBERT for first 2 epochs, then unfreeze | Warms up head before joint tuning |
| Loss | `CrossEntropyLoss` with `compute_class_weight("balanced")` | Handles class imbalance (~78% legit / 22% scam) |

---

## Notebook Structure

| Section | Contents |
|---------|----------|
| §1 Runtime check | GPU verify, install `safetensors` |
| §2 Google Drive | Mount + configure `BASE_DIR`, `DATA_DIR`, `FUSED_OUT_DIR` |
| §3 Hyperparameters | All constants in one cell for easy tuning |
| §4 Data loading | `merged_train/val/test.csv` → `combined_text` + `fraudulent` |
| §5 Vocab building | `word2idx` from train corpus, saved as `word_index.json` |
| §6 `HybridFusedClassifier` | Full PyTorch `nn.Module` + forward-pass shape check |
| §7 Dataset & DataLoader | `HybridFusedDataset` + `collate_fn` building both token streams |
| §8 Model/Loss/Optimizer | Param groups, `AdamW`, linear warmup scheduler, `GradScaler` |
| §9 Training loop | Freeze/unfreeze logic, gradient accumulation, per-epoch checkpointing |
| §10 Training curves | Loss / F1 / AUC plots saved to Drive |
| §11 Best ckpt + Test eval | Load best checkpoint, classification report + confusion matrix |
| §12 Export artifacts | `model.safetensors`, `config.json`, tokenizer, `word_index.json`, `fused_meta.json`, `fused_metrics.csv` |
| §13 Roundtrip verification | Reload from safetensors, assert max abs diff < 1e-4 |
| §14 Baseline comparison | Markdown table to fill after training |
| §15 Resume from checkpoint | Utility cell to continue interrupted training |

---

## Google Drive Layout

```
BASE_DIR/                          (default: /content/drive/MyDrive/job-sentry)
├── data/
│   └── processed/
│       ├── merged_train.csv       ← input
│       ├── merged_val.csv         ← input
│       └── merged_test.csv        ← input
└── models/
    └── phase6_fused/              ← all outputs written here
        ├── model.safetensors
        ├── config.json            (DistilBERT config + custom fused keys)
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── word_index.json
        ├── fused_meta.json        (hyperparams + final metrics)
        ├── fused_metrics.csv
        ├── training_curves.png
        ├── confusion_matrices.png
        └── checkpoints/
            ├── epoch_01.pt
            ├── epoch_02.pt
            └── ...
```

---

## Key Implementation Notes

- **`HybridFusedClassifier.forward`** takes three tensors: `input_ids`, `attention_mask` (DistilBERT), and `lstm_ids` (word-level padded LongTensor). Output is `[B, 2]` logits.
- **Mean pooling** over non-pad DistilBERT positions (not just `[CLS]`), producing a 768-dim vector.
- **BiLSTM** uses `h_n[0]` and `h_n[1]` (forward/backward last hidden) concatenated to 128-dim.
- **Gradient accumulation** (`GRAD_ACCUM=2`) doubles effective batch size without extra GPU memory.
- **Mixed precision** (`USE_AMP=True`) enabled when CUDA is available.
- **Vocab** is saved before training so it can be loaded at inference time (`word_index.json`).
- **Existing notebook** (`phase6_deep_learning.ipynb`) is not modified — full reproducibility preserved.

---

## Phases Completed

- [x] Phase 1: Architecture decisions locked (see table above)
- [x] Phase 2: `HybridFusedClassifier` module implemented with forward-pass sanity check
- [x] Phase 3: Dataset, collate_fn, training loop with checkpointing
- [x] Phase 4: Export format (safetensors + config + word_index + fused_meta + metrics CSV)
- [ ] Phase 5: Production inference path (follow-up after training stabilizes)

---

## Follow-up (Phase 5)

Once training is complete and metrics are validated, a new `app/fused_model.py` loader should:
1. Load DistilBERT tokenizer from `phase6_fused/`
2. Load `word_index.json`
3. Reconstruct `HybridFusedClassifier` from `fused_meta.json` hyperparams
4. Load `model.safetensors` with `safetensors.torch.load_file`
5. Implement `predict_proba(texts: List[str]) -> List[float]` matching the `ScamPredictor` protocol in `app/ensemble.py`
