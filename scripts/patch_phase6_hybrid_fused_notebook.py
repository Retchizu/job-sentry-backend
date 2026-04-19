#!/usr/bin/env python3
"""One-off patch: align phase6_hybrid_fused.ipynb with sequential 3-class HybridFusedClassifier."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_path = root / "artifacts" / "ipynb" / "phase6_hybrid_fused.ipynb"
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    def set_md(idx: int, text: str) -> None:
        cells[idx]["source"] = [text + "\n"] if not text.endswith("\n") else [text]
        # normalize to line list like Jupyter
        lines = text.rstrip("\n").split("\n")
        cells[idx]["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])

    def set_code(idx: int, text: str) -> None:
        lines = text.rstrip("\n").split("\n")
        cells[idx]["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])

    set_md(
        0,
        """# Phase 6 — Sequential fused model: DistilBERT → BiLSTM → 3-class head

This notebook trains **`HybridFusedClassifier`** (imported from `app/hybrid_fused_model` in-repo):
- **DistilBERT** → `last_hidden_state` `[B, L, 768]`
- **BiLSTM** on the full token sequence (packed padded) → sequence `[B, L, 128]` (H=64 bi-directional)
- **Masked mean pool** over LSTM outputs → `[B, 128]` → MLP → **3 logits** (`risk_class`: 0=legit, 1=warning, 2=fraud)

**Hyperparameters (defaults):** DistilBERT `max_length = 256`; BERT LR `2e-5`, head/LSTM LR `1e-3`; BERT frozen for `FREEZE_BERT_EPOCHS`; `CrossEntropyLoss` with balanced class weights on `risk_class`.

**Data:** `merged_*.csv` with columns `combined_text` and **`risk_class`** (0/1/2).

**Output** (`FUSED_OUT_DIR`): `model.safetensors`, tokenizer files, `fused_meta.json` (`num_labels: 3`, `arch: sequential_distilbert_bilstm_v1`), metrics.""",
    )

    set_code(
        7,
        """# ── Architecture ──────────────────────────────────────────────────────────────
DISTILBERT_MODEL   = "distilbert-base-uncased"
MAX_LEN_BERT       = 256     # DistilBERT max token length
LSTM_HIDDEN        = 64      # per direction → 128 after BiLSTM concat
FUSION_HIDDEN      = 256     # MLP hidden after pool
DROPOUT            = 0.3
NUM_LABELS         = 3       # legit / warning / fraud

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE         = 16      # Reduce to 8 if OOM
GRAD_ACCUM         = 2       # Effective batch = BATCH_SIZE × GRAD_ACCUM
EPOCHS             = 10
FREEZE_BERT_EPOCHS = 2       # Epochs to keep DistilBERT frozen
LR_BERT            = 2e-5
LR_HEAD            = 1e-3
WARMUP_RATIO       = 0.1
MAX_GRAD_NORM      = 1.0
USE_AMP            = True and torch.cuda.is_available()  # Mixed precision

# ── Legacy export compatibility (optional threshold in fused_meta; API uses softmax)
THRESHOLD          = 0.5

print("Hyperparameters set.")
print(f"  DistilBERT max_length : {MAX_LEN_BERT}")
print(f"  num_labels            : {NUM_LABELS}")
print(f"  Freeze BERT for       : {FREEZE_BERT_EPOCHS} epoch(s)")
print(f"  Mixed precision (AMP) : {USE_AMP}")""",
    )

    set_code(
        9,
        """train_df = pd.read_csv(os.path.join(DATA_DIR, "merged_train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "merged_val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "merged_test.csv"))

for df in [train_df, val_df, test_df]:
    df["combined_text"] = df["combined_text"].fillna("").astype(str)
    df["risk_class"]    = df["risk_class"].astype(int)

train_texts = train_df["combined_text"].tolist()
val_texts   = val_df["combined_text"].tolist()
test_texts  = test_df["combined_text"].tolist()

y_train = train_df["risk_class"].values
y_val   = val_df["risk_class"].values
y_test  = test_df["risk_class"].values

print(f"Train: {train_df.shape}  Val: {val_df.shape}  Test: {test_df.shape}")
print("risk_class distribution (train):", pd.Series(y_train).value_counts().sort_index().to_dict())""",
    )

    set_md(
        10,
        """## 5 · (Removed) Word vocabulary

The production **sequential** stack feeds **DistilBERT token embeddings** into BiLSTM. There is no separate `word_index.json` or parallel word-ID tensor.""",
    )

    set_code(
        11,
        """# No word vocabulary — inference uses DistilBERT tokenizer only (see app.fused_predictor).
print("OK: sequential fused path has no word_index.json.")""",
    )

    set_md(
        12,
        """## 6 · HybridFusedClassifier (import from `app`)

**Tensor shape cheat sheet** (defaults `L=256`, `H=64` BiLSTM hidden per direction):

| Stage | Shape |
| --- | --- |
| BERT `last_hidden_state` | `[B, L, 768]` |
| Packed → BiLSTM → `pad_packed_sequence` | `[B, L, 2H]` = `[B, L, 128]` |
| Masked mean pool (attention mask) | `[B, 2H]` = `[B, 128]` |
| Logits | `[B, 3]` |

Run from the **repository root** so `app.hybrid_fused_model` resolves.""",
    )

    set_code(
        13,
        """import sys
from pathlib import Path

_repo = Path.cwd().resolve()
for candidate in (_repo, _repo.parent):
    if (candidate / "app" / "hybrid_fused_model.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError(
        "Could not find app/hybrid_fused_model.py — run the notebook with cwd = job-sentry-backend repo root."
    )

from app.hybrid_fused_model import HybridFusedClassifier

_m = HybridFusedClassifier(
    lstm_hidden=LSTM_HIDDEN,
    fusion_hidden=FUSION_HIDDEN,
    num_labels=NUM_LABELS,
    dropout=DROPOUT,
    distilbert_name=DISTILBERT_MODEL,
)
_m.eval()
with torch.no_grad():
    _ids = torch.randint(0, 1000, (2, MAX_LEN_BERT))
    _mask = torch.ones(2, MAX_LEN_BERT, dtype=torch.long)
    _out = _m(_ids, _mask)
print(f"Imported HybridFusedClassifier — sanity output shape: {_out.shape} (expect [2, {NUM_LABELS}])")
del _m, _ids, _mask, _out""",
    )

    set_code(
        15,
        r'''class HybridFusedDataset(Dataset):
    """Stores raw texts + integer risk_class labels."""

    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": int(self.labels[idx])}


def build_collate_fn(bert_tokenizer, max_len_bert):
    """Collate: DistilBERT batch only (no word-id tensor)."""

    def collate_fn(batch):
        texts  = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        bert_enc = bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len_bert,
            return_tensors="pt",
        )

        return {
            "input_ids":      bert_enc["input_ids"],
            "attention_mask": bert_enc["attention_mask"],
            "labels":         torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


bert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL)

collate_fn = build_collate_fn(bert_tokenizer, max_len_bert=MAX_LEN_BERT)

train_dataset = HybridFusedDataset(train_texts, y_train)
val_dataset   = HybridFusedDataset(val_texts,   y_val)
test_dataset  = HybridFusedDataset(test_texts,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2, shuffle=False,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}  |  Test batches: {len(test_loader)}")''',
    )

    set_code(
        17,
        """# ── Class weights (3-class) ─────────────────────────────────────────────────
class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"Class weights: {dict(enumerate(class_weights.round(4)))}")

# ── Model (same class as app.hybrid_fused_model) ─────────────────────────────
model = HybridFusedClassifier(
    lstm_hidden=LSTM_HIDDEN,
    fusion_hidden=FUSION_HIDDEN,
    num_labels=NUM_LABELS,
    dropout=DROPOUT,
    distilbert_name=DISTILBERT_MODEL,
).to(DEVICE)

model.freeze_bert()
print(f"DistilBERT frozen for first {FREEZE_BERT_EPOCHS} epoch(s).")

criterion = nn.CrossEntropyLoss(weight=class_weights_t)

bert_params  = list(model.bert.parameters())
other_params = [p for p in model.parameters() if id(p) not in {id(q) for q in bert_params}]

optimizer = AdamW([
    {"params": bert_params,  "lr": LR_BERT},
    {"params": other_params, "lr": LR_HEAD},
], weight_decay=1e-2)

total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

scaler = GradScaler(enabled=USE_AMP)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params : {total_params:,}")
print(f"Trainable    : {trainable_params:,}  (BERT frozen)")""",
    )

    set_code(
        19,
        r'''def evaluate(model, loader, criterion, device):
    """Returns loss, acc, macro-F1, weighted-F1, macro-OVR AUC, probs, preds."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast(enabled=USE_AMP):
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)

            total_loss  += loss.item() * labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    probs      = torch.softmax(all_logits, dim=-1).numpy()
    preds      = all_logits.argmax(dim=-1).numpy()

    n     = len(all_labels)
    loss  = total_loss / n
    acc   = accuracy_score(all_labels, preds)
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return loss, acc, f1_macro, f1_weighted, auc, probs, preds


# ── Resume from latest epoch_*.pt (so we don't restart at 1 after a crash / re-run) ──
import glob as _glob
import re as _re

RESUME_FROM_LATEST_CKPT = True  # set False to ignore checkpoints and train from epoch 1


def _state_dict_compatible(ckpt_state, model) -> bool:
    """True if a strict load_state_dict would succeed (same keys and tensor shapes)."""
    cur = model.state_dict()
    if set(ckpt_state.keys()) != set(cur.keys()):
        return False
    for k in cur:
        if ckpt_state[k].shape != cur[k].shape:
            return False
    return True


def _latest_checkpoint_path():
    paths = _glob.glob(os.path.join(CKPT_DIR, "epoch_*.pt"))
    best_p, best_e = None, -1
    for p in paths:
        m = _re.search(r"epoch_(\d+)\.pt$", os.path.basename(p))
        if m:
            e = int(m.group(1))
            if e > best_e:
                best_e, best_p = e, p
    return best_p, best_e


_resume_path, _ = _latest_checkpoint_path()
ckpt_data = None
if RESUME_FROM_LATEST_CKPT and _resume_path is not None:
    _raw = torch.load(_resume_path, map_location=DEVICE, weights_only=False)
    if _state_dict_compatible(_raw["model_state"], model):
        ckpt_data = _raw
    else:
        print(
            f"Ignoring incompatible checkpoint: {_resume_path}\n"
            "  (different architecture than current HybridFusedClassifier — e.g. old embedding+LSTM+2-class).\n"
            "  Delete stale epoch_*.pt in CKPT_DIR or set RESUME_FROM_LATEST_CKPT = False to train from scratch."
        )

if ckpt_data is not None:
    model.load_state_dict(ckpt_data["model_state"])
    optimizer.load_state_dict(ckpt_data["optimizer_state"])
    if "scheduler_state" in ckpt_data:
        scheduler.load_state_dict(ckpt_data["scheduler_state"])
    else:
        print("Note: checkpoint has no scheduler_state — LR schedule may not match pre-interruption training.")
    if "scaler_state" in ckpt_data and ckpt_data["scaler_state"] is not None:
        scaler.load_state_dict(ckpt_data["scaler_state"])
    history = ckpt_data.get("history", [])
    start_epoch = int(ckpt_data["epoch"]) + 1
    if history:
        _best = max(history, key=lambda r: r["val_f1"])
        best_val_f1 = _best["val_f1"]
        best_ckpt_path = os.path.join(CKPT_DIR, f"epoch_{_best['epoch']:02d}.pt")
    else:
        best_val_f1 = float(ckpt_data.get("val_f1", -1.0))
        best_ckpt_path = _resume_path
    print(
        f"Resumed from {_resume_path}  (completed epoch {ckpt_data['epoch']}, "
        f"next training epoch → {start_epoch})"
    )
else:
    history = []
    start_epoch = 1
    best_val_f1 = -1.0
    best_ckpt_path = None
    if RESUME_FROM_LATEST_CKPT and _resume_path is None:
        print("No epoch_*.pt checkpoints in CKPT_DIR — starting from epoch 1.")

if start_epoch > EPOCHS:
    print(f"Nothing to run: last checkpoint epoch ≥ EPOCHS (start_epoch={start_epoch}, EPOCHS={EPOCHS}).")

for epoch in range(start_epoch, EPOCHS + 1):
    if epoch == FREEZE_BERT_EPOCHS + 1:
        model.unfreeze_bert()
        bert_params  = list(model.bert.parameters())
        other_params = [p for p in model.parameters()
                        if id(p) not in {id(q) for q in bert_params}]
        optimizer = AdamW([
            {"params": bert_params,  "lr": LR_BERT},
            {"params": other_params, "lr": LR_HEAD},
        ], weight_decay=1e-2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_loader) // GRAD_ACCUM) * (EPOCHS - FREEZE_BERT_EPOCHS),
        )
        print(f"  [epoch {epoch}] DistilBERT unfrozen.")

    model.train()
    epoch_loss    = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(train_loader, start=1):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        with autocast(enabled=USE_AMP):
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels) / GRAD_ACCUM

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * GRAD_ACCUM * labels.size(0)

        if step % GRAD_ACCUM == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if step % 200 == 0:
            print(f"  [epoch {epoch}  step {step}/{len(train_loader)}] "
                  f"loss={epoch_loss / (step * BATCH_SIZE):.4f}")

    train_loss = epoch_loss / len(train_dataset)

    val_loss, val_acc, val_f1, val_f1_weighted, val_auc, val_probs, val_preds = evaluate(
        model, val_loader, criterion, DEVICE
    )

    elapsed = time.time() - t0
    row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
               val_acc=val_acc, val_f1=val_f1, val_f1_weighted=val_f1_weighted, val_auc=val_auc)
    history.append(row)

    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  "
          f"val_acc={val_acc:.4f}  "
          f"val_F1_macro={val_f1:.4f}  "
          f"val_F1_wt={val_f1_weighted:.4f}  "
          f"val_AUC={val_auc:.4f}  "
          f"[{elapsed:.0f}s]")

    ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch:02d}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "val_f1": val_f1,
        "history": history,
    }, ckpt_path)

    if val_f1 > best_val_f1:
        best_val_f1   = val_f1
        best_ckpt_path = ckpt_path
        print(f"  ★ New best val F1 (macro): {best_val_f1:.4f}  → {ckpt_path}")

print() 
if start_epoch <= EPOCHS:
    print(f"Training complete. Best val F1 (macro): {best_val_f1:.4f}")
else:
    print(f"No new epochs run. Best val F1 (macro, from checkpoint history): {best_val_f1:.4f}")
''',
    )

    set_code(
        23,
        """# Reload best checkpoint weights
best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(best_ckpt["model_state"])
print(f"Loaded best checkpoint: {best_ckpt_path} (epoch {best_ckpt['epoch']}, "
      f"val_F1_macro={best_ckpt['val_f1']:.4f})")

val_loss, val_acc, val_f1, val_f1_weighted, val_auc, val_probs, val_preds = evaluate(
    model, val_loader, criterion, DEVICE,
)
print("
=== Validation ===")
print(f"acc={val_acc:.4f}  F1_macro={val_f1:.4f}  F1_weighted={val_f1_weighted:.4f}")
print(classification_report(
    y_val, val_preds, labels=[0, 1, 2], target_names=["legit", "warning", "fraud"], zero_division=0,
))
print(f"ROC-AUC (OVR) : {val_auc:.4f}")

test_loss, test_acc, test_f1, test_f1_weighted, test_auc, test_probs, test_preds = evaluate(
    model, test_loader, criterion, DEVICE,
)
print("
=== Test ===")
print(f"acc={test_acc:.4f}  F1_macro={test_f1:.4f}  F1_weighted={test_f1_weighted:.4f}")
print(classification_report(
    y_test, test_preds, labels=[0, 1, 2], target_names=["legit", "warning", "fraud"], zero_division=0,
))
print(f"ROC-AUC (OVR) : {test_auc:.4f}")

LABEL_NAMES = ["legit", "warning", "fraud"]


def _per_class_metrics(y_true, preds):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        LABEL_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
        for i in range(3)
    }


fused_per_class = {
    "val": _per_class_metrics(y_val, val_preds),
    "test": _per_class_metrics(y_test, test_preds),
}
_per_class_path = os.path.join(FUSED_OUT_DIR, "fused_per_class_metrics.json")
with open(_per_class_path, "w") as f:
    json.dump(fused_per_class, f, indent=2)
print(f"Saved fused_per_class_metrics.json → {_per_class_path}")
""",
    )

    set_code(
        24,
        """# Confusion matrices (3x3)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
labels_cm = ["legit", "warning", "fraud"]
for ax, preds, y_true, title in [
    (axes[0], val_preds,  y_val,  "Val"),
    (axes[1], test_preds, y_test, "Test"),
]:
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=labels_cm, yticklabels=labels_cm)
    ax.set_title(f"{title} — 3-class (legit / warning / fraud)")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(FUSED_OUT_DIR, "confusion_matrices.png"), dpi=120)
plt.show()
""",
    )

    set_code(
        26,
        """# ── 12.1  model.safetensors ───────────────────────────────────────────────────
safetensors_path = os.path.join(FUSED_OUT_DIR, "model.safetensors")
save_file(model.state_dict(), safetensors_path)
print(f"Saved model.safetensors → {safetensors_path}")

# ── 12.2  DistilBERT config + tokenizer ──────────────────────────────────────
config = model.bert.config
config_dict = config.to_dict()
config_dict.update({
    "architectures":      ["HybridFusedClassifier"],
    "fused_arch":         "sequential_distilbert_bilstm_v1",
    "max_len_bert":       MAX_LEN_BERT,
    "lstm_hidden":        LSTM_HIDDEN,
    "fusion_hidden":      FUSION_HIDDEN,
    "num_labels":         NUM_LABELS,
    "dropout":            DROPOUT,
})
config_path = os.path.join(FUSED_OUT_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config_dict, f, indent=2)
print(f"Saved config.json → {config_path}")

bert_tokenizer.save_pretrained(FUSED_OUT_DIR)
print(f"Saved tokenizer → {FUSED_OUT_DIR}")

# ── 12.3  fused_meta.json (matches app.fused_loader) ─────────────────────────
# best_val_f1, test_f1, test_f1_macro: macro F1 (same as val_f1 / checkpoint metric).
meta = {
    "arch":               "sequential_distilbert_bilstm_v1",
    "max_len_bert":       MAX_LEN_BERT,
    "lstm_hidden":        LSTM_HIDDEN,
    "fusion_hidden":      FUSION_HIDDEN,
    "num_labels":         NUM_LABELS,
    "risk_class_labels":  ["legit", "warning", "fraud"],
    "dropout":            DROPOUT,
    "distilbert_model":   DISTILBERT_MODEL,
    "threshold":          THRESHOLD,
    "best_val_f1":        best_val_f1,
    "test_f1":            test_f1,
    "test_f1_macro":      test_f1,
    "test_f1_weighted":   test_f1_weighted,
    "test_auc":           test_auc,
}
meta_path = os.path.join(FUSED_OUT_DIR, "fused_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved fused_meta.json → {meta_path}")

metrics_df = pd.DataFrame([{
    "split":        "val",
    "loss":         val_loss,
    "acc":          val_acc,
    "f1":           val_f1,
    "f1_macro":     val_f1,
    "f1_weighted":  val_f1_weighted,
    "auc":          val_auc,
}, {
    "split":        "test",
    "loss":         test_loss,
    "acc":          test_acc,
    "f1":           test_f1,
    "f1_macro":     test_f1,
    "f1_weighted":  test_f1_weighted,
    "auc":          test_auc,
}])
metrics_path = os.path.join(FUSED_OUT_DIR, "fused_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved fused_metrics.csv → {metrics_path}")

print("\n── All artifacts saved ──")
for fname in os.listdir(FUSED_OUT_DIR):
    fpath = os.path.join(FUSED_OUT_DIR, fname)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {fname:<40} {size_mb:>8.2f} MB")
""",
    )

    set_code(
        28,
        """# Verify that loading the saved safetensors produces identical logits
from safetensors.torch import load_file as _load_file

model.eval()
sample_batch = next(iter(val_loader))
with torch.no_grad():
    before_logits = model(
        sample_batch["input_ids"].to(DEVICE),
        sample_batch["attention_mask"].to(DEVICE),
    ).cpu()

model_reload = HybridFusedClassifier(
    lstm_hidden=LSTM_HIDDEN,
    fusion_hidden=FUSION_HIDDEN,
    num_labels=NUM_LABELS,
    dropout=DROPOUT,
    distilbert_name=DISTILBERT_MODEL,
).to(DEVICE)
state = _load_file(safetensors_path, device=str(DEVICE))
model_reload.load_state_dict(state, strict=True)
model_reload.eval()

with torch.no_grad():
    after_logits = model_reload(
        sample_batch["input_ids"].to(DEVICE),
        sample_batch["attention_mask"].to(DEVICE),
    ).cpu()

max_diff = (before_logits - after_logits).abs().max().item()
print(f"Max abs diff before/after reload: {max_diff:.2e}")
assert max_diff < 1e-4, f"Roundtrip mismatch: {max_diff}"
print("✓ Save/load roundtrip verified.")""",
    )

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    print(f"Wrote {nb_path}")


if __name__ == "__main__":
    main()
