"""Batch inference: DistilBERT + LSTM branches aligned with training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast

from app.hybrid_fused_model import HybridFusedClassifier, texts_to_lstm_batch
from app.fused_loader import load_fused_artifacts


class FusedScamPredictor:
    """Runs HybridFusedClassifier with training-aligned preprocessing."""

    def __init__(
        self,
        model: HybridFusedClassifier,
        tokenizer: DistilBertTokenizerFast,
        word2idx: dict[str, int],
        fused_meta: dict[str, Any],
        device: torch.device,
        *,
        max_batch_size: int = 50,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.word2idx = word2idx
        self.fused_meta = fused_meta
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_len_bert = int(fused_meta["max_len_bert"])
        self.max_len_bilstm = int(fused_meta["max_len_bilstm"])

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path | str,
        *,
        checkpoint_override: Optional[Path] = None,
        device: Optional[torch.device] = None,
        max_batch_size: int = 50,
    ) -> FusedScamPredictor:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer, word2idx, fused_meta, _, _ = load_fused_artifacts(
            Path(artifact_dir),
            checkpoint_override=checkpoint_override,
            map_location=dev,
        )
        return cls(
            model,
            tokenizer,
            word2idx,
            fused_meta,
            dev,
            max_batch_size=max_batch_size,
        )

    def predict_proba(self, texts: list[str]) -> list[float]:
        """Return P(scam) = softmax(logits)[1] per row, values in [0, 1]."""
        if not texts:
            return []
        out: list[float] = []
        for start in range(0, len(texts), self.max_batch_size):
            batch = texts[start : start + self.max_batch_size]
            out.extend(self._predict_proba_batch(batch))
        return out

    def _predict_proba_batch(self, texts: list[str]) -> list[float]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len_bert,
            return_tensors="pt",
        )
        lstm_ids = texts_to_lstm_batch(
            texts,
            self.word2idx,
            self.max_len_bilstm,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        lstm_ids = lstm_ids.to(self.device)

        with torch.inference_mode():
            logits = self.model(input_ids, attention_mask, lstm_ids)
            probs = F.softmax(logits, dim=-1)[:, 1]
        return [float(x) for x in probs.cpu().tolist()]


def resolve_device(setting: Optional[str]) -> torch.device:
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("JOBSENTRY_DEVICE=cuda but CUDA is not available")
        return torch.device("cuda")
    if setting == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
