"""Batch inference: DistilBERT → BiLSTM fused model (training-aligned tokenization)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast

from app.hybrid_fused_model import HybridFusedClassifier
from app.fused_loader import load_fused_artifacts
from app.risk_labels import class_from_softmax_triple_with_p_fraud_threshold


@dataclass(frozen=True)
class RiskPrediction:
    """One row of inference output (aligned with ``schemas.PredictResponse`` per post)."""

    predicted_class: int
    predicted_label: str
    legit_probability: float
    warning_probability: float
    fraud_probability: float
    confidence: float


def risk_predictions_from_softmax_triples(
    triples: list[tuple[float, float, float]],
    *,
    p_fraud_threshold: float = 0.4,
) -> list[RiskPrediction]:
    """Map softmax triples to structured rows (same rules as ``POST /predict``)."""
    out: list[RiskPrediction] = []
    for pl, pw, pf in triples:
        cls_i, label_i, conf_i = class_from_softmax_triple_with_p_fraud_threshold(
            pl, pw, pf, p_fraud_threshold=p_fraud_threshold
        )
        out.append(
            RiskPrediction(
                predicted_class=cls_i,
                predicted_label=label_i,
                legit_probability=pl,
                warning_probability=pw,
                fraud_probability=pf,
                confidence=conf_i,
            )
        )
    return out


class FusedScamPredictor:
    """Runs HybridFusedClassifier with DistilBERT tokenizer only (no word-id branch)."""

    def __init__(
        self,
        model: HybridFusedClassifier,
        tokenizer: DistilBertTokenizerFast,
        fused_meta: dict[str, Any],
        device: torch.device,
        *,
        max_batch_size: int = 50,
        p_fraud_threshold: float = 0.4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.fused_meta = fused_meta
        self.device = device
        self.max_batch_size = max_batch_size
        self.p_fraud_threshold = float(p_fraud_threshold)
        self.max_len_bert = int(fused_meta["max_len_bert"])
        self.num_labels = int(fused_meta.get("num_labels", 3))

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path | str,
        *,
        checkpoint_override: Optional[Path] = None,
        device: Optional[torch.device] = None,
        max_batch_size: int = 50,
        p_fraud_threshold: float = 0.4,
    ) -> FusedScamPredictor:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer, fused_meta, _, _ = load_fused_artifacts(
            Path(artifact_dir),
            checkpoint_override=checkpoint_override,
            map_location=dev,
        )
        return cls(
            model,
            tokenizer,
            fused_meta,
            dev,
            max_batch_size=max_batch_size,
            p_fraud_threshold=p_fraud_threshold,
        )

    def predict_risk_distribution(
        self, texts: list[str]
    ) -> list[tuple[float, float, float]]:
        """
        Return (P(legit), P(warning), P(fraud)) per row from softmax over logits.

        Class indices match ``app.risk_labels``: 0=legit, 1=warning, 2=fraud.
        """
        if not texts:
            return []
        out: list[tuple[float, float, float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            batch = texts[start : start + self.max_batch_size]
            out.extend(self._predict_risk_batch(batch))
        return out

    def _predict_risk_batch(self, texts: list[str]) -> list[tuple[float, float, float]]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len_bert,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.inference_mode():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
        rows = probs.cpu().tolist()
        triples: list[tuple[float, float, float]] = []
        for row in rows:
            if len(row) != 3:
                raise RuntimeError(
                    f"Expected num_labels=3 softmax row, got length {len(row)}"
                )
            triples.append((float(row[0]), float(row[1]), float(row[2])))
        return triples

    def predict_full(self, texts: list[str]) -> list[RiskPrediction]:
        """
        Raw texts → tokenizer + forward + softmax → class, label, per-class probs, confidence.

        Uses ``class_from_softmax_triple_with_p_fraud_threshold`` (same as ``POST /predict``).
        """
        triples = self.predict_risk_distribution(texts)
        return risk_predictions_from_softmax_triples(
            triples, p_fraud_threshold=self.p_fraud_threshold
        )


def resolve_device(setting: Optional[str]) -> torch.device:
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("JOBSENTRY_DEVICE=cuda but CUDA is not available")
        return torch.device("cuda")
    if setting == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
