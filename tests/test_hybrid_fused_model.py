"""Tests for HybridFusedClassifier (sequential BERT → BiLSTM → pool)."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn as nn

from app.hybrid_fused_model import HybridFusedClassifier


class _FakeBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"dim": 768})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        b, seq = input_ids.shape
        lhs = torch.ones(b, seq, 768, dtype=torch.float32)
        return type("Out", (), {"last_hidden_state": lhs})()


@patch("app.hybrid_fused_model.DistilBertModel.from_pretrained", return_value=_FakeBert())
def test_forward_logits_shape(mock_from_pretrained) -> None:
    m = HybridFusedClassifier(
        lstm_hidden=8,
        fusion_hidden=32,
        num_labels=3,
        dropout=0.1,
        distilbert_name="distilbert-base-uncased",
    )
    m.eval()
    b, lb = 2, 32
    ids = torch.randint(0, 100, (b, lb))
    mask = torch.ones(b, lb, dtype=torch.long)
    with torch.no_grad():
        out = m(ids, mask)
    assert out.shape == (b, 3)


@patch("app.hybrid_fused_model.DistilBertModel.from_pretrained", return_value=_FakeBert())
def test_mask_changes_pooled_output(mock_from_pretrained) -> None:
    """Shorter effective length (mask) must change the masked mean after BiLSTM."""
    torch.manual_seed(0)
    m = HybridFusedClassifier(
        lstm_hidden=8,
        fusion_hidden=32,
        num_labels=3,
        dropout=0.0,
        distilbert_name="distilbert-base-uncased",
    )
    m.eval()
    b, lb = 1, 16
    ids = torch.randint(0, 100, (b, lb))
    mask_full = torch.ones(b, lb, dtype=torch.long)
    mask_short = torch.tensor([[1] * 8 + [0] * 8], dtype=torch.long)
    with torch.no_grad():
        o_full = m(ids, mask_full)
        o_short = m(ids, mask_short)
    assert not torch.allclose(o_full, o_short)
