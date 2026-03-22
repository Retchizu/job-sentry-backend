"""Tests for HybridFusedClassifier and LSTM preprocessing."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn as nn

from app.hybrid_fused_model import (
    HybridFusedClassifier,
    texts_to_lstm_batch,
    tokenize_words,
)


def test_tokenize_words_golden() -> None:
    assert tokenize_words("Hello, world! 123") == ["hello", "world", "123"]


def test_texts_to_lstm_batch_toy_vocab() -> None:
    word2idx = {"<PAD>": 0, "<OOV>": 1, "hello": 2, "world": 3}
    batch = texts_to_lstm_batch(["hello unknown", "world"], word2idx, max_len=6)
    assert batch.shape == (2, 6)
    assert batch[0, 0].item() == 2  # hello
    assert batch[0, 1].item() == 1  # unknown -> OOV
    assert batch[1, 0].item() == 3  # world


class _FakeBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"dim": 768})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        b, seq = input_ids.shape
        lhs = torch.randn(b, seq, 768, dtype=torch.float32)
        return type("Out", (), {"last_hidden_state": lhs})()


@patch("app.hybrid_fused_model.DistilBertModel.from_pretrained", return_value=_FakeBert())
def test_forward_logits_shape(mock_from_pretrained) -> None:
    m = HybridFusedClassifier(
        vocab_size=50,
        embed_dim=16,
        lstm_hidden=8,
        fusion_hidden=32,
        num_labels=2,
        dropout=0.1,
        distilbert_name="distilbert-base-uncased",
    )
    m.eval()
    b, lb, ll = 2, 32, 40
    ids = torch.randint(0, 100, (b, lb))
    mask = torch.ones(b, lb, dtype=torch.long)
    lstm = torch.randint(0, 50, (b, ll))
    with torch.no_grad():
        out = m(ids, mask, lstm)
    assert out.shape == (b, 2)
