"""Tests for FusedScamPredictor.predict_proba."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from app.fused_predictor import FusedScamPredictor, resolve_device


class _FlatLogits(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lstm_ids: torch.Tensor,
    ) -> torch.Tensor:
        b = input_ids.shape[0]
        return torch.zeros(b, 2, dtype=torch.float32)


def _fake_tokenizer(texts, **kwargs):
    b = len(texts)
    return {
        "input_ids": torch.zeros(b, 3, dtype=torch.long),
        "attention_mask": torch.ones(b, 3, dtype=torch.long),
    }


def test_predict_proba_length_and_range() -> None:
    tok = MagicMock(side_effect=_fake_tokenizer)
    meta = {"max_len_bert": 16, "max_len_bilstm": 16, "threshold": 0.5}
    word2idx = {"<PAD>": 0, "<OOV>": 1, "hello": 2}
    p = FusedScamPredictor(
        _FlatLogits(),
        tok,
        word2idx,
        meta,
        torch.device("cpu"),
        max_batch_size=10,
    )
    probs = p.predict_proba(["a", "bb", "ccc"])
    assert len(probs) == 3
    for x in probs:
        assert 0.0 <= x <= 1.0
        assert abs(x - 0.5) < 1e-5


def test_resolve_device_explicit_cpu() -> None:
    assert str(resolve_device("cpu")) == "cpu"


def test_predict_proba_empty() -> None:
    tok = MagicMock(side_effect=_fake_tokenizer)
    meta = {"max_len_bert": 8, "max_len_bilstm": 8}
    p = FusedScamPredictor(
        _FlatLogits(),
        tok,
        {"<PAD>": 0},
        meta,
        torch.device("cpu"),
    )
    assert p.predict_proba([]) == []
