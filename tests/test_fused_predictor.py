"""Tests for FusedScamPredictor.predict_risk_distribution."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from app.fused_predictor import FusedScamPredictor, resolve_device, risk_predictions_from_softmax_triples


class _FlatThreeLogits(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b = input_ids.shape[0]
        # logits [0,0,0] -> softmax ≈ (1/3, 1/3, 1/3)
        return torch.zeros(b, 3, dtype=torch.float32)


def _fake_tokenizer(texts, **kwargs):
    b = len(texts)
    return {
        "input_ids": torch.zeros(b, 3, dtype=torch.long),
        "attention_mask": torch.ones(b, 3, dtype=torch.long),
    }


def test_predict_risk_distribution_length_and_softmax() -> None:
    tok = MagicMock(side_effect=_fake_tokenizer)
    meta = {"max_len_bert": 16, "num_labels": 3}
    p = FusedScamPredictor(
        _FlatThreeLogits(),
        tok,
        meta,
        torch.device("cpu"),
        max_batch_size=10,
    )
    triples = p.predict_risk_distribution(["a", "bb", "ccc"])
    assert len(triples) == 3
    for t in triples:
        assert len(t) == 3
        assert abs(sum(t) - 1.0) < 1e-5
        assert abs(t[0] - 1 / 3) < 1e-4


def test_resolve_device_explicit_cpu() -> None:
    assert str(resolve_device("cpu")) == "cpu"


def test_predict_risk_distribution_empty() -> None:
    tok = MagicMock(side_effect=_fake_tokenizer)
    meta = {"max_len_bert": 8, "num_labels": 3}
    p = FusedScamPredictor(
        _FlatThreeLogits(),
        tok,
        meta,
        torch.device("cpu"),
    )
    assert p.predict_risk_distribution([]) == []


def test_risk_predictions_from_softmax_triples_known_values() -> None:
    rows = risk_predictions_from_softmax_triples([(0.1, 0.2, 0.7)])
    assert len(rows) == 1
    r = rows[0]
    assert r.predicted_class == 2
    assert r.predicted_label == "fraud"
    assert r.legit_probability == 0.1
    assert r.warning_probability == 0.2
    assert r.fraud_probability == 0.7
    assert r.confidence == 0.7


def test_predict_full_matches_triples_and_labels() -> None:
    tok = MagicMock(side_effect=_fake_tokenizer)
    meta = {"max_len_bert": 16, "num_labels": 3}
    p = FusedScamPredictor(
        _FlatThreeLogits(),
        tok,
        meta,
        torch.device("cpu"),
        max_batch_size=10,
    )
    full = p.predict_full(["x", "yy"])
    triples = p.predict_risk_distribution(["x", "yy"])
    assert len(full) == len(triples) == 2
    for row, t in zip(full, triples, strict=True):
        assert row.legit_probability == t[0]
        assert row.warning_probability == t[1]
        assert row.fraud_probability == t[2]
        assert row.confidence == max(t)
    assert full[0].predicted_label == "legit"  # tie at 1/3 → argmax index 0


def test_predict_full_empty() -> None:
    tok = MagicMock(side_effect=_fake_tokenizer)
    meta = {"max_len_bert": 8, "num_labels": 3}
    p = FusedScamPredictor(
        _FlatThreeLogits(),
        tok,
        meta,
        torch.device("cpu"),
    )
    assert p.predict_full([]) == []
