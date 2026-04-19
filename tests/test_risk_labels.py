"""Unit tests for 3-class risk mapping from binary P(scam)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.risk_labels import (
    CLASS_FRAUD,
    CLASS_LEGIT,
    CLASS_WARNING,
    class_from_softmax_triple,
    class_from_softmax_triple_with_p_fraud_threshold,
    map_binary_to_risk,
)


def test_class_from_softmax_triple_argmax() -> None:
    cls, label, conf = class_from_softmax_triple(0.1, 0.2, 0.7)
    assert cls == CLASS_FRAUD
    assert label == "fraud"
    assert conf == pytest.approx(0.7)


def test_class_from_softmax_triple_with_p_fraud_threshold_fraud_and_legit() -> None:
    cls, label, conf = class_from_softmax_triple_with_p_fraud_threshold(
        0.5, 0.15, 0.35, p_fraud_threshold=0.4
    )
    assert cls == CLASS_LEGIT
    assert label == "legit"
    assert conf == pytest.approx(0.5)

    cls2, label2, conf2 = class_from_softmax_triple_with_p_fraud_threshold(
        0.2, 0.2, 0.6, p_fraud_threshold=0.4
    )
    assert cls2 == CLASS_FRAUD
    assert label2 == "fraud"
    assert conf2 == pytest.approx(0.6)


def test_map_fraud_when_p_at_or_above_fraud_threshold() -> None:
    cls, label, probs, conf = map_binary_to_risk(
        0.65,
        [],
        warn_threshold=0.35,
        fraud_threshold=0.65,
    )
    assert cls == CLASS_FRAUD
    assert label == "fraud"
    assert probs == (0.0, 0.0, 1.0)
    assert conf == pytest.approx(0.65)


def test_map_warning_when_p_between_thresholds_no_heuristic() -> None:
    cls, label, probs, conf = map_binary_to_risk(
        0.5,
        [],
        warn_threshold=0.35,
        fraud_threshold=0.65,
    )
    assert cls == CLASS_WARNING
    assert label == "warning"
    assert probs == (0.0, 1.0, 0.0)
    assert conf == pytest.approx(0.5)


def test_map_legit_when_p_low_no_heuristic() -> None:
    cls, label, probs, conf = map_binary_to_risk(
        0.2,
        [],
        warn_threshold=0.35,
        fraud_threshold=0.65,
    )
    assert cls == CLASS_LEGIT
    assert label == "legit"
    assert probs == (1.0, 0.0, 0.0)
    assert conf == pytest.approx(0.8)


def test_heuristic_forces_warning_even_when_p_low() -> None:
    cls, label, probs, conf = map_binary_to_risk(
        0.1,
        ["upfront_payment"],
        warn_threshold=0.35,
        fraud_threshold=0.65,
    )
    assert cls == CLASS_WARNING
    assert label == "warning"
    assert probs == (0.0, 1.0, 0.0)
    assert conf == pytest.approx(0.9)


def test_fraud_wins_over_heuristic_when_p_high() -> None:
    cls, label, probs, _ = map_binary_to_risk(
        0.9,
        ["upfront_payment"],
        warn_threshold=0.35,
        fraud_threshold=0.65,
    )
    assert cls == CLASS_FRAUD
    assert label == "fraud"
    assert probs == (0.0, 0.0, 1.0)


def test_p_clamped_to_unit_interval() -> None:
    cls, _, _, conf = map_binary_to_risk(
        1.5,
        [],
        warn_threshold=0.35,
        fraud_threshold=0.65,
    )
    assert cls == CLASS_FRAUD
    assert conf == pytest.approx(1.0)


def test_settings_rejects_warn_ge_fraud() -> None:
    with pytest.raises(ValidationError):
        Settings(warn_threshold=0.7, fraud_threshold=0.65)


def test_settings_rejects_path_to_placeholder() -> None:
    with pytest.raises(ValidationError, match="path/to"):
        Settings(phase6_fused_dir=Path("/path/to/artifacts/models/phase6_fused"))
