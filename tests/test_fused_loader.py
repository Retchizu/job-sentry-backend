"""Tests for fused artifact weight resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.fused_loader import _validate_risk_class_labels, resolve_weight_source


def test_resolve_prefers_safetensors(tmp_path: Path) -> None:
    d = tmp_path / "art"
    d.mkdir()
    (d / "checkpoints").mkdir()
    (d / "checkpoints" / "epoch_99.pt").write_bytes(b"y")  # ignored when safetensors exists
    (d / "model.safetensors").write_bytes(b"x")
    kind, path = resolve_weight_source(d, None)
    assert kind == "safetensors"
    assert path.name == "model.safetensors"


def test_resolve_checkpoint_override(tmp_path: Path) -> None:
    d = tmp_path / "art"
    d.mkdir()
    ckpt = d / "my.pt"
    ckpt.write_bytes(b"{}")
    kind, path = resolve_weight_source(d, ckpt)
    assert kind == "checkpoint"
    assert path == ckpt


def test_resolve_prefers_epoch_8_when_present(tmp_path: Path) -> None:
    d = tmp_path / "art"
    ck = d / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "epoch_03.pt").write_bytes(b"a")
    (ck / "epoch_08.pt").write_bytes(b"best")
    (ck / "epoch_12.pt").write_bytes(b"b")
    kind, path = resolve_weight_source(d, None)
    assert kind == "checkpoint"
    assert path.name == "epoch_08.pt"


def test_resolve_highest_epoch_when_epoch_8_missing(tmp_path: Path) -> None:
    d = tmp_path / "art"
    ck = d / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "epoch_03.pt").write_bytes(b"a")
    (ck / "epoch_12.pt").write_bytes(b"b")
    (ck / "epoch_07.pt").write_bytes(b"c")
    kind, path = resolve_weight_source(d, None)
    assert kind == "checkpoint"
    assert path.name == "epoch_12.pt"


def test_resolve_missing_weights_raises(tmp_path: Path) -> None:
    d = tmp_path / "art"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_weight_source(d, None)


def test_validate_risk_class_labels_warns_on_length_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    _validate_risk_class_labels({"num_labels": 3, "risk_class_labels": ["legit", "warning"]})
    assert "length" in caplog.text


def test_validate_risk_class_labels_warns_on_wrong_order(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    _validate_risk_class_labels(
        {"num_labels": 3, "risk_class_labels": ["fraud", "warning", "legit"]}
    )
    assert "does not match expected order" in caplog.text


def test_validate_risk_class_labels_ok_when_aligned(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    _validate_risk_class_labels(
        {"num_labels": 3, "risk_class_labels": ["legit", "warning", "fraud"]}
    )
    assert caplog.text == ""
