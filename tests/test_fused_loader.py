"""Tests for fused artifact weight resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.fused_loader import resolve_weight_source


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


def test_resolve_highest_epoch(tmp_path: Path) -> None:
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
