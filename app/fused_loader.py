"""Load phase6 fused artifacts: tokenizer, hyperparameters, weights.

Legacy 2-class parallel-fusion checkpoints (separate word BiLSTM branch, ``word_index.json``)
are **not** compatible with this loader: ``HybridFusedClassifier`` is sequential 3-class only.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import DistilBertTokenizerFast

from app.hybrid_fused_model import HybridFusedClassifier

logger = logging.getLogger(__name__)

_EXPECTED_THREE_CLASS_LABELS: tuple[str, str, str] = ("legit", "warning", "fraud")


def _validate_risk_class_labels(fused_meta: dict[str, Any]) -> None:
    """Warn when ``risk_class_labels`` (if present) disagrees with ``num_labels`` or app order."""
    num_labels = int(fused_meta.get("num_labels", 3))
    labels = fused_meta.get("risk_class_labels")
    if labels is None:
        return
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        logger.warning(
            "fused_meta risk_class_labels is not a list[str]; skipping validation (artifact_dir metadata)"
        )
        return
    if len(labels) != num_labels:
        logger.warning(
            "fused_meta risk_class_labels length (%d) != num_labels (%d)",
            len(labels),
            num_labels,
        )
    if num_labels == 3 and len(labels) == 3 and tuple(labels) != _EXPECTED_THREE_CLASS_LABELS:
        logger.warning(
            "fused_meta risk_class_labels %s does not match expected order %s",
            labels,
            list(_EXPECTED_THREE_CLASS_LABELS),
        )


# Training may save many epochs; production uses the checkpoint with best validation (here: epoch 8).
_DEFAULT_CHECKPOINT_EPOCH = 8


def _select_epoch_checkpoint(checkpoints_dir: Path) -> Path:
    candidates = sorted(checkpoints_dir.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No epoch_*.pt files under {checkpoints_dir}. "
            "Add model.safetensors to the artifact dir or train/export checkpoints."
        )

    preferred = checkpoints_dir / f"epoch_{_DEFAULT_CHECKPOINT_EPOCH:02d}.pt"
    if preferred.is_file():
        return preferred

    def epoch_num(p: Path) -> int:
        m = re.match(r"epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=epoch_num)


def resolve_weight_source(
    artifact_dir: Path,
    checkpoint_override: Optional[Path],
) -> tuple[str, Path]:
    """
    Prefer model.safetensors; else JOBSENTRY_PHASE6_FUSED_CHECKPOINT or epoch_08.pt
    (best validation) if present, else highest epoch_NN.pt.
    Returns (kind, path) where kind is 'safetensors' or 'checkpoint'.
    """
    safetensors_path = artifact_dir / "model.safetensors"
    if safetensors_path.is_file():
        return "safetensors", safetensors_path

    if checkpoint_override is not None:
        cp = Path(checkpoint_override)
        if not cp.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")
        return "checkpoint", cp

    ckpt_dir = artifact_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"No model.safetensors in {artifact_dir} and no checkpoints/ directory."
        )
    path = _select_epoch_checkpoint(ckpt_dir)
    return "checkpoint", path


def _torch_load_checkpoint(path: Path, map_location: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": map_location}
    try:
        return torch.load(path, **kwargs, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_weights_into_model(
    model: HybridFusedClassifier,
    kind: str,
    path: Path,
    map_location: torch.device,
) -> None:
    if kind == "safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        data = _torch_load_checkpoint(path, map_location)
        if "model_state" not in data:
            raise KeyError(f"Checkpoint {path} missing 'model_state' key")
        state = data["model_state"]
    model.load_state_dict(state, strict=True)


def load_fused_artifacts(
    artifact_dir: Path,
    *,
    checkpoint_override: Optional[Path] = None,
    map_location: Optional[torch.device] = None,
) -> tuple[
    HybridFusedClassifier,
    DistilBertTokenizerFast,
    dict[str, Any],
    str,
    Path,
]:
    """
    Build the sequential fused model, load weights, load tokenizer from ``artifact_dir``.

    Returns (model, tokenizer, fused_meta, weight_kind, weight_path).
    """
    artifact_dir = Path(artifact_dir).resolve()
    if not artifact_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {artifact_dir}")

    meta_path = artifact_dir / "fused_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing fused_meta.json in {artifact_dir}")

    with open(meta_path, encoding="utf-8") as f:
        fused_meta: dict[str, Any] = json.load(f)

    _validate_risk_class_labels(fused_meta)

    device = map_location or torch.device("cpu")
    kind, wpath = resolve_weight_source(artifact_dir, checkpoint_override)

    distilbert_name = str(fused_meta.get("distilbert_model", "distilbert-base-uncased"))
    num_labels = int(fused_meta.get("num_labels", 3))
    model = HybridFusedClassifier(
        lstm_hidden=int(fused_meta["lstm_hidden"]),
        fusion_hidden=int(fused_meta["fusion_hidden"]),
        num_labels=num_labels,
        dropout=float(fused_meta.get("dropout", 0.3)),
        distilbert_name=distilbert_name,
    )
    load_weights_into_model(model, kind, wpath, device)
    model.to(device)
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(artifact_dir))

    logger.info(
        "Loaded fused weights (%s) from %s (artifact_dir=%s)",
        kind,
        wpath,
        artifact_dir,
    )
    return model, tokenizer, fused_meta, kind, wpath
