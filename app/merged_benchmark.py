"""Benchmark fused 3-class model on merged_test (TICKET-006). Used by scripts and tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.fused_predictor import FusedScamPredictor, resolve_device


def binary_pred_fraud_only(y_pred_class: np.ndarray) -> np.ndarray:
    """Binary positive = predicted fraud (class 2)."""
    return (y_pred_class == 2).astype(np.int64)


def binary_pred_warning_or_fraud(y_pred_class: np.ndarray) -> np.ndarray:
    """Binary positive = predicted warning or fraud (classes 1 or 2)."""
    return ((y_pred_class == 1) | (y_pred_class == 2)).astype(np.int64)


def fraud_vs_rest_ranking_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    """
    Binary positive = true class is fraud (2). Score = predicted P(fraud) = probs[:, 2].

    Use this to judge whether fraud is *ranked* well (AP / ROC-AUC) independently of argmax.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if y.shape[0] == 0:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    y_fraud = (y == 2).astype(np.int64)
    scores = p[:, 2]
    n_pos = int(y_fraud.sum())
    n_neg = int(len(y_fraud) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    roc = float(roc_auc_score(y_fraud, scores))
    ap = float(average_precision_score(y_fraud, scores))
    return {"roc_auc": roc, "average_precision": ap}


def binary_true_fraud(y_true: np.ndarray) -> np.ndarray:
    """Binary positive = true class is fraud (risk_class == 2)."""
    return (np.asarray(y_true, dtype=np.int64) == 2).astype(np.int64)


def p_fraud_threshold_sweep_rows(
    y_true: np.ndarray,
    p_fraud: np.ndarray,
    *,
    n_thresholds: int = 1001,
) -> list[dict[str, float | int]]:
    """
    Binary rule: predict fraud iff P(fraud) >= tau. Sweep tau in ``linspace(0, 1, n_thresholds)``.

    Positive label for metrics: true fraud (class 2).
    """
    y = binary_true_fraud(y_true)
    scores = np.asarray(p_fraud, dtype=np.float64)
    if len(y) != len(scores):
        raise ValueError("y_true and p_fraud must have the same length")
    taus = np.linspace(0.0, 1.0, int(n_thresholds), endpoint=True)
    rows: list[dict[str, float | int]] = []
    for tau in taus:
        pred = (scores >= float(tau)).astype(np.int64)
        p, r, f, _ = precision_recall_fscore_support(
            y, pred, average="binary", pos_label=1, zero_division=0
        )
        rows.append(
            {
                "threshold": float(tau),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "positives_pred": int(pred.sum()),
            }
        )
    return rows


def summarize_p_fraud_threshold_sweep(
    y_true: np.ndarray,
    p_fraud: np.ndarray,
    y_pred_class: np.ndarray | None = None,
    *,
    n_thresholds: int = 1001,
    min_precision_targets: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
) -> dict[str, Any]:
    """
    Summarize sweep: best F1, best recall subject to minimum precision, vs argmax fraud baseline.

    If ``y_pred_class`` is provided (multiclass 0/1/2), reports ``argmax_predict_fraud`` binary metrics
    for comparison (predict positive iff predicted class == 2).
    """
    rows = p_fraud_threshold_sweep_rows(
        y_true, p_fraud, n_thresholds=n_thresholds
    )
    best_f1 = max(rows, key=lambda r: r["f1"])
    out: dict[str, Any] = {
        "n_rows": len(np.asarray(y_true)),
        "positives_true_fraud": int(binary_true_fraud(y_true).sum()),
        "n_thresholds": int(n_thresholds),
        "best_f1": dict(best_f1),
        "max_recall_at_min_precision": {},
    }
    for mp in min_precision_targets:
        eligible = [r for r in rows if r["precision"] + 1e-12 >= mp]
        if not eligible:
            out["max_recall_at_min_precision"][str(mp)] = None
            continue
        best_r = max(eligible, key=lambda r: r["recall"])
        out["max_recall_at_min_precision"][str(mp)] = dict(best_r)

    if y_pred_class is not None:
        yp = np.asarray(y_pred_class, dtype=np.int64)
        pred_f = binary_pred_fraud_only(yp)
        y = binary_true_fraud(y_true)
        p, r, f, _ = precision_recall_fscore_support(
            y, pred_f, average="binary", pos_label=1, zero_division=0
        )
        out["argmax_predict_fraud_class2"] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "positives_pred": int(pred_f.sum()),
        }

    return out


def metrics_p_fraud_ge_threshold(
    y_true: np.ndarray,
    p_fraud: np.ndarray,
    tau: float,
) -> dict[str, float | int]:
    """
    Binary rule: predict fraud iff ``P(fraud) >= tau``. Positive class = true fraud (risk_class 2).
    """
    y = binary_true_fraud(y_true)
    scores = np.asarray(p_fraud, dtype=np.float64)
    pred = (scores >= float(tau)).astype(np.int64)
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {
        "threshold": float(tau),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "positives_true_fraud": int(y.sum()),
        "positives_pred": int(pred.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def _git_head_short(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()[:12] or None
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate_fused_on_merged_test(
    *,
    artifact_dir: Path,
    merged_test_csv: Path,
    device: Optional[torch.device] = None,
    max_batch_size: int = 50,
    limit: Optional[int] = None,
    repo_root: Optional[Path] = None,
    include_p_fraud: bool = True,
) -> dict[str, Any]:
    """
    Run production-aligned inference on merged_test.csv; return metrics dict (JSON-serializable).

    Binary comparison uses column ``fraudulent`` when present (0/1). Two predicted-positive
    definitions are reported: fraud-only (argmax==2) vs any risk (argmax in {1,2}).
    """
    artifact_dir = Path(artifact_dir).resolve()
    merged_test_csv = Path(merged_test_csv).resolve()
    df = pd.read_csv(merged_test_csv)
    if limit is not None:
        df = df.head(int(limit)).copy()

    texts = df["combined_text"].astype(str).tolist()
    y_true = df["risk_class"].astype(int).values
    n = len(df)

    dev = device or resolve_device(None)
    predictor = FusedScamPredictor.from_artifact_dir(
        artifact_dir,
        device=dev,
        max_batch_size=max_batch_size,
    )
    triples = predictor.predict_risk_distribution(texts)
    probs = np.array(triples, dtype=np.float64)
    y_pred = np.argmax(probs, axis=1).astype(np.int64)

    labels = [0, 1, 2]
    target_names = ["legit", "warning", "fraud"]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    )
    try:
        ovr_auc = float(
            roc_auc_score(y_true, probs, multi_class="ovr", labels=labels)
        )
    except ValueError:
        ovr_auc = float("nan")

    fraud_rank = fraud_vs_rest_ranking_metrics(y_true, probs)
    p_fraud = probs[:, 2].tolist()

    out: dict[str, Any] = {
        "n_rows": n,
        "merged_test_csv": str(merged_test_csv),
        "artifact_dir": str(artifact_dir),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "ovr_auc": ovr_auc,
        "fraud_vs_rest": fraud_rank,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": target_names,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    if include_p_fraud:
        out["p_fraud"] = p_fraud

    fused_meta_path = artifact_dir / "fused_meta.json"
    artifact_version: Optional[str] = None
    if fused_meta_path.is_file():
        import json

        with open(fused_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        artifact_version = meta.get("artifact_version")
        out["fused_meta_artifact_version"] = artifact_version

    rr = repo_root
    if rr is None:
        rr = merged_test_csv.parent.parent.parent.parent
    git_c = _git_head_short(rr) if rr and rr.is_dir() else None
    if git_c:
        out["git_commit"] = git_c

    if "fraudulent" in df.columns:
        y_bin = df["fraudulent"].astype(int).values
        pred_fraud = binary_pred_fraud_only(y_pred)
        pred_scam = binary_pred_warning_or_fraud(y_pred)
        for name, y_p in (
            ("pred_fraud_only_argmax_2", pred_fraud),
            ("pred_warning_or_fraud_argmax_1_or_2", pred_scam),
        ):
            p, r, f, _ = precision_recall_fscore_support(
                y_bin, y_p, average="binary", pos_label=1, zero_division=0
            )
            out[f"binary_vs_fraudulent__{name}"] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "positives_true": int(y_bin.sum()),
                "positives_pred": int(y_p.sum()),
            }

    return out


def error_analysis_rows(
    y_true: list[int],
    y_pred: list[int],
    texts: list[str],
    *,
    max_chars: int = 400,
    k: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """
    Collect example rows for fraud/warning FP/FN (truncated text for display).
    """
    yt = np.array(y_true, dtype=np.int64)
    yp = np.array(y_pred, dtype=np.int64)
    t_arr = np.array(texts, dtype=object)

    def _take(mask: np.ndarray) -> list[dict[str, Any]]:
        idx = np.flatnonzero(mask)[:k]
        rows: list[dict[str, Any]] = []
        for i in idx:
            s = str(t_arr[i])
            rows.append(
                {
                    "index": int(i),
                    "y_true": int(yt[i]),
                    "y_pred": int(yp[i]),
                    "text_excerpt": s[:max_chars] + ("…" if len(s) > max_chars else ""),
                }
            )
        return rows

    return {
        "fraud_false_negative": _take((yt == 2) & (yp != 2)),
        "fraud_false_positive": _take((yt != 2) & (yp == 2)),
        "warning_false_negative": _take((yt == 1) & (yp != 1)),
        "warning_false_positive": _take((yt != 1) & (yp == 1)),
    }
