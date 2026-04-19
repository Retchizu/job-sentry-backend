"""Tests for TICKET-006 binary collapse helpers."""

from __future__ import annotations

import numpy as np

from app.merged_benchmark import (
    binary_pred_fraud_only,
    binary_pred_warning_or_fraud,
    fraud_vs_rest_ranking_metrics,
    metrics_p_fraud_ge_threshold,
    summarize_p_fraud_threshold_sweep,
)


def test_binary_pred_fraud_only() -> None:
    y = np.array([0, 1, 2, 2])
    assert np.array_equal(binary_pred_fraud_only(y), np.array([0, 0, 1, 1]))


def test_binary_pred_warning_or_fraud() -> None:
    y = np.array([0, 1, 2, 0])
    assert np.array_equal(binary_pred_warning_or_fraud(y), np.array([0, 1, 1, 0]))


def test_fraud_vs_rest_ranking_metrics_perfect_separation() -> None:
    y = np.array([0, 1, 2, 0, 2])
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.85, 0.1, 0.05],
            [0.05, 0.05, 0.9],
        ],
        dtype=np.float64,
    )
    m = fraud_vs_rest_ranking_metrics(y, probs)
    assert m["roc_auc"] == 1.0
    assert m["average_precision"] == 1.0


def test_fraud_vs_rest_ranking_metrics_single_class_nan() -> None:
    y = np.array([0, 1, 1])
    probs = np.ones((3, 3), dtype=np.float64) / 3.0
    m = fraud_vs_rest_ranking_metrics(y, probs)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["average_precision"])


def test_metrics_p_fraud_ge_threshold() -> None:
    y_true = np.array([0, 2, 2], dtype=np.int64)
    p_fraud = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    m = metrics_p_fraud_ge_threshold(y_true, p_fraud, 0.5)
    assert m["true_positives"] == 2
    assert m["false_positives"] == 0
    assert m["false_negatives"] == 0


def test_summarize_p_fraud_threshold_sweep_beats_argmax_when_scores_rank_well() -> None:
    """Fraud rows get high p_fraud; argmax still picks legit — sweep can recover recall."""
    y_true = np.array([0, 0, 2, 2], dtype=np.int64)
    p_fraud = np.array([0.01, 0.01, 0.45, 0.55], dtype=np.float64)
    y_pred = np.array([0, 0, 0, 2], dtype=np.int64)
    s = summarize_p_fraud_threshold_sweep(
        y_true, p_fraud, y_pred_class=y_pred, n_thresholds=101
    )
    ax = s["argmax_predict_fraud_class2"]
    assert ax["positives_pred"] == 1
    bf = s["best_f1"]
    assert bf["recall"] >= ax["recall"]
    assert bf["f1"] >= ax["f1"]
