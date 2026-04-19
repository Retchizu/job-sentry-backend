#!/usr/bin/env python3
"""
Apply ONE chosen threshold tau to saved scores (beginner-friendly).

Rule: "predict fraud" for this check when P(fraud) >= tau.
True fraud = risk label is fraud (class 2).

Example (pick tau on val, then measure on test once):

  python scripts/apply_p_fraud_threshold.py \\
    artifacts/models/phase6_fused/validation_merged_val.json --tau 0.4

  python scripts/apply_p_fraud_threshold.py \\
    artifacts/models/phase6_fused/validation_merged_test.json --tau 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.merged_benchmark import metrics_p_fraud_ge_threshold


def main() -> int:
    p = argparse.ArgumentParser(
        description="Binary metrics for P(fraud) >= tau using y_true + p_fraud from eval JSON."
    )
    p.add_argument(
        "validation_json",
        type=Path,
        help="JSON from evaluate_fused_on_merged_test.py (needs y_true + p_fraud).",
    )
    p.add_argument(
        "--tau",
        type=float,
        required=True,
        help="Threshold between 0 and 1 (same tau you chose from the val sweep).",
    )
    args = p.parse_args()

    path = Path(args.validation_json).expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "p_fraud" not in data or "y_true" not in data:
        print(
            "ERROR: JSON must contain 'y_true' and 'p_fraud'.",
            file=sys.stderr,
        )
        return 1

    y_true = np.array(data["y_true"], dtype=np.int64)
    p_fraud = np.array(data["p_fraud"], dtype=np.float64)
    m = metrics_p_fraud_ge_threshold(y_true, p_fraud, args.tau)

    print(f"File: {path}")
    print(f"Rule: predict fraud when P(fraud) >= {m['threshold']}")
    print(f"Rows: {len(y_true)}  true fraud cases: {m['positives_true_fraud']}")
    print(
        f"Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}"
    )
    print(f"Predicted fraud count: {m['positives_pred']}")
    print(
        f"Confusion (fraud positive): TP={m['true_positives']} FP={m['false_positives']} "
        f"FN={m['false_negatives']} TN={m['true_negatives']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
