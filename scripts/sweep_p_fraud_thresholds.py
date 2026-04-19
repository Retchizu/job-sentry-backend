#!/usr/bin/env python3
"""Summarize P(fraud) >= tau vs argmax (from validation JSON with y_true + p_fraud + y_pred)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Threshold sweep on p_fraud from evaluate_fused_on_merged_test JSON output."
    )
    p.add_argument(
        "validation_json",
        type=Path,
        nargs="?",
        default=_REPO_ROOT
        / "artifacts"
        / "models"
        / "phase6_fused"
        / "validation_merged_test.json",
        help="JSON with y_true, p_fraud, and optionally y_pred",
    )
    p.add_argument(
        "--n-thresholds",
        type=int,
        default=1001,
        help="Number of tau values in [0, 1] (default 1001).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write summarize_p_fraud_threshold_sweep() result to this path.",
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
            "ERROR: JSON must contain 'y_true' and 'p_fraud' "
            "(re-run scripts/evaluate_fused_on_merged_test.py without --omit-p-fraud).",
            file=sys.stderr,
        )
        return 1

    import numpy as np

    from app.merged_benchmark import summarize_p_fraud_threshold_sweep

    y_true = np.array(data["y_true"], dtype=np.int64)
    p_fraud = np.array(data["p_fraud"], dtype=np.float64)
    y_pred = np.array(data["y_pred"], dtype=np.int64) if "y_pred" in data else None

    summary = summarize_p_fraud_threshold_sweep(
        y_true,
        p_fraud,
        y_pred_class=y_pred,
        n_thresholds=args.n_thresholds,
    )

    print(f"Source: {path}")
    print(f"n_rows={summary['n_rows']} true_fraud={summary['positives_true_fraud']}")
    bf = summary["best_f1"]
    print(
        "Best binary rule P(fraud)>=tau (true label = fraud class):\n"
        f"  tau={bf['threshold']:.6f}  P={bf['precision']:.4f}  "
        f"R={bf['recall']:.4f}  F1={bf['f1']:.4f}  pred_pos={bf['positives_pred']}"
    )
    if "argmax_predict_fraud_class2" in summary:
        ax = summary["argmax_predict_fraud_class2"]
        print(
            "Argmax multiclass (pred class == fraud):\n"
            f"  P={ax['precision']:.4f}  R={ax['recall']:.4f}  "
            f"F1={ax['f1']:.4f}  pred_pos={ax['positives_pred']}"
        )
    print("Max recall at minimum precision (tau sweep):")
    for k, v in summary["max_recall_at_min_precision"].items():
        if v is None:
            print(f"  min_precision={k}: (no point on grid)")
        else:
            print(
                f"  min_precision={k}: tau={v['threshold']:.6f}  "
                f"P={v['precision']:.4f}  R={v['recall']:.4f}  "
                f"pred_pos={v['positives_pred']}"
            )

    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
