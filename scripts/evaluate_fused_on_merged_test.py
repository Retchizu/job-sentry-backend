#!/usr/bin/env python3
"""Evaluate phase6 fused model on merged_test (TICKET-006). See cursor/project/notes/TICKET-006-evaluation-summary.md."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _repo_root() -> Path:
    return _REPO_ROOT


def main() -> int:
    root = _repo_root()
    default_csv = root / "artifacts" / "data" / "processed" / "merged_test.csv"
    default_art = root / "artifacts" / "models" / "phase6_fused"

    p = argparse.ArgumentParser(
        description="Run FusedScamPredictor on merged_test and print/save metrics."
    )
    p.add_argument(
        "--merged-test-csv",
        type=Path,
        default=default_csv,
        help=f"Default: {default_csv}",
    )
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Fused artifact dir (default: JOBSENTRY_PHASE6_FUSED_DIR or phase6_fused under artifacts).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write full metrics payload (e.g. artifacts/models/phase6_fused/validation_merged_test.json).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only first N rows (for quick CPU smoke tests).",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=int(os.environ.get("JOBSENTRY_MAX_BATCH_SIZE", "50")),
    )
    p.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
    )
    p.add_argument(
        "--omit-p-fraud",
        action="store_true",
        help="Do not write per-row p_fraud scores to JSON (smaller file; metrics unchanged).",
    )
    args = p.parse_args()

    art = args.artifact_dir
    if art is None:
        env = os.environ.get("JOBSENTRY_PHASE6_FUSED_DIR")
        art = Path(env) if env else default_art
    art = Path(art).expanduser().resolve()

    if not art.is_dir():
        print(f"ERROR: artifact dir not found: {art}", file=sys.stderr)
        return 1
    if not (art / "fused_meta.json").is_file():
        print(f"ERROR: missing fused_meta.json in {art}", file=sys.stderr)
        return 1

    import torch

    from app.merged_benchmark import evaluate_fused_on_merged_test

    dev = None
    if args.device == "cpu":
        dev = torch.device("cpu")
    elif args.device == "cuda":
        dev = torch.device("cuda")

    payload = evaluate_fused_on_merged_test(
        artifact_dir=art,
        merged_test_csv=args.merged_test_csv,
        device=dev,
        max_batch_size=args.max_batch_size,
        limit=args.limit,
        repo_root=root,
        include_p_fraud=not args.omit_p_fraud,
    )

    print(payload["classification_report"])
    print("Confusion matrix (rows=true, cols=pred):", payload["confusion_labels"])
    for row in payload["confusion_matrix"]:
        print(row)
    print(
        f"accuracy={payload['accuracy']:.4f} macro_f1={payload['macro_f1']:.4f} "
        f"weighted_f1={payload['weighted_f1']:.4f} ovr_auc={payload['ovr_auc']}"
    )
    fv = payload.get("fraud_vs_rest") or {}
    if fv:
        print(
            f"fraud_vs_rest roc_auc={fv.get('roc_auc')} "
            f"average_precision={fv.get('average_precision')}"
        )
    for k, v in payload.items():
        if k.startswith("binary_vs_fraudulent__"):
            print(k, v)

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = {k: v for k, v in payload.items() if k != "classification_report"}
        to_save["classification_report"] = payload["classification_report"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
