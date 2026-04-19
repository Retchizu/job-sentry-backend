#!/usr/bin/env python3
"""CLI: stratified train/val/test from combined row-level CSV (TICKET-002)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root = parent of scripts/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import datasets_row_merge as drm  # noqa: E402
import datasets_row_splits as drs  # noqa: E402


def _default_input() -> Path:
    return _ROOT / "artifacts" / "datasets" / "combined_job_postings_rows.csv"


def _default_out_dir() -> Path:
    return _ROOT / "artifacts" / "data" / "processed"


def _output_frame(df: pd.DataFrame) -> pd.DataFrame:
    combined = drs.build_combined_text(df)
    out = df.copy()
    if "combined_text" in out.columns:
        out = out.drop(columns=["combined_text"])
    out["combined_text"] = combined
    preferred = [c for c in drm.REQUIRED_COLUMNS if c in out.columns]
    rest = [
        c
        for c in out.columns
        if c not in preferred and c != "combined_text"
    ]
    return out[preferred + rest + ["combined_text"]]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build combined_text and stratified merged_train/val/test CSVs (TICKET-002)."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=_default_input(),
        help="Combined CSV from combine_job_postings_rows.py",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="Directory for merged_*.csv and summary JSON",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default 42)",
    )
    p.add_argument(
        "--job-only",
        action="store_true",
        help='Keep only rows with dataset_source == "job_rows" before splitting',
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if args.job_only:
        if "dataset_source" not in df.columns:
            raise ValueError("job_only requires dataset_source column")
        df = df.loc[df["dataset_source"] == "job_rows"].reset_index(drop=True)

    train_df, val_df, test_df = drs.stratified_train_val_test(
        df,
        label_col="risk_class",
        id_col="id",
        random_state=args.seed,
    )
    drs.assert_split_ids_disjoint(train_df, val_df, test_df, id_col="id")

    train_out = _output_frame(train_df)
    val_out = _output_frame(val_df)
    test_out = _output_frame(test_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "merged_train.csv"
    val_path = args.out_dir / "merged_val.csv"
    test_path = args.out_dir / "merged_test.csv"
    summary_path = args.out_dir / "merged_splits.summary.json"

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    summary: dict = {
        "schema_version": "1.0",
        "input_csv": str(args.input.resolve()),
        "seed": args.seed,
        "job_only": args.job_only,
        "outputs": {
            "merged_train": str(train_path.resolve()),
            "merged_val": str(val_path.resolve()),
            "merged_test": str(test_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
        "ids_disjoint": True,
        **drs.split_summary(train_df, val_df, test_df, label_col="risk_class"),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
