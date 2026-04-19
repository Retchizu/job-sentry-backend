"""Stratified train/val/test splits on row-level combined data (TICKET-002).

Consumes output from ``datasets_row_merge`` / ``combine_job_postings_rows.py``:
``combined_job_postings_rows.csv`` with ``risk_class`` and stable ``id``.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets_row_merge import TEXT_COLUMNS

MIN_CLASS_FOR_STRATIFY = 2
_WS_RE = re.compile(r"\s+")


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate text fields with null-safe strings; collapse whitespace (Phase 6 style)."""
    parts: list[pd.Series] = []
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing text column {col!r} for combined_text")
        s = df[col].fillna("").astype(str).str.strip()
        parts.append(s)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out.str.replace(_WS_RE, " ", regex=True).str.strip()


def _assert_stratify_possible(
    labels: pd.Series, *, context: str, label_col: str
) -> None:
    vc = labels.value_counts()
    too_small = vc[vc < MIN_CLASS_FOR_STRATIFY]
    if len(too_small) > 0:
        raise ValueError(
            f"Stratified split requires at least {MIN_CLASS_FOR_STRATIFY} rows per "
            f"class in {context} ({label_col!r}); per-class counts: "
            f"{vc.sort_index().to_dict()}"
        )


def stratified_train_val_test(
    df: pd.DataFrame,
    *,
    label_col: str = "risk_class",
    id_col: str = "id",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 70% / 15% / 15% with stratification on ``label_col`` (two-stage split)."""
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}")
    if id_col not in df.columns:
        raise ValueError(f"Missing id column {id_col!r}")

    y = df[label_col]
    _assert_stratify_possible(y, context="full dataframe", label_col=label_col)

    train_df, holdout_df = train_test_split(
        df,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )
    y_hold = holdout_df[label_col]
    _assert_stratify_possible(
        y_hold, context="30% holdout (before val/test)", label_col=label_col
    )

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.5,
        random_state=random_state,
        stratify=y_hold,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def assert_split_ids_disjoint(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_col: str = "id",
) -> None:
    """Raise if any ``id`` appears in more than one split."""
    s_train = set(train[id_col].astype(str))
    s_val = set(val[id_col].astype(str))
    s_test = set(test[id_col].astype(str))
    pairs = [
        ("train", "val", s_train & s_val),
        ("train", "test", s_train & s_test),
        ("val", "test", s_val & s_test),
    ]
    for a, b, inter in pairs:
        if inter:
            raise ValueError(
                f"Overlapping {id_col!r} between {a} and {b}: "
                f"{len(inter)} ids (example: {next(iter(inter))!r})"
            )


def split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    label_col: str = "risk_class",
) -> dict[str, Any]:
    """Row counts and per-split label value counts (JSON-serializable)."""
    def _counts(frame: pd.DataFrame) -> dict[str, int]:
        vc = frame[label_col].value_counts().sort_index()
        return {str(k): int(v) for k, v in vc.items()}

    return {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "total_rows": int(len(train) + len(val) + len(test)),
        f"{label_col}_counts": {
            "train": _counts(train),
            "val": _counts(val),
            "test": _counts(test),
        },
    }
