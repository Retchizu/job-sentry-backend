"""Merge fake_job_postings_rows + job_postings_rows for TICKET-007.

See cursor/project/plan/2026-04-18-TICKET-007-combine-row-datasets.md for rules.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "job_title",
    "job_desc",
    "skills_desc",
    "company_profile",
    "rate_min",
    "rate_max",
    "currency",
    "rate_type",
    "created_at",
    "fraudulent",
    "warnings",
)

TEXT_COLUMNS: tuple[str, ...] = (
    "job_title",
    "job_desc",
    "skills_desc",
    "company_profile",
)

_WS_RE = re.compile(r"\s+")


def assert_same_columns(fake_df: pd.DataFrame, job_df: pd.DataFrame) -> None:
    c1 = set(fake_df.columns)
    c2 = set(job_df.columns)
    if c1 != c2:
        raise ValueError(
            f"Column mismatch: fake_only={c1 - c2!r} job_only={c2 - c1!r}"
        )
    missing_fake = set(REQUIRED_COLUMNS) - c1
    missing_job = set(REQUIRED_COLUMNS) - c2
    if missing_fake or missing_job:
        raise ValueError(
            f"Missing required columns: fake {missing_fake!r} job {missing_job!r}"
        )


def assign_source(
    df: pd.DataFrame, source: Literal["fake_rows", "job_rows"]
) -> pd.DataFrame:
    out = df.copy()
    out["dataset_source"] = source
    return out


def normalized_text_key(row: pd.Series) -> str:
    parts: list[str] = []
    for col in TEXT_COLUMNS:
        raw = row.get(col, "")
        if pd.isna(raw):
            s = ""
        else:
            s = str(raw)
        s = unicodedata.normalize("NFC", s).strip()
        parts.append(s)
    joined = " ".join(parts)
    joined = _WS_RE.sub(" ", joined).strip()
    return joined.casefold()


def parse_warnings_flags(raw: Any) -> tuple[bool, bool]:
    """Return (has_non_empty_flags, parse_error).

    has_non_empty_flags: True if JSON parses to an object with non-empty ``flags`` list.
    parse_error: True if JSON is invalid or ``flags`` is present but not a list.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return False, False
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return False, False
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return False, True
    if not isinstance(obj, dict):
        return False, True
    if "flags" not in obj:
        return False, False
    flags = obj["flags"]
    if not isinstance(flags, list):
        return False, True
    return len(flags) > 0, False


def normalize_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalize dtypes; drop rows with invalid ``fraudulent``. Returns (df, n_dropped)."""
    out = df.copy()
    for col in TEXT_COLUMNS:
        out[col] = out[col].fillna("").astype(str)
    out["warnings"] = out["warnings"].apply(
        lambda x: "" if pd.isna(x) or str(x).strip().lower() == "nan" else str(x)
    )
    out["id"] = out["id"].fillna("").astype(str)
    for col in ("currency", "rate_type"):
        out[col] = out[col].fillna("").astype(str)
    out["created_at"] = out["created_at"].apply(
        lambda x: ""
        if pd.isna(x) or str(x).strip().lower() == "nan"
        else str(x).strip()
    )
    out["rate_min"] = pd.to_numeric(out["rate_min"], errors="coerce")
    out["rate_max"] = pd.to_numeric(out["rate_max"], errors="coerce")
    out["fraudulent"] = pd.to_numeric(out["fraudulent"], errors="coerce")
    bad = out["fraudulent"].isna()
    dropped = int(bad.sum())
    out = out.loc[~bad].copy()
    out["fraudulent"] = out["fraudulent"].astype(int)
    return out, dropped


def dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, before - len(out)


def dedupe_near(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    keys = df.apply(normalized_text_key, axis=1)
    dup = keys.duplicated(keep="first")
    out = df.loc[~dup].reset_index(drop=True)
    return out, before - len(out)


def align_user_risk_class_columns(
    fake_df: pd.DataFrame, job_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """So legacy CSVs without ``user_risk_class`` still pair-merge with the same column set."""
    f = fake_df.copy()
    j = job_df.copy()
    if "user_risk_class" not in f.columns:
        f["user_risk_class"] = np.nan
    if "user_risk_class" not in j.columns:
        j["user_risk_class"] = np.nan
    return f, j


def derive_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add warning_label and risk_class; return parse error count for warnings JSON."""
    out = df.copy()
    if "user_risk_class" not in out.columns:
        out["user_risk_class"] = np.nan

    parse_errors = 0
    has_flags_list: list[int] = []
    for raw in out["warnings"]:
        has_f, err = parse_warnings_flags(raw)
        if err:
            parse_errors += 1
        has_flags_list.append(1 if has_f else 0)
    out["warning_label"] = has_flags_list
    f = out["fraudulent"].to_numpy(dtype=int)
    wl = np.asarray(has_flags_list, dtype=int)
    legacy_rc = np.where(f == 1, 2, np.where(wl == 1, 1, 0))

    ur = pd.to_numeric(out["user_risk_class"], errors="coerce")
    valid = ur.notna() & ((ur == 0) | (ur == 1) | (ur == 2))
    ur_round = ur.round()
    legacy_series = pd.Series(legacy_rc, index=out.index)
    out["risk_class"] = ur_round.where(valid, legacy_series).astype(int)
    return out, parse_errors


def merge_dataframes(fake_df: pd.DataFrame, job_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full pipeline; returns merged dataframe and summary (pre-JSON)."""
    fake_df, job_df = align_user_risk_class_columns(fake_df, job_df)
    assert_same_columns(fake_df, job_df)
    fake_n = len(fake_df)
    job_n = len(job_df)
    f1 = assign_source(fake_df, "fake_rows")
    f2 = assign_source(job_df, "job_rows")
    merged = pd.concat([f1, f2], ignore_index=True)
    concat_before = len(merged)
    merged, dropped_fraud = normalize_dtypes(merged)
    merged, exact_rm = dedupe_exact(merged)
    merged, near_rm = dedupe_near(merged)
    merged, warn_parse_err = derive_labels(merged)
    final_n = len(merged)

    fraudulent_counts = merged["fraudulent"].value_counts().to_dict()
    wl_counts = merged["warning_label"].value_counts().to_dict()
    rc_counts = merged["risk_class"].value_counts().to_dict()

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "inputs": {"fake_rows_read": fake_n, "job_rows_read": job_n},
        "dropped_invalid_fraudulent": dropped_fraud,
        "concat_rows_before_dedupe": concat_before,
        "exact_duplicates_removed": exact_rm,
        "near_duplicates_removed": near_rm,
        "warnings_parse_errors": warn_parse_err,
        "final_rows": final_n,
        "fraudulent_counts": {str(k): int(v) for k, v in fraudulent_counts.items()},
        "warning_label_counts": {str(k): int(v) for k, v in wl_counts.items()},
        "risk_class_counts": {str(k): int(v) for k, v in rc_counts.items()},
        "rules": [
            "warning_label=1 iff warnings JSON has non-empty flags list; invalid JSON counts in warnings_parse_errors.",
            "risk_class: if user_risk_class is 0, 1, or 2 (reviewer label from export), use it; else legacy: fraudulent==1 -> 2; elif warning_label==1 -> 1; else 0 (FE-TICKET-004 / TICKET-001).",
            "Exact dedupe: all columns including dataset_source.",
            "Near dedupe: first row per normalized text key (NFC, casefold, whitespace collapsed) in concat order.",
        ],
    }
    return merged, summary


def merge_sources(path_fake: Path | str, path_job: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    p1 = Path(path_fake)
    p2 = Path(path_job)
    fake_df = pd.read_csv(p1)
    job_df = pd.read_csv(p2)
    merged, summary = merge_dataframes(fake_df, job_df)
    summary["inputs"] = {
        "fake_path": str(p1.resolve()),
        "job_path": str(p2.resolve()),
        "fake_rows_read": summary["inputs"]["fake_rows_read"],
        "job_rows_read": summary["inputs"]["job_rows_read"],
    }
    return merged, summary
