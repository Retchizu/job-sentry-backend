"""Tests for TICKET-007 row dataset merge (datasets_row_merge)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import datasets_row_merge as drm

_FIX = Path(__file__).resolve().parent / "fixtures"


def test_parse_warnings_flags() -> None:
    assert drm.parse_warnings_flags(None) == (False, False)
    assert drm.parse_warnings_flags(float("nan")) == (False, False)
    assert drm.parse_warnings_flags("") == (False, False)
    assert drm.parse_warnings_flags("{}") == (False, False)
    assert drm.parse_warnings_flags('{"flags":[]}') == (False, False)
    assert drm.parse_warnings_flags('{"flags":["a"]}') == (True, False)
    assert drm.parse_warnings_flags("not json") == (False, True)
    assert drm.parse_warnings_flags('{"flags": "bad"}') == (False, True)


def test_risk_class_from_user_risk_class() -> None:
    """Reviewer user_risk_class=1 with fraudulent=0 yields risk_class=1 (FE-TICKET-004)."""
    df2, err = drm.derive_labels(
        pd.DataFrame(
            {
                "id": ["w"],
                "job_title": [""],
                "job_desc": [""],
                "skills_desc": [""],
                "company_profile": [""],
                "rate_min": [pd.NA],
                "rate_max": [pd.NA],
                "currency": [""],
                "rate_type": [""],
                "created_at": [""],
                "fraudulent": [0],
                "warnings": ['{"flags":["x"]}'],
                "user_risk_class": [1],
                "dataset_source": ["job_rows"],
            }
        )
    )
    assert err == 0
    assert list(df2["risk_class"]) == [1]


def test_risk_class_precedence() -> None:
    df2, err = drm.derive_labels(
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "job_title": ["", "", ""],
                "job_desc": ["", "", ""],
                "skills_desc": ["", "", ""],
                "company_profile": ["", "", ""],
                "rate_min": [pd.NA, pd.NA, pd.NA],
                "rate_max": [pd.NA, pd.NA, pd.NA],
                "currency": ["", "", ""],
                "rate_type": ["", "", ""],
                "created_at": ["", "", ""],
                "fraudulent": [1, 0, 0],
                "warnings": ['{"flags":["x"]}', '{"flags":["x"]}', ""],
                "dataset_source": ["fake_rows", "fake_rows", "fake_rows"],
            }
        )
    )
    assert err == 0
    assert list(df2["risk_class"]) == [2, 1, 0]


def test_normalized_text_key_collapses_whitespace() -> None:
    r = pd.Series(
        {
            "job_title": "  Hello  ",
            "job_desc": "World",
            "skills_desc": "",
            "company_profile": "",
        }
    )
    k1 = drm.normalized_text_key(r)
    r2 = pd.Series(
        {
            "job_title": "hello",
            "job_desc": "world",
            "skills_desc": "",
            "company_profile": "",
        }
    )
    k2 = drm.normalized_text_key(r2)
    assert k1 == k2


def test_merge_tiny_fixtures_near_dedupe() -> None:
    fake = pd.read_csv(_FIX / "fake_job_postings_rows_tiny.csv")
    job = pd.read_csv(_FIX / "job_postings_rows_tiny.csv")
    merged, summary = drm.merge_dataframes(fake, job)
    assert summary["concat_rows_before_dedupe"] == 6
    assert summary["near_duplicates_removed"] == 1
    assert summary["final_rows"] == 5
    assert set(merged["dataset_source"]) == {"fake_rows", "job_rows"}
    # j1 dropped as near-dup of f3; f3 kept first
    assert "j1" not in set(merged["id"])


def test_merge_sources_paths_and_json_roundtrip(tmp_path: Path) -> None:
    out_csv = tmp_path / "out.csv"
    out_js = tmp_path / "summary.json"
    merged, summary = drm.merge_sources(
        _FIX / "fake_job_postings_rows_tiny.csv",
        _FIX / "job_postings_rows_tiny.csv",
    )
    out_order = list(drm.REQUIRED_COLUMNS) + [
        "user_risk_class",
        "dataset_source",
        "warning_label",
        "risk_class",
    ]
    merged[out_order].to_csv(out_csv, index=False)
    summary["outputs"] = {
        "combined_csv": str(out_csv),
        "summary_json": str(out_js),
    }
    with open(out_js, "w", encoding="utf-8") as f:
        json.dump(summary, f)
    loaded = pd.read_csv(out_csv)
    assert "risk_class" in loaded.columns
    assert "dataset_source" in loaded.columns
    with open(out_js, encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert "fraudulent_counts" in data
    assert "rules" in data


def test_column_mismatch_raises() -> None:
    fake = pd.read_csv(_FIX / "fake_job_postings_rows_tiny.csv")
    bad = fake.drop(columns=["id"])
    with pytest.raises(ValueError, match="Column mismatch|Missing required"):
        drm.merge_dataframes(fake, bad)
