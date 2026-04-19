"""Tests for TICKET-002 row-level stratified splits (datasets_row_splits)."""

from __future__ import annotations

import pandas as pd
import pytest

import datasets_row_splits as drs


def _minimal_row(
    rid: str,
    *,
    t0: str = "a",
    t1: str = "b",
    t2: str = "c",
    t3: str = "d",
    risk: int = 0,
) -> dict:
    return {
        "id": rid,
        "job_title": t0,
        "job_desc": t1,
        "skills_desc": t2,
        "company_profile": t3,
        "rate_min": 1.0,
        "rate_max": 2.0,
        "currency": "USD",
        "rate_type": "hr",
        "created_at": "2020-01-01",
        "fraudulent": 0,
        "warnings": "",
        "dataset_source": "job_rows",
        "warning_label": 0,
        "risk_class": risk,
    }


def test_build_combined_text_nan_and_strip() -> None:
    df = pd.DataFrame(
        {
            "job_title": ["  Hello ", pd.NA],
            "job_desc": [float("nan"), "There"],
            "skills_desc": ["", " "],
            "company_profile": ["World", None],
        }
    )
    s = drs.build_combined_text(df)
    assert s.iloc[0] == "Hello World"
    assert s.iloc[1] == "There"


def test_stratified_sizes_disjoint_and_proportions() -> None:
    rows = []
    for cls in (0, 1, 2):
        for i in range(10):
            rows.append(
                _minimal_row(f"{cls}_{i}", risk=cls),
            )
    df = pd.DataFrame(rows)
    train, val, test = drs.stratified_train_val_test(
        df, label_col="risk_class", id_col="id", random_state=42
    )
    assert len(train) == 21
    assert len(val) == 4
    assert len(test) == 5
    drs.assert_split_ids_disjoint(train, val, test, id_col="id")
    for part in (train, val, test):
        vc = part["risk_class"].value_counts().sort_index()
        assert set(vc.index) == {0, 1, 2}


def test_unstratifiable_raises() -> None:
    df = pd.DataFrame(
        [
            _minimal_row("only0", risk=0),
            _minimal_row("only0b", risk=0),
            _minimal_row("one1", risk=1),
        ]
    )
    with pytest.raises(ValueError, match="at least 2 rows per class"):
        drs.stratified_train_val_test(df, random_state=42)


def test_split_summary_shape() -> None:
    train = pd.DataFrame({"risk_class": [0, 0, 1]})
    val = pd.DataFrame({"risk_class": [0, 1]})
    test = pd.DataFrame({"risk_class": [1, 2]})
    s = drs.split_summary(train, val, test, label_col="risk_class")
    assert s["train_rows"] == 3
    assert s["val_rows"] == 2
    assert s["test_rows"] == 2
    assert s["total_rows"] == 7
    assert "risk_class_counts" in s
    assert "train" in s["risk_class_counts"]


def test_assert_split_ids_disjoint_detects_overlap() -> None:
    train = pd.DataFrame({"id": ["a", "b"]})
    val = pd.DataFrame({"id": ["b"]})
    test = pd.DataFrame({"id": ["c"]})
    with pytest.raises(ValueError, match="Overlapping"):
        drs.assert_split_ids_disjoint(train, val, test, id_col="id")
