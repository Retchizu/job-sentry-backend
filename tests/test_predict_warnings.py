"""Tests for heuristic predict warnings."""

from __future__ import annotations

from app.predict_warnings import compute_warnings


def test_empty_text() -> None:
    assert compute_warnings("") == []
    assert compute_warnings("   ") == []


def test_no_match_clean_posting() -> None:
    t = "Senior backend engineer. On-site interviews. Apply via company careers page."
    assert compute_warnings(t) == []


def test_multiple_flags() -> None:
    t = (
        "Urgent! Send $50 processing fee via WhatsApp today only. "
        "Guaranteed income, no interview needed."
    )
    out = compute_warnings(t)
    assert "upfront_payment" in out
    assert "off_platform_contact" in out
    assert "high_pressure" in out
    assert "guaranteed_income" in out
    assert out == sorted(out)


def test_stable_sorted_codes() -> None:
    t = "Pay upfront on telegram"
    assert compute_warnings(t) == ["off_platform_contact", "upfront_payment"]
