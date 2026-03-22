"""Unit tests for Pydantic API schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import JobPostInput, PredictRequest, RateInput


def test_rate_input_valid() -> None:
    r = RateInput(
        amount_min=100.0,
        amount_max=500.0,
        currency="PHP",
        type="daily",
    )
    assert r.amount_min == 100.0
    assert r.amount_max == 500.0


def test_job_post_input_with_full_rate() -> None:
    post = JobPostInput.model_validate(
        {
            "job_title": "Engineer",
            "job_desc": "Build things",
            "rate": {
                "amount_min": 100.0,
                "amount_max": 500.0,
                "currency": "PHP",
                "type": "daily",
            },
        }
    )
    assert post.rate is not None
    assert post.rate.currency == "PHP"
    assert post.rate.type == "daily"


def test_predict_request_deserializes_posts_with_rate() -> None:
    req = PredictRequest.model_validate(
        {
            "posts": [
                {
                    "job_title": "X",
                    "rate": {
                        "amount_min": 1.0,
                        "amount_max": 2.0,
                        "currency": "USD",
                        "type": "hourly",
                    },
                }
            ]
        }
    )
    assert len(req.posts) == 1
    assert req.posts[0].rate is not None
    assert req.posts[0].rate.type == "hourly"


def test_rate_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValidationError) as exc:
        RateInput(
            amount_min=500.0,
            amount_max=100.0,
            currency="PHP",
            type="daily",
        )
    assert "amount_min" in str(exc.value).lower()


def test_rate_rejects_bad_currency_length() -> None:
    with pytest.raises(ValidationError):
        RateInput(
            amount_min=1.0,
            amount_max=2.0,
            currency="PH",
            type="daily",
        )


def test_rate_rejects_lowercase_currency() -> None:
    with pytest.raises(ValidationError):
        RateInput(
            amount_min=1.0,
            amount_max=2.0,
            currency="php",
            type="daily",
        )


def test_rate_rejects_unknown_type_literal() -> None:
    with pytest.raises(ValidationError):
        RateInput.model_validate(
            {
                "amount_min": 1.0,
                "amount_max": 2.0,
                "currency": "USD",
                "type": "per_project",
            }
        )
