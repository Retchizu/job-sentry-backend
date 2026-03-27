"""HTTP API tests (TestClient)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


def test_root() -> None:
    with TestClient(create_app()) as client:
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "job-sentry-backend"
        assert "version" in data


def test_health_without_fused_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # Override .env: empty string is coerced to None in Settings (see app.config).
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is False
        assert body["mode"] == "none"
        assert body["status"] == "degraded"


def test_predict_503_when_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        r = client.post("/predict", json={"posts": [{"text": "hello world"}]})
        assert r.status_code == 503


def test_predict_with_injected_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5}

        def predict_proba(self, texts: list[str]) -> list[float]:
            return [0.9] * len(texts)

    with TestClient(app) as client:
        app.state.predictor = _FakePred()
        r = client.post(
            "/predict",
            json={
                "posts": [
                    {"job_title": "Engineer", "job_desc": "Do things"},
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["scam_probabilities"] == [0.9]
        assert body["predicted_scam"] == [True]
        assert body["threshold"] == 0.5
        assert body["warnings"] == [[]]


def test_predict_with_injected_predictor_and_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5}

        def predict_proba(self, texts: list[str]) -> list[float]:
            return [0.9] * len(texts)

    with TestClient(app) as client:
        app.state.predictor = _FakePred()
        r = client.post(
            "/predict",
            json={
                "posts": [
                    {
                        "job_title": "Engineer",
                        "job_desc": "Do things",
                        "rate": {
                            "amount_min": 100.0,
                            "amount_max": 500.0,
                            "currency": "PHP",
                            "type": "daily",
                        },
                    },
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["scam_probabilities"] == [0.9]
        assert body["predicted_scam"] == [True]
        assert body["threshold"] == 0.5
        assert body["warnings"] == [[]]


def test_predict_returns_warnings_for_scammy_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5}

        def predict_proba(self, texts: list[str]) -> list[float]:
            return [0.5] * len(texts)

    with TestClient(app) as client:
        app.state.predictor = _FakePred()
        r = client.post(
            "/predict",
            json={
                "posts": [
                    {
                        "text": "Urgent! Pay $50 fee via WhatsApp. Guaranteed income today only.",
                    },
                ]
            },
        )
        assert r.status_code == 200
        w = r.json()["warnings"]
        assert len(w) == 1
        assert set(w[0]) >= {
            "upfront_payment",
            "off_platform_contact",
            "high_pressure",
            "guaranteed_income",
        }


def test_predict_422_when_rate_min_exceeds_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5}

        def predict_proba(self, texts: list[str]) -> list[float]:
            return [0.9] * len(texts)

    with TestClient(app) as client:
        app.state.predictor = _FakePred()
        r = client.post(
            "/predict",
            json={
                "posts": [
                    {
                        "text": "Legit posting",
                        "rate": {
                            "amount_min": 500.0,
                            "amount_max": 100.0,
                            "currency": "USD",
                            "type": "hourly",
                        },
                    },
                ]
            },
        )
        assert r.status_code == 422


@pytest.mark.skipif(
    not os.path.isdir(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "artifacts",
            "models",
            "phase6_fused",
        )
    ),
    reason="Local phase6_fused artifacts not present",
)
def test_health_with_real_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fused = os.path.join(root, "artifacts", "models", "phase6_fused")
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", fused)
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True
