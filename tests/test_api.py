"""HTTP API tests (TestClient)."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.fused_predictor import risk_predictions_from_softmax_triples
from app.main import create_app


def test_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
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
        fused_meta = {"threshold": 0.5, "num_labels": 3}

        def predict_risk_distribution(self, texts: list[str]) -> list[tuple[float, float, float]]:
            return [(0.05, 0.05, 0.9)] * len(texts)

        def predict_full(self, texts: list[str]):
            return risk_predictions_from_softmax_triples(self.predict_risk_distribution(texts))

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
        assert body["predicted_class"] == [2]
        assert body["predicted_label"] == ["fraud"]
        assert body["legit_probability"] == [pytest.approx(0.05)]
        assert body["warning_probability"] == [pytest.approx(0.05)]
        assert body["fraud_probability"] == [pytest.approx(0.9)]
        assert body["confidence"] == [pytest.approx(0.9)]
        assert body["warnings"] == [[]]


def test_predict_with_injected_predictor_and_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5, "num_labels": 3}

        def predict_risk_distribution(self, texts: list[str]) -> list[tuple[float, float, float]]:
            return [(0.05, 0.05, 0.9)] * len(texts)

        def predict_full(self, texts: list[str]):
            return risk_predictions_from_softmax_triples(self.predict_risk_distribution(texts))

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
        assert body["predicted_class"] == [2]
        assert body["predicted_label"] == ["fraud"]
        assert body["fraud_probability"] == [pytest.approx(0.9)]
        assert body["warnings"] == [[]]


def test_predict_warning_mid_triple_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5, "num_labels": 3}

        def predict_risk_distribution(self, texts: list[str]) -> list[tuple[float, float, float]]:
            return [(0.2, 0.5, 0.3)] * len(texts)

        def predict_full(self, texts: list[str]):
            return risk_predictions_from_softmax_triples(self.predict_risk_distribution(texts))

    with TestClient(app) as client:
        app.state.predictor = _FakePred()
        r = client.post(
            "/predict",
            json={"posts": [{"text": "Neutral posting text."}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_class"] == [1]
        assert body["predicted_label"] == ["warning"]
        assert body["warning_probability"] == [pytest.approx(0.5)]
        assert body["confidence"] == [pytest.approx(0.5)]


def test_predict_legit_with_injected_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5, "num_labels": 3}

        def predict_risk_distribution(self, texts: list[str]) -> list[tuple[float, float, float]]:
            return [(0.8, 0.1, 0.1)] * len(texts)

        def predict_full(self, texts: list[str]):
            return risk_predictions_from_softmax_triples(self.predict_risk_distribution(texts))

    with TestClient(app) as client:
        app.state.predictor = _FakePred()
        r = client.post(
            "/predict",
            json={"posts": [{"text": "Senior engineer. Standard hiring process."}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_class"] == [0]
        assert body["predicted_label"] == ["legit"]
        assert body["legit_probability"] == [pytest.approx(0.8)]
        assert body["confidence"] == [pytest.approx(0.8)]


def test_predict_returns_warnings_heuristics_do_not_override_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model argmax picks class; heuristic codes are returned for display only."""
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5, "num_labels": 3}

        def predict_risk_distribution(self, texts: list[str]) -> list[tuple[float, float, float]]:
            return [(0.4, 0.35, 0.25)] * len(texts)

        def predict_full(self, texts: list[str]):
            return risk_predictions_from_softmax_triples(self.predict_risk_distribution(texts))

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
        body = r.json()
        w = body["warnings"]
        assert len(w) == 1
        assert set(w[0]) >= {
            "upfront_payment",
            "off_platform_contact",
            "high_pressure",
            "guaranteed_income",
        }
        assert body["predicted_class"] == [0]
        assert body["predicted_label"] == ["legit"]


def test_predict_422_when_rate_min_exceeds_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", "")
    get_settings.cache_clear()
    app = create_app()

    class _FakePred:
        fused_meta = {"threshold": 0.5, "num_labels": 3}

        def predict_risk_distribution(self, texts: list[str]) -> list[tuple[float, float, float]]:
            return [(0.05, 0.05, 0.9)] * len(texts)

        def predict_full(self, texts: list[str]):
            return risk_predictions_from_softmax_triples(self.predict_risk_distribution(texts))

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


def _sequential_fused_bundle_ready(repo_root: str) -> tuple[bool, str]:
    meta_path = os.path.join(repo_root, "artifacts", "models", "phase6_fused", "fused_meta.json")
    if not os.path.isfile(meta_path):
        return False, "no fused_meta.json under artifacts/models/phase6_fused"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta.get("num_labels", 0)) != 3:
        return False, "bundle is not sequential 3-class (num_labels!=3)"
    return True, ""


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SEQ_OK, _SEQ_REASON = _sequential_fused_bundle_ready(_REPO_ROOT)


@pytest.mark.skipif(
    not _SEQ_OK,
    reason=_SEQ_REASON or "sequential 3-class fused artifacts not available",
)
def test_health_with_real_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    fused = os.path.join(_REPO_ROOT, "artifacts", "models", "phase6_fused")
    monkeypatch.setenv("JOBSENTRY_PHASE6_FUSED_DIR", fused)
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True
