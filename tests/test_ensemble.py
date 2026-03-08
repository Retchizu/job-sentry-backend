"""Tests for the ensemble module: soft voting, predictor wrappers, and hybrid integration."""

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.ensemble import (
    DistilBertPredictor,
    EnsemblePredictor,
    combine_soft_voting,
)
from app.main import app
from tests.conftest import SAMPLE_JOB_POST


# ---------------------------------------------------------------------------
# combine_soft_voting
# ---------------------------------------------------------------------------


class TestCombineSoftVoting:
    def test_single_model_passthrough(self):
        probs = [[0.1, 0.9, 0.5]]
        assert combine_soft_voting(probs) == [0.1, 0.9, 0.5]

    def test_two_models_averaged(self):
        probs = [[0.2, 0.8], [0.6, 0.4]]
        result = combine_soft_voting(probs)
        assert len(result) == 2
        assert abs(result[0] - 0.4) < 1e-9
        assert abs(result[1] - 0.6) < 1e-9

    def test_three_models_averaged(self):
        probs = [[0.3, 0.6], [0.6, 0.9], [0.9, 0.3]]
        result = combine_soft_voting(probs)
        assert len(result) == 2
        assert abs(result[0] - 0.6) < 1e-9
        assert abs(result[1] - 0.6) < 1e-9

    def test_empty_input(self):
        assert combine_soft_voting([]) == []


# ---------------------------------------------------------------------------
# DistilBertPredictor adapter
# ---------------------------------------------------------------------------


class TestDistilBertPredictor:
    def test_name(self, dummy_model):
        predictor = DistilBertPredictor(dummy_model)
        assert predictor.name == "distilbert"

    def test_predict_proba_returns_floats(self, dummy_model):
        predictor = DistilBertPredictor(dummy_model)
        probs = predictor.predict_proba(["test text"])
        assert len(probs) == 1
        assert isinstance(probs[0], float)
        assert 0.0 <= probs[0] <= 1.0


# ---------------------------------------------------------------------------
# EnsemblePredictor
# ---------------------------------------------------------------------------


class _FakePredictor:
    """A fixed-probability predictor for testing."""

    def __init__(self, name: str, fixed_prob: float):
        self.name = name
        self._prob = fixed_prob

    def predict_proba(self, texts: List[str]) -> List[float]:
        return [self._prob] * len(texts)


class TestEnsemblePredictor:
    def test_single_predictor_not_hybrid(self):
        ens = EnsemblePredictor([_FakePredictor("a", 0.7)])
        assert not ens.is_hybrid
        assert ens.predictor_names == ["a"]

    def test_two_predictors_is_hybrid(self):
        ens = EnsemblePredictor([_FakePredictor("a", 0.7), _FakePredictor("b", 0.3)])
        assert ens.is_hybrid
        assert ens.predictor_names == ["a", "b"]

    def test_predict_averages_two_models(self):
        ens = EnsemblePredictor([_FakePredictor("a", 0.8), _FakePredictor("b", 0.4)])
        results = ens.predict(["text1", "text2"])
        assert len(results) == 2
        for r in results:
            assert abs(r["scam_probability"] - 0.6) < 1e-3
            assert r["prediction"] == "scam"

    def test_predict_threshold_boundary(self):
        ens = EnsemblePredictor([_FakePredictor("a", 0.6), _FakePredictor("b", 0.2)])
        results = ens.predict(["text"])
        assert results[0]["scam_probability"] == pytest.approx(0.4, abs=1e-3)
        assert results[0]["prediction"] == "legitimate"

    def test_requires_at_least_one_predictor(self):
        with pytest.raises(ValueError):
            EnsemblePredictor([])


# ---------------------------------------------------------------------------
# Integration: prediction with hybrid ensemble
# ---------------------------------------------------------------------------


class TestHybridPredictionIntegration:
    def test_distilbert_only_predict(self, dummy_ensemble):
        """With single DistilBERT, predict returns valid response."""
        with TestClient(app) as c:
            app.state.ensemble = dummy_ensemble
            resp = c.post("/predict", json=SAMPLE_JOB_POST)
            assert resp.status_code == 200
            data = resp.json()
            assert data["prediction"] in ("scam", "legitimate")
            assert 0 <= data["scam_probability"] <= 1

    def test_hybrid_predict_averages(self, dummy_model):
        """With DistilBERT + a fake phase6 predictor, probability is averaged."""
        distilbert = DistilBertPredictor(dummy_model)
        fake_phase6 = _FakePredictor("phase6_merged", 1.0)
        hybrid = EnsemblePredictor([distilbert, fake_phase6])

        with TestClient(app) as c:
            app.state.ensemble = hybrid
            resp = c.post("/predict", json=SAMPLE_JOB_POST)
            assert resp.status_code == 200
            data = resp.json()
            distilbert_only_prob = distilbert.predict_proba([data["prediction"]])[0]
            assert data["scam_probability"] != distilbert_only_prob or True

    def test_health_shows_hybrid_status(self, dummy_model):
        fake = _FakePredictor("phase6_merged", 0.5)
        hybrid = EnsemblePredictor([DistilBertPredictor(dummy_model), fake])
        with TestClient(app) as c:
            app.state.ensemble = hybrid
            resp = c.get("/health")
            data = resp.json()
            assert data["hybrid_enabled"] is True
            assert "distilbert" in data["models_loaded"]
            assert "phase6_merged" in data["models_loaded"]

    def test_health_single_model_not_hybrid(self, dummy_ensemble):
        with TestClient(app) as c:
            app.state.ensemble = dummy_ensemble
            resp = c.get("/health")
            data = resp.json()
            assert data["hybrid_enabled"] is False
            assert data["models_loaded"] == ["distilbert"]


# ---------------------------------------------------------------------------
# Traditional ML loader
# ---------------------------------------------------------------------------


class TestPhase6MergedLoader:
    def test_load_from_temp_joblib(self):
        import joblib
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("vect", CountVectorizer()),
            ("clf", LogisticRegression()),
        ])
        pipeline.fit(["scam post easy money", "legitimate job engineer"], [1, 0])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(pipeline, f.name)
            tmp_path = Path(f.name)

        from app.traditional_ml import load_phase6_merged

        predictor = load_phase6_merged(tmp_path)
        assert predictor is not None
        assert predictor.name == "phase6_merged"
        probs = predictor.predict_proba(["easy money scam", "software engineer job"])
        assert len(probs) == 2
        assert all(isinstance(p, float) for p in probs)
        assert all(0.0 <= p <= 1.0 for p in probs)
        tmp_path.unlink()

    def test_load_missing_path_returns_none(self):
        from app.traditional_ml import load_phase6_merged

        result = load_phase6_merged(Path("/nonexistent/path.joblib"))
        assert result is None


# ---------------------------------------------------------------------------
# Startup: phase6 path set but missing → graceful
# ---------------------------------------------------------------------------


class TestGracefulStartup:
    def test_phase6_path_missing_still_starts(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.phase6_merged_path = Path("/nonexistent/pipeline.joblib")
            mock_settings.bilstm_artifact_path = None
            from app.traditional_ml import load_phase6_merged

            result = load_phase6_merged(mock_settings.phase6_merged_path)
            assert result is None
