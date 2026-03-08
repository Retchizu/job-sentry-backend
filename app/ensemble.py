from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from app.config import settings


@runtime_checkable
class ScamPredictor(Protocol):
    """Any model that can produce a scam probability from text."""

    @property
    def name(self) -> str: ...

    def predict_proba(self, texts: List[str]) -> List[float]:
        """Return scam probability for each text."""
        ...


class DistilBertPredictor:
    """Adapts ScamDetectionModel to the ScamPredictor protocol."""

    name: str = "distilbert"

    def __init__(self, model):
        self._model = model

    def predict_proba(self, texts: List[str]) -> List[float]:
        results = self._model.predict_batch(texts)
        return [r["scam_probability"] for r in results]


def combine_soft_voting(probabilities_per_model: List[List[float]]) -> List[float]:
    """Average scam probabilities across models (soft voting).

    If only one model is present, returns its probabilities unchanged.
    All inner lists must have the same length.
    """
    if not probabilities_per_model:
        return []
    n_models = len(probabilities_per_model)
    if n_models == 1:
        return probabilities_per_model[0]
    n_items = len(probabilities_per_model[0])
    return [
        sum(probabilities_per_model[m][i] for m in range(n_models)) / n_models
        for i in range(n_items)
    ]


class EnsemblePredictor:
    """Combines multiple ScamPredictors via soft voting."""

    def __init__(self, predictors: List[ScamPredictor]):
        if not predictors:
            raise ValueError("EnsemblePredictor requires at least one predictor")
        self._predictors = predictors

    @property
    def predictor_names(self) -> List[str]:
        return [p.name for p in self._predictors]

    @property
    def is_hybrid(self) -> bool:
        return len(self._predictors) > 1

    def predict_proba(self, texts: List[str]) -> List[float]:
        all_probs = [p.predict_proba(texts) for p in self._predictors]
        return combine_soft_voting(all_probs)

    def predict(self, texts: List[str]) -> List[dict]:
        averaged = self.predict_proba(texts)
        results = []
        for scam_prob in averaged:
            is_scam = scam_prob >= settings.confidence_threshold
            results.append({
                "prediction": "scam" if is_scam else "legitimate",
                "confidence": round(scam_prob if is_scam else 1 - scam_prob, 4),
                "scam_probability": round(scam_prob, 4),
            })
        return results
