from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class Phase6MergedPredictor:
    """Wraps a joblib pipeline (or tfidf + classifier) as a ScamPredictor.

    Supports either a single pipeline (predict_proba on raw text) or a directory
    with thesis-style artifacts: tfidf_merged.joblib + merged_lr/rf/xgb.joblib.
    """

    name: str = "phase6_merged"

    def __init__(self, pipeline=None, vectorizer=None, classifier=None):
        if pipeline is not None:
            self._pipeline = pipeline
            self._vectorizer = None
            self._classifier = None
        else:
            self._pipeline = None
            self._vectorizer = vectorizer
            self._classifier = classifier

    def predict_proba(self, texts: List[str]) -> List[float]:
        if self._pipeline is not None:
            return self._predict_pipeline(texts)
        return self._predict_vectorizer_classifier(texts)

    def _predict_pipeline(self, texts: List[str]) -> List[float]:
        if hasattr(self._pipeline, "predict_proba"):
            probs = self._pipeline.predict_proba(texts)
            return [float(row[1]) for row in probs]
        preds = self._pipeline.predict(texts)
        return [float(p) for p in preds]

    def _predict_vectorizer_classifier(self, texts: List[str]) -> List[float]:
        import scipy.sparse as sp

        X = self._vectorizer.transform(texts)
        expected = getattr(self._classifier, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            n_in = X.shape[1]
            if n_in < expected:
                pad = expected - n_in
                zeros = sp.csr_matrix((X.shape[0], pad))
                X = sp.hstack([X, zeros])
                logger.warning(
                    "Phase6 feature mismatch: vectorizer outputs %d features, classifier expects %d; padded with zeros. Consider using a matching artifact.",
                    n_in,
                    expected,
                )
            else:
                raise ValueError(
                    "Phase6 artifact mismatch: vectorizer outputs %d features but classifier expects %d. Use a matching artifact or disable phase6 (unset JOBSENTRY_PHASE6_MERGED_PATH)."
                    % (n_in, expected)
                )
        if hasattr(self._classifier, "predict_proba"):
            probs = self._classifier.predict_proba(X)
            return [float(row[1]) for row in probs]
        preds = self._classifier.predict(X)
        return [float(p) for p in preds]


def load_phase6_merged(path: Path) -> Optional[Phase6MergedPredictor]:
    """Load phase6_merged from a joblib file or a directory of thesis artifacts.

    - File: single joblib pipeline (e.g. TfidfVectorizer + classifier).
    - Directory: expects tfidf_merged.joblib and one of merged_lr.joblib,
      merged_rf.joblib, merged_xgb.joblib. Note: if the thesis trained with
      TF-IDF + numeric features, the classifier may expect more features than
      TF-IDF alone provides; then predictions will raise or you need to add
      numeric feature construction.
    Returns None if the path doesn't exist or loading fails.
    """
    if not path.exists():
        logger.warning("phase6_merged artifact not found at '%s'; skipping", path)
        return None
    try:
        import joblib

        if path.is_file():
            pipeline = joblib.load(path)
            logger.info("Loaded phase6_merged pipeline from '%s'", path)
            return Phase6MergedPredictor(pipeline=pipeline)
        # Directory: thesis-style tfidf + classifier
        dir_path = path
        tfidf_path = dir_path / "tfidf_merged.joblib"
        for clf_name in ("merged_lr.joblib", "merged_rf.joblib", "merged_xgb.joblib"):
            clf_path = dir_path / clf_name
            if tfidf_path.exists() and clf_path.exists():
                vectorizer = joblib.load(tfidf_path)
                classifier = joblib.load(clf_path)
                logger.info(
                    "Loaded phase6_merged (tfidf + %s) from '%s'",
                    clf_name,
                    dir_path,
                )
                return Phase6MergedPredictor(
                    vectorizer=vectorizer,
                    classifier=classifier,
                )
        logger.warning(
            "phase6_merged directory '%s' missing tfidf_merged.joblib or merged_*.joblib; skipping",
            dir_path,
        )
        return None
    except Exception:
        logger.exception("Failed to load phase6_merged from '%s'; skipping", path)
        return None
