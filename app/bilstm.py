from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTH = 200


class BiLSTMPredictor:
    """Wraps a Keras Bi-LSTM model as a ScamPredictor.

    Expects a tokenizer word_index JSON alongside the model so that raw text
    can be converted to padded integer sequences before inference.
    """

    name: str = "bilstm"

    def __init__(self, model, word_index: dict, max_len: int = MAX_SEQUENCE_LENGTH):
        self._model = model
        self._word_index = word_index
        self._max_len = max_len

    def _tokenize(self, texts: List[str]) -> np.ndarray:
        from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore[import]

        sequences = []
        for text in texts:
            tokens = text.lower().split()
            seq = [self._word_index.get(w, 0) for w in tokens]
            sequences.append(seq)
        return pad_sequences(sequences, maxlen=self._max_len, padding="post", truncating="post")

    def predict_proba(self, texts: List[str]) -> List[float]:
        padded = self._tokenize(texts)
        preds = self._model.predict(padded, verbose=0)
        if preds.ndim == 2 and preds.shape[1] == 1:
            return [float(p[0]) for p in preds]
        if preds.ndim == 2 and preds.shape[1] == 2:
            return [float(p[1]) for p in preds]
        return [float(p) for p in preds.flatten()]


def load_bilstm(path: Path) -> Optional[BiLSTMPredictor]:
    """Load a Keras Bi-LSTM model and its tokenizer word_index.

    Looks for a tokenizer word_index JSON in the same directory (or parent)
    named ``word_index.json`` or ``tokenizer.json``.
    Returns None if model or tokenizer can't be loaded.
    """
    if not path.exists():
        logger.warning("Bi-LSTM artifact not found at '%s'; skipping", path)
        return None
    try:
        from tensorflow import keras  # type: ignore[import]

        model = keras.models.load_model(path)
        logger.info("Loaded Bi-LSTM model from '%s'", path)
    except Exception:
        logger.exception("Failed to load Bi-LSTM model from '%s'; skipping", path)
        return None

    word_index = _find_word_index(path)
    if word_index is None:
        logger.warning(
            "No tokenizer word_index found near '%s'; "
            "Bi-LSTM will use empty vocab (predictions will be meaningless)",
            path,
        )
        word_index = {}

    return BiLSTMPredictor(model, word_index)


def _find_word_index(model_path: Path) -> Optional[dict]:
    """Search for a word_index JSON or joblib tokenizer near the model path."""
    parent = model_path.parent if model_path.is_file() else model_path
    stem = model_path.stem if model_path.is_file() else model_path.name

    # JSON candidates
    for name in ("word_index.json", "tokenizer.json"):
        candidate = parent / name
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text())
                if isinstance(data, dict) and "word_index" in data:
                    return data["word_index"]
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.warning("Failed to parse tokenizer file '%s'", candidate)

    # Thesis-style: phase42_bilstm.keras -> phase42_bilstm_tokenizer.joblib
    joblib_name = f"{stem}_tokenizer.joblib"
    joblib_path = parent / joblib_name
    if joblib_path.exists():
        try:
            import joblib
            tokenizer = joblib.load(joblib_path)
            if hasattr(tokenizer, "word_index"):
                return tokenizer.word_index
            if isinstance(tokenizer, dict):
                return tokenizer
        except Exception:
            logger.warning("Failed to load tokenizer from '%s'", joblib_path)
    return None
