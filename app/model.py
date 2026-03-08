import logging
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from app.config import settings

logger = logging.getLogger(__name__)


class ScamDetectionModel:
    def __init__(
        self,
        model: DistilBertForSequenceClassification,
        tokenizer: DistilBertTokenizerFast,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text: str) -> Dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        results = []
        for probs in probabilities:
            scam_prob = probs[1].item()
            is_scam = scam_prob >= settings.confidence_threshold
            results.append({
                "prediction": "scam" if is_scam else "legitimate",
                "confidence": round(scam_prob if is_scam else 1 - scam_prob, 4),
                "scam_probability": round(scam_prob, 4),
            })
        return results


def load_model() -> ScamDetectionModel:
    artifact_path = Path(settings.model_artifact_path)
    model_name = settings.model_name

    logger.info("Loading tokenizer from '%s'", model_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    safetensors_file = artifact_path / "model.safetensors"
    if safetensors_file.exists():
        logger.info("Loading fine-tuned model from '%s'", artifact_path)
        config = DistilBertConfig.from_pretrained(model_name, num_labels=2)
        model = DistilBertForSequenceClassification(config)
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_file))
        # Remap old-style TF LayerNorm keys (gamma/beta → weight/bias)
        remapped = {}
        for k, v in state_dict.items():
            new_key = k.replace(".gamma", ".weight").replace(".beta", ".bias")
            remapped[new_key] = v
        model.load_state_dict(remapped)
    else:
        logger.warning(
            "No model.safetensors found at '%s', loading base model (untrained). "
            "Predictions will not be meaningful.",
            artifact_path,
        )
        config = DistilBertConfig.from_pretrained(model_name, num_labels=2)
        model = DistilBertForSequenceClassification(config)

    model.to(device)
    model.eval()
    logger.info("Model loaded and ready for inference")

    return ScamDetectionModel(model=model, tokenizer=tokenizer, device=device)
