from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

# Supported model paths (set JOBSENTRY_MODEL_ARTIFACT_PATH to switch):
#   distilbert_run: thesis-scam-job-post/models/distilbert_run/checkpoint-{783|1566|2349|3608|5412}
#                   Default: checkpoint-5412 (final training step)
#   phase32:        thesis-scam-job-post/models/phase32_distilbert
#   phase42:        thesis-scam-job-post/models/phase42_distilbert
#
# Hybrid model (optional):
#   When phase6_merged_path or bilstm_artifact_path is set, hybrid mode is enabled.
#   The app loads DistilBERT plus the specified components and combines predictions
#   via the method set in hybrid_combination (default: soft_voting).


class Settings(BaseSettings):
    model_artifact_path: Path = Path("thesis-scam-job-post/models/distilbert_run/checkpoint-5412")
    model_name: str = "distilbert-base-uncased"
    max_sequence_length: int = 512
    max_batch_size: int = 50
    confidence_threshold: float = 0.5

    phase6_merged_path: Optional[Path] = None
    bilstm_artifact_path: Optional[Path] = None
    hybrid_combination: str = "soft_voting"

    model_config = {"env_prefix": "JOBSENTRY_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
