from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_artifact_path: Path = Path("thesis-scam-job-post/models/distilbert_run/checkpoint-1566")
    model_name: str = "distilbert-base-uncased"
    max_sequence_length: int = 512
    max_batch_size: int = 50
    confidence_threshold: float = 0.5

    model_config = {"env_prefix": "JOBSENTRY_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
