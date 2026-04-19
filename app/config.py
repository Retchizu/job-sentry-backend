"""Application settings (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JOBSENTRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    phase6_fused_dir: Optional[Path] = Field(
        default=None,
        description="Directory with phase6 fused artifacts (tokenizer, fused_meta.json, word_index.json, weights).",
    )
    phase6_fused_checkpoint: Optional[Path] = Field(
        default=None,
        description=(
            "Optional explicit .pt checkpoint; otherwise epoch_08.pt under checkpoints/ "
            "when present (best validation), else highest epoch_*.pt."
        ),
    )
    device: Optional[Literal["cpu", "cuda"]] = Field(
        default=None,
        description="Force device; default auto (CUDA if available).",
    )

    max_batch_size: int = Field(default=50, ge=1)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    warn_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Map P(scam) to warning when p >= this and p < fraud_threshold (unless fraud).",
    )
    fraud_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Map P(scam) to fraud when p >= this.",
    )
    p_fraud_decision_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Fused 3-class: predict fraud when P(fraud) >= this; else legit vs warning "
            "from the two non-fraud probabilities (POST /predict)."
        ),
    )

    @model_validator(mode="after")
    def _warn_below_fraud(self) -> Settings:
        if self.warn_threshold >= self.fraud_threshold:
            raise ValueError(
                "warn_threshold must be less than fraud_threshold "
                "(JOBSENTRY_WARN_THRESHOLD < JOBSENTRY_FRAUD_THRESHOLD)"
            )
        return self

    @field_validator("phase6_fused_dir", "phase6_fused_checkpoint", mode="before")
    @classmethod
    def _empty_str_path_to_none(cls, v: object) -> object:
        if v == "" or (isinstance(v, str) and not v.strip()):
            return None
        return v

    @field_validator("phase6_fused_dir", mode="after")
    @classmethod
    def _reject_unsubstituted_placeholder(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        normalized = str(v).replace("\\", "/")
        if "/path/to/" in normalized:
            raise ValueError(
                "JOBSENTRY_PHASE6_FUSED_DIR looks like an unsubstituted placeholder "
                "(contains '/path/to/'). Set it to a real directory, e.g. "
                "artifacts/models/phase6_fused from the repo root, or remove it to run "
                "without a model. Also check shell exports: env | grep JOBSENTRY"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
