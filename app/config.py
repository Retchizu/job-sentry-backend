"""Application settings (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
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
        description="Optional explicit .pt checkpoint; otherwise highest epoch_*.pt under checkpoints/.",
    )
    device: Optional[Literal["cpu", "cuda"]] = Field(
        default=None,
        description="Force device; default auto (CUDA if available).",
    )

    max_batch_size: int = Field(default=50, ge=1)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("phase6_fused_dir", "phase6_fused_checkpoint", mode="before")
    @classmethod
    def _empty_str_path_to_none(cls, v: object) -> object:
        if v == "" or (isinstance(v, str) and not v.strip()):
            return None
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
