"""Pydantic models for the HTTP API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class RateInput(BaseModel):
    """Structured compensation rate (optional on `JobPostInput`); not used by `combined_text()` yet."""

    amount_min: float = Field(..., ge=0)
    amount_max: float = Field(..., ge=0)
    currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
        description="ISO 4217 alphabetic currency code in uppercase (e.g. PHP, USD).",
    )
    type: Literal["hourly", "daily", "weekly", "monthly", "yearly"]

    @model_validator(mode="after")
    def min_le_max(self) -> RateInput:
        if self.amount_min > self.amount_max:
            raise ValueError("amount_min must be <= amount_max")
        return self


class JobPostInput(BaseModel):
    """One job posting; either `text` or structured fields (merged like `combined_text` in training)."""

    text: Optional[str] = None
    job_title: Optional[str] = None
    job_desc: Optional[str] = None
    skills_desc: Optional[str] = None
    company_profile: Optional[str] = None
    rate: Optional[RateInput] = None

    def combined_text(self) -> str:
        if self.text is not None and str(self.text).strip():
            return str(self.text).strip()
        parts = [
            self.job_title or "",
            self.job_desc or "",
            self.skills_desc or "",
            self.company_profile or "",
        ]
        joined = " ".join(p.strip() for p in parts if p and str(p).strip())
        if not joined:
            raise ValueError(
                "Empty input: provide non-empty `text` or at least one structured field."
            )
        return joined


class PredictRequest(BaseModel):
    posts: list[JobPostInput] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    scam_probabilities: list[float]
    predicted_scam: list[bool]
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: str
    artifact_path: Optional[str] = None
    device: str
    message: Optional[str] = None


class RootResponse(BaseModel):
    service: str
    version: str
    docs: str
