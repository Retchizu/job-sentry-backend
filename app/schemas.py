from typing import List, Optional

from pydantic import BaseModel, Field


class JobPostInput(BaseModel):
    job_title: str = Field(..., min_length=1, max_length=500)
    job_desc: str = Field(..., min_length=1, max_length=50_000)
    company_profile: Optional[str] = Field(None, max_length=10_000)
    skills_desc: Optional[str] = Field(None, max_length=10_000)
    salary_range: Optional[str] = Field(None, max_length=200)
    employment_type: Optional[str] = Field(None, max_length=100)


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="'scam' or 'legitimate'")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the predicted class"
    )
    scam_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Raw probability of being a scam"
    )
    warning_signals: List[str] = Field(
        default_factory=list,
        description="List of detected warning signals",
    )


class BatchPredictRequest(BaseModel):
    job_posts: List[JobPostInput] = Field(
        ..., min_length=1, max_length=50, description="List of job posts to analyze"
    )


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    hybrid_enabled: Optional[bool] = None
    models_loaded: Optional[List[str]] = None
