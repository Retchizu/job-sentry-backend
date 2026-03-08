import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.config import settings
from app.model import ScamDetectionModel, load_model
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    JobPostInput,
    PredictResponse,
)
from app.services.prediction import predict_batch, predict_single

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model on startup...")
    try:
        app.state.model = load_model()
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model")
        app.state.model = None
    yield


app = FastAPI(
    title="Job Sentry API",
    description="Backend API for Job Sentry — scam job post detection",
    version="0.2.0",
    lifespan=lifespan,
)


def _get_model(request: Request) -> ScamDetectionModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/")
async def root():
    return {"message": "Job Sentry API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    model = request.app.state.model
    if model is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=settings.model_name,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(job_post: JobPostInput, request: Request):
    model = _get_model(request)
    return predict_single(job_post, model)


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(body: BatchPredictRequest, request: Request):
    model = _get_model(request)
    results = predict_batch(body.job_posts, model)
    return BatchPredictResponse(results=results, count=len(results))
