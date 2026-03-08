import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.bilstm import load_bilstm
from app.config import settings
from app.ensemble import DistilBertPredictor, EnsemblePredictor
from app.model import load_model
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    JobPostInput,
    PredictResponse,
)
from app.services.prediction import predict_batch, predict_single
from app.traditional_ml import load_phase6_merged

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model on startup...")
    try:
        distilbert_model = load_model()
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model")
        distilbert_model = None

    phase6_model = None
    if settings.phase6_merged_path:
        phase6_model = load_phase6_merged(settings.phase6_merged_path)

    bilstm_model = None
    if settings.bilstm_artifact_path:
        bilstm_model = load_bilstm(settings.bilstm_artifact_path)

    if distilbert_model is not None:
        predictors = [DistilBertPredictor(distilbert_model)]
        if phase6_model is not None:
            predictors.append(phase6_model)
        if bilstm_model is not None:
            predictors.append(bilstm_model)
        app.state.ensemble = EnsemblePredictor(predictors)
        logger.info(
            "Ensemble ready: %s (hybrid=%s)",
            app.state.ensemble.predictor_names,
            app.state.ensemble.is_hybrid,
        )
    else:
        app.state.ensemble = None

    yield


app = FastAPI(
    title="Job Sentry API",
    description="Backend API for Job Sentry — scam job post detection",
    version="0.2.0",
    lifespan=lifespan,
)


def _get_ensemble(request: Request) -> EnsemblePredictor:
    ensemble = request.app.state.ensemble
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ensemble


@app.get("/")
async def root():
    return {"message": "Job Sentry API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    ensemble = request.app.state.ensemble
    if ensemble is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=settings.model_name,
        hybrid_enabled=ensemble.is_hybrid,
        models_loaded=ensemble.predictor_names,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(job_post: JobPostInput, request: Request):
    ensemble = _get_ensemble(request)
    return predict_single(job_post, ensemble)


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(body: BatchPredictRequest, request: Request):
    ensemble = _get_ensemble(request)
    results = predict_batch(body.job_posts, ensemble)
    return BatchPredictResponse(results=results, count=len(results))
