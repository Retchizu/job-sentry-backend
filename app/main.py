"""FastAPI entrypoint: phase6 fused scam detection."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.fused_predictor import FusedScamPredictor, resolve_device
from app.predict_warnings import compute_warnings
from app.schemas import HealthResponse, PredictRequest, PredictResponse, RootResponse

logger = logging.getLogger(__name__)

SERVICE_NAME = "job-sentry-backend"
SERVICE_VERSION = "0.3.0"


def _load_predictor(settings: Settings) -> Optional[FusedScamPredictor]:
    if not settings.phase6_fused_dir:
        logger.info(
            "JOBSENTRY_PHASE6_FUSED_DIR is not set — no fused model loaded. "
            "Set it to your artifacts directory (e.g. artifacts/models/phase6_fused)."
        )
        return None
    device = resolve_device(settings.device)
    return FusedScamPredictor.from_artifact_dir(
        settings.phase6_fused_dir,
        checkpoint_override=settings.phase6_fused_checkpoint,
        device=device,
        max_batch_size=settings.max_batch_size,
        p_fraud_threshold=settings.p_fraud_decision_threshold,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        app.state.predictor = _load_predictor(settings)
    except Exception as e:
        logger.exception("Fused model failed to load (JOBSENTRY_PHASE6_FUSED_DIR is set).")
        raise RuntimeError(
            "Failed to load phase6 fused model. Fix artifacts or unset JOBSENTRY_PHASE6_FUSED_DIR."
        ) from e
    yield


def root() -> RootResponse:
    return RootResponse(
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        docs="/docs",
    )


def health(request: Request) -> HealthResponse:
    settings = get_settings()
    predictor: Optional[FusedScamPredictor] = getattr(request.app.state, "predictor", None)
    device_str = "n/a"
    if predictor is not None:
        device_str = str(predictor.device)
    elif settings.device:
        device_str = settings.device
    else:
        import torch

        device_str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if predictor is not None:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            mode="phase6_fused",
            artifact_path=str(settings.phase6_fused_dir) if settings.phase6_fused_dir else None,
            device=device_str,
            message=None,
        )

    msg = (
        "No fused model configured. Set JOBSENTRY_PHASE6_FUSED_DIR to enable predictions."
    )
    return HealthResponse(
        status="degraded",
        model_loaded=False,
        mode="none",
        artifact_path=None,
        device=device_str,
        message=msg,
    )


def predict(request: Request, body: PredictRequest) -> PredictResponse:
    settings = get_settings()
    predictor: Optional[FusedScamPredictor] = getattr(request.app.state, "predictor", None)
    if predictor is None:
        logger.warning("predict_unavailable reason=no_model_loaded")
        raise HTTPException(
            status_code=503,
            detail="Fused model not loaded. Set JOBSENTRY_PHASE6_FUSED_DIR and restart.",
        )

    texts: list[str] = []
    for post in body.posts:
        try:
            texts.append(post.combined_text())
        except ValueError as e:
            logger.warning("predict_reject reason=empty_input")
            raise HTTPException(status_code=422, detail=str(e)) from e

    warnings = [compute_warnings(t) for t in texts]

    if len(texts) > settings.max_batch_size:
        logger.warning(
            "predict_reject reason=batch_too_large size=%d max=%d",
            len(texts),
            settings.max_batch_size,
        )
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(texts)} exceeds max_batch_size={settings.max_batch_size}.",
        )

    t0 = time.perf_counter()
    rows = predictor.predict_full(texts)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    predicted_class = [r.predicted_class for r in rows]
    predicted_label = [r.predicted_label for r in rows]
    legit_probability = [r.legit_probability for r in rows]
    warning_probability = [r.warning_probability for r in rows]
    fraud_probability = [r.fraud_probability for r in rows]
    confidence = [r.confidence for r in rows]

    logger.info(
        "predict_ok posts=%d latency_ms=%.2f",
        len(texts),
        latency_ms,
    )

    return PredictResponse(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        legit_probability=legit_probability,
        warning_probability=warning_probability,
        fraud_probability=fraud_probability,
        confidence=confidence,
        warnings=warnings,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title=SERVICE_NAME,
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_api_route("/", root, methods=["GET"], response_model=RootResponse)
    app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/predict", predict, methods=["POST"], response_model=PredictResponse)
    return app


app = create_app()
