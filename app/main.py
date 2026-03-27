"""FastAPI entrypoint: phase6 fused scam detection."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.fused_predictor import FusedScamPredictor, resolve_device
from app.schemas import HealthResponse, PredictRequest, PredictResponse, RootResponse

logger = logging.getLogger(__name__)

SERVICE_NAME = "job-sentry-backend"
SERVICE_VERSION = "0.2.0"


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
        raise HTTPException(
            status_code=503,
            detail="Fused model not loaded. Set JOBSENTRY_PHASE6_FUSED_DIR and restart.",
        )

    texts: list[str] = []
    for post in body.posts:
        try:
            texts.append(post.combined_text())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    if len(texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(texts)} exceeds max_batch_size={settings.max_batch_size}.",
        )

    probs = predictor.predict_proba(texts)
    thr = float(predictor.fused_meta.get("threshold", settings.confidence_threshold))
    labels = [p >= thr for p in probs]
    return PredictResponse(
        scam_probabilities=probs,
        predicted_scam=labels,
        threshold=thr,
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
