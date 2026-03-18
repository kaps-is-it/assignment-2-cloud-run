from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.model_store import ModelStore
from app.monitoring import RuntimeMetrics
from app.schemas import (
    AdminReloadResponse,
    ErrorResponse,
    InputSchemaResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
    RuntimeMetricsResponse,
    StatusResponse,
)


def _get_state(request: Request) -> tuple[Settings, ModelStore, RuntimeMetrics]:
    settings: Settings = request.app.state.settings
    model_store: ModelStore = request.app.state.model_store
    runtime_metrics: RuntimeMetrics = request.app.state.runtime_metrics
    return settings, model_store, runtime_metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.model_store = ModelStore(settings.model_path, settings.metadata_path)
    app.state.runtime_metrics = RuntimeMetrics(
        expected_daily_requests=settings.expected_daily_requests,
        history_size=settings.request_history_size,
    )
    yield


app = FastAPI(
    title="Bankruptcy Risk Modeling API",
    version=get_settings().app_version,
    description=(
        "Risk modeling API for company bankruptcy prediction. "
        "The service accepts normalized financial ratios, returns a bankruptcy probability, "
        "and exposes runtime monitoring endpoints for the assignment deliverable."
    ),
    lifespan=lifespan,
)


@app.exception_handler(FileNotFoundError)
async def model_not_ready_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.middleware("http")
async def capture_runtime_metrics(request: Request, call_next):
    start = time.perf_counter()
    runtime_metrics = request.app.state.runtime_metrics
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        runtime_metrics.observe(duration_ms=duration_ms, memory_mb=runtime_metrics.current_memory_mb())
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    runtime_metrics.observe(duration_ms=duration_ms, memory_mb=runtime_metrics.current_memory_mb())
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.get("/", tags=["root"])
async def root() -> dict[str, Any]:
    return {
        "service": "Bankruptcy Risk Modeling API",
        "docs_url": "/docs",
        "predict_url": "/api/v1/predict",
        "status_url": "/api/v1/status",
    }


@app.get("/api/v1/status", response_model=StatusResponse, tags=["monitoring"])
async def status(request: Request) -> StatusResponse:
    settings, model_store, runtime_metrics = _get_state(request)
    return StatusResponse(
        status="healthy",
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_version=model_store.model_version,
        trained_at_utc=datetime.fromisoformat(model_store.metadata["trained_at_utc"]),
        uptime_seconds=runtime_metrics.uptime_seconds(),
        expected_daily_requests=settings.expected_daily_requests,
        total_requests_observed=runtime_metrics.request_count(),
        feature_count=len(model_store.feature_names),
    )


@app.get("/api/v1/metrics", response_model=RuntimeMetricsResponse, tags=["monitoring"])
async def metrics(request: Request) -> RuntimeMetricsResponse:
    settings, model_store, runtime_metrics = _get_state(request)
    snapshot = runtime_metrics.snapshot()
    return RuntimeMetricsResponse(
        uptime_seconds=snapshot["uptime_seconds"],
        request_count=int(snapshot["request_count"]),
        avg_response_time_ms=snapshot["avg_response_time_ms"],
        p95_response_time_ms=snapshot["p95_response_time_ms"],
        min_response_time_ms=snapshot["min_response_time_ms"],
        max_response_time_ms=snapshot["max_response_time_ms"],
        min_memory_mb=snapshot["min_memory_mb"],
        max_memory_mb=snapshot["max_memory_mb"],
        latest_memory_mb=snapshot["latest_memory_mb"],
        expected_daily_requests=settings.expected_daily_requests,
        training_metrics={key: float(value) for key, value in model_store.metadata["test_metrics"].items()},
    )


@app.get("/api/v1/schema/input", response_model=InputSchemaResponse, tags=["schema"])
async def input_schema(request: Request) -> InputSchemaResponse:
    _, model_store, _ = _get_state(request)
    return InputSchemaResponse(
        feature_count=len(model_store.feature_names),
        feature_names=model_store.feature_names,
        required_payload_shape={
            "company_id": "string",
            "as_of_date": "YYYY-MM-DD",
            "features": {feature_name: "float" for feature_name in model_store.feature_names[:5]},
        },
        notes=[
            "Submit all required feature keys exactly as listed in feature_names.",
            "The model expects the same normalized financial ratios used during training.",
            "Any missing or extra feature will be rejected with HTTP 422.",
        ],
    )


@app.get("/api/v1/admin/model", response_model=ModelMetadataResponse, tags=["admin"])
async def admin_model_metadata(request: Request) -> ModelMetadataResponse:
    _, model_store, _ = _get_state(request)
    return ModelMetadataResponse(**model_store.admin_metadata())


@app.post(
    "/api/v1/admin/reload",
    response_model=AdminReloadResponse,
    responses={403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["admin"],
)
async def admin_reload_model(
    request: Request,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> AdminReloadResponse:
    settings, model_store, _ = _get_state(request)
    if not settings.admin_token:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not configured for this deployment.")
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid X-Admin-Token.")

    model_store.reload()
    return AdminReloadResponse(
        status="reloaded",
        model_version=model_store.model_version,
        reloaded_at_utc=datetime.now(timezone.utc),
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["prediction"],
)
async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    _, model_store, _ = _get_state(request)
    started_at = time.perf_counter()
    missing_features, extra_features = model_store.validate_features(payload.features)
    if missing_features or extra_features:
        raise HTTPException(
            status_code=422,
            detail={
                "detail": "The submitted feature set does not match the trained model contract.",
                "missing_features": missing_features,
                "extra_features": extra_features,
            },
        )

    probability = model_store.predict_probability(payload.features)
    processing_time_ms = (time.perf_counter() - started_at) * 1000
    return PredictionResponse(
        request_id=str(uuid.uuid4()),
        company_id=payload.company_id,
        as_of_date=payload.as_of_date,
        bankruptcy_probability=round(probability, 6),
        decision_threshold=model_store.threshold,
        risk_label=model_store.risk_label(probability),
        risk_band=model_store.risk_band(probability),
        model_version=model_store.model_version,
        processing_time_ms=round(processing_time_ms, 3),
        input_feature_count=len(payload.features),
        timestamp_utc=datetime.now(timezone.utc),
    )
