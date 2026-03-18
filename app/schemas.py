from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings


def _load_sample_request() -> Dict[str, Any]:
    sample_path = get_settings().sample_request_path
    if sample_path.exists():
        return json.loads(sample_path.read_text(encoding="utf-8"))
    return {
        "company_id": "DEMO-001",
        "as_of_date": "2025-12-31",
        "features": {
            "Persistent EPS in the Last Four Seasons": 0.22,
            "Net Income to Total Assets": 0.11,
        },
    }


class PredictionRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=64, description="Client-side identifier for the company being scored.")
    as_of_date: date = Field(description="Financial reporting date for the submitted ratios.")
    features: Dict[str, float] = Field(description="Exact set of financial ratio features expected by the trained model.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": _load_sample_request(),
        }
    )

    @field_validator("company_id")
    @classmethod
    def normalize_company_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("company_id must not be blank")
        return value


class PredictionResponse(BaseModel):
    request_id: str
    company_id: str
    as_of_date: date
    bankruptcy_probability: float
    decision_threshold: float
    risk_label: Literal["HIGH_RISK", "LOW_RISK"]
    risk_band: Literal["LOW", "WATCHLIST", "ELEVATED", "CRITICAL"]
    model_version: str
    processing_time_ms: float
    input_feature_count: int
    timestamp_utc: datetime


class StatusResponse(BaseModel):
    status: Literal["healthy"]
    app_name: str
    app_version: str
    model_version: str
    trained_at_utc: datetime
    uptime_seconds: float
    expected_daily_requests: int
    total_requests_observed: int
    feature_count: int


class RuntimeMetricsResponse(BaseModel):
    uptime_seconds: float
    request_count: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    min_memory_mb: float
    max_memory_mb: float
    latest_memory_mb: float
    expected_daily_requests: int
    training_metrics: Dict[str, float]


class ModelMetadataResponse(BaseModel):
    model_version: str
    algorithm: str
    threshold: float
    feature_count: int
    feature_names: List[str]
    dataset_name: str
    dataset_url: str
    evaluation_split: Dict[str, int]
    test_metrics: Dict[str, float]


class InputSchemaResponse(BaseModel):
    feature_count: int
    feature_names: List[str]
    required_payload_shape: Dict[str, Any]
    notes: List[str]


class AdminReloadResponse(BaseModel):
    status: Literal["reloaded"]
    model_version: str
    reloaded_at_utc: datetime


class ErrorResponse(BaseModel):
    detail: str
    missing_features: Optional[List[str]] = None
    extra_features: Optional[List[str]] = None
