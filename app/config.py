from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    model_path: Path
    metadata_path: Path
    sample_request_path: Path
    request_history_size: int
    expected_daily_requests: int
    admin_token: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "artifacts"
    return Settings(
        app_name="Bankruptcy Risk Modeling API",
        app_version="1.0.0",
        model_path=Path(os.getenv("MODEL_PATH", artifacts_dir / "bankruptcy_risk_model.joblib")),
        metadata_path=Path(os.getenv("MODEL_METADATA_PATH", artifacts_dir / "model_metadata.json")),
        sample_request_path=Path(os.getenv("SAMPLE_REQUEST_PATH", artifacts_dir / "sample_request.json")),
        request_history_size=int(os.getenv("REQUEST_HISTORY_SIZE", "5000")),
        expected_daily_requests=int(os.getenv("EXPECTED_DAILY_REQUESTS", "2000")),
        admin_token=os.getenv("ADMIN_TOKEN", ""),
    )
