from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

SAMPLE_REQUEST_PATH = PROJECT_ROOT / "artifacts" / "sample_request.json"


def load_sample_payload() -> dict:
    return json.loads(SAMPLE_REQUEST_PATH.read_text(encoding="utf-8"))


def test_status_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["feature_count"] == 95


def test_predict_endpoint() -> None:
    payload = load_sample_payload()
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["company_id"] == payload["company_id"]
        assert body["input_feature_count"] == len(payload["features"])
        assert 0.0 <= body["bankruptcy_probability"] <= 1.0


def test_predict_rejects_missing_feature() -> None:
    payload = load_sample_payload()
    payload["features"].pop(next(iter(payload["features"])))
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "missing_features" in body["detail"]
