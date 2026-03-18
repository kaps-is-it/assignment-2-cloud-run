from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd


class ModelStore:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._model = None
        self._metadata: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self._model_path}")
        if not self._metadata_path.exists():
            raise FileNotFoundError(f"Metadata artifact not found at {self._metadata_path}")

        self._model = joblib.load(self._model_path)
        self._metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def feature_names(self) -> List[str]:
        return list(self._metadata["feature_names"])

    @property
    def threshold(self) -> float:
        return float(self._metadata["decision_threshold"])

    @property
    def model_version(self) -> str:
        return str(self._metadata["model_version"])

    def validate_features(self, features: Dict[str, float]) -> Tuple[List[str], List[str]]:
        expected = self.feature_names
        expected_set = set(expected)
        provided_set = set(features)
        missing = [name for name in expected if name not in provided_set]
        extra = sorted(provided_set - expected_set)
        return missing, extra

    def predict_probability(self, features: Dict[str, float]) -> float:
        row = [[float(features[name]) for name in self.feature_names]]
        frame = pd.DataFrame(row, columns=self.feature_names)
        probability = float(self._model.predict_proba(frame)[0, 1])
        return probability

    def risk_band(self, probability: float) -> str:
        if probability >= 0.50:
            return "CRITICAL"
        if probability >= self.threshold:
            return "ELEVATED"
        if probability >= 0.03:
            return "WATCHLIST"
        return "LOW"

    def risk_label(self, probability: float) -> str:
        return "HIGH_RISK" if probability >= self.threshold else "LOW_RISK"

    def admin_metadata(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "algorithm": self._metadata["algorithm"],
            "threshold": self.threshold,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "dataset_name": self._metadata["dataset"]["name"],
            "dataset_url": self._metadata["dataset"]["url"],
            "evaluation_split": self._metadata["evaluation_split"],
            "test_metrics": self._metadata["test_metrics"],
        }

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)
