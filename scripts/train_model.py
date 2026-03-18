from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "kaggle_dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DATASET_URL = "https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data"
DATASET_NAME = "Company Bankruptcy Prediction (Kaggle / fedesoriano)"
MODEL_VERSION = "risk-hgb-v1"
DECISION_THRESHOLD = 0.08
RANDOM_STATE = 42


def resolve_dataset() -> Path:
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    raise FileNotFoundError(
        "Expected kaggle_dataset.csv at the project root. "
        "Add the professor-aligned Kaggle dataset file before training."
    )


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]
    target = df.pop("Bankrupt?").astype(int)
    return df, target


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=300,
                    min_samples_leaf=20,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_test, probabilities, labels=[0, 1])), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
    }


def save_figures(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix @ threshold={DECISION_THRESHOLD}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def build_sample_request(X_reference: pd.DataFrame) -> dict[str, object]:
    medians = X_reference.median(numeric_only=True).to_dict()
    return {
        "company_id": "DEMO-001",
        "as_of_date": "2025-12-31",
        "features": {key: round(float(value), 6) for key, value in medians.items()},
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_dataset()
    X, y = load_dataset(dataset_path)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train_full, y_train_full)

    test_metrics = evaluate_model(model, X_test, y_test)
    save_figures(model, X_test, y_test)

    model_path = ARTIFACT_DIR / "bankruptcy_risk_model.joblib"
    sample_request_path = ARTIFACT_DIR / "sample_request.json"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    joblib.dump(model, model_path)
    sample_request_path.write_text(
        json.dumps(build_sample_request(X_train_full), indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "HistGradientBoostingClassifier",
        "decision_threshold": DECISION_THRESHOLD,
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": {
            "name": DATASET_NAME,
            "url": DATASET_URL,
            "local_path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "positive_class_rate": round(float(y.mean()), 4),
        },
        "evaluation_split": {
            "train_rows": int(len(X_train_full)),
            "test_rows": int(len(X_test)),
        },
        "feature_names": list(X.columns),
        "test_metrics": test_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({"model_path": str(model_path), "metadata_path": str(metadata_path), "metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
