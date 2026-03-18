# Bankruptcy Risk Modeling API

This project implements Assignment 2 Part B as a production-style machine learning API. The chosen prediction task is `Risk Modeling`, using the professor-aligned Kaggle `Company Bankruptcy Prediction` dataset. The service trains a binary classifier, exposes a FastAPI `/docs` interface, validates input with Pydantic, tracks runtime metrics, and is packaged for Docker and Google Cloud Run deployment.

## Why This Direction

Part A was completed as a secure ML API with `predict`, `status`, and admin-oriented operations. For Part B, the assignment only allows a smaller set of tasks, so this implementation keeps the same API engineering discipline while switching the prediction problem to an allowed task: company bankruptcy risk scoring.

The final model uses `HistGradientBoostingClassifier` because it gave the best probability quality during local benchmarking across the tested options. The deployed decision threshold is set to `0.08` to bias the service slightly toward catching risky companies instead of optimizing raw accuracy.

## Users

- `Credit risk analysts` use the API for synchronous, single-company checks during underwriting reviews.
- `Portfolio monitoring services` call the API in batch-style workflows to rescore companies across a watchlist.
- `MLOps / platform admins` monitor model health, runtime metrics, and model metadata.

Expected daily volume: `2,000 requests/day`

User requirements:

- Real-time JSON scoring for individual companies.
- Stable schema with strict input validation.
- Probability output, not just a hard class label.
- Visibility into model version, threshold, and API health.

## Dataset

- Dataset: `Company Bankruptcy Prediction`
- Source: [Kaggle dataset page](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data)
- Local file used by the training script: `kaggle_dataset.csv`
- Rows: `6,819`
- Features: `95`
- Positive class rate: `3.23%`

## Model Results

Held-out test metrics from the saved artifact:

- `ROC-AUC`: `0.9597`
- `PR-AUC`: `0.5294`
- `F1`: `0.4842`
- `Precision`: `0.4510`
- `Recall`: `0.5227`
- `Log Loss`: `0.0995`
- `Brier Score`: `0.0222`

Why these metrics:

- `Accuracy` is not reported as the primary metric because the dataset is heavily imbalanced.
- `ROC-AUC` and `PR-AUC` show ranking quality.
- `Log Loss` and `Brier Score` show probability quality.
- `Precision`, `Recall`, and `F1` show the operational effect of the chosen threshold.

## API Performance

Local measurements from 30 live `/api/v1/predict` requests against the saved model:

- Average response time: `9.237 ms`
- p95 response time: `15.942 ms`
- Min response time: `4.903 ms`
- Max response time: `26.095 ms`
- Runtime memory observed by the API: `161.23 MB` to `162.83 MB`

The service also exposes live runtime metrics at `/api/v1/metrics`.

## Architecture

```mermaid
flowchart LR
    A["Risk Analyst / Monitoring Client"] --> B["FastAPI Service"]
    B --> C["Pydantic Request Validation"]
    C --> D["Saved HistGradientBoosting Model"]
    D --> E["Probability + Risk Label Response"]
    B --> F["Runtime Metrics Tracker"]
    F --> G["/api/v1/metrics"]
    B --> H["/api/v1/status"]
    B --> I["/api/v1/admin/model"]
```

## Live Deployment

The current deployed Cloud Run version is publicly available here:

- Base URL: [https://assignment-2-cloud-run-1095831070964.us-central1.run.app](https://assignment-2-cloud-run-1095831070964.us-central1.run.app)
- Interactive API docs: [https://assignment-2-cloud-run-1095831070964.us-central1.run.app/docs](https://assignment-2-cloud-run-1095831070964.us-central1.run.app/docs)
- Status endpoint: [https://assignment-2-cloud-run-1095831070964.us-central1.run.app/api/v1/status](https://assignment-2-cloud-run-1095831070964.us-central1.run.app/api/v1/status)
- Metrics endpoint: [https://assignment-2-cloud-run-1095831070964.us-central1.run.app/api/v1/metrics](https://assignment-2-cloud-run-1095831070964.us-central1.run.app/api/v1/metrics)
- Model metadata endpoint: [https://assignment-2-cloud-run-1095831070964.us-central1.run.app/api/v1/admin/model](https://assignment-2-cloud-run-1095831070964.us-central1.run.app/api/v1/admin/model)

## How a New User Can Try the API


```mermaid
flowchart TD
    A["Open the live Cloud Run URL"] --> B["Open /docs"]
    B --> C["Expand POST /api/v1/predict"]
    C --> D["Click Try it out"]
    D --> E["Paste the sample request JSON"]
    E --> F["Click Execute"]
    F --> G["Read bankruptcy_probability, risk_label, risk_band"]
    G --> H["Open GET /api/v1/status"]
    H --> I["Open GET /api/v1/metrics"]
    I --> J["Open GET /api/v1/admin/model"]
```

## Sample Request JSON: 
{
  "company_id": "DEMO-001",
  "as_of_date": "2025-12-31",
  "features": {
    "ROA(C) before interest and depreciation before interest": 0.542214,
    "ROA(A) before interest and % after tax": 0.554595,
    "ROA(B) before interest and depreciation after tax": 0.548471,
    "Operating Gross Margin": 0.607948,
    "Realized Sales Gross Margin": 0.607929,
    "Operating Profit Rate": 0.998969,
    "Pre-tax net Interest Rate": 0.797558,
    "After-tax net Interest Rate": 0.809376,
    "Non-industry income and expenditure/revenue": 0.303623,
    "Continuous interest rate (after tax)": 0.781381,
    "Operating Expense Rate": 0.00018,
    "Research and development expense rate": 0.0,
    "Cash flow rate": 0.460382,
    "Interest-bearing debt interest rate": 0.000321,
    "Tax rate (A)": 0.0,
    "Net Value Per Share (B)": 0.186192,
    "Net Value Per Share (A)": 0.186192,
    "Net Value Per Share (C)": 0.186192,
    "Persistent EPS in the Last Four Seasons": 0.228813,
    "Cash Flow Per Share": 0.322293,
    "Revenue Per Share (Yuan ¥)": 0.031847,
    "Operating Profit Per Share (Yuan ¥)": 0.100299,
    "Per Share Net profit before tax (Yuan ¥)": 0.175584,
    "Realized Sales Gross Profit Growth Rate": 0.022102,
    "Operating Profit Growth Rate": 0.848044,
    "After-tax Net Profit Growth Rate": 0.689706,
    "Regular Net Profit Growth Rate": 0.689703,
    "Continuous Net Profit Growth Rate": 0.217639,
    "Total Asset Growth Rate": 0.000472,
    "Net Value Growth Rate": 0.000464,
    "Total Asset Return Growth Rate Ratio": 0.264049,
    "Cash Reinvestment %": 0.386218,
    "Current Ratio": 0.01031,
    "Quick Ratio": 0.00655,
    "Interest Expense Ratio": 0.630098,
    "Total debt/Total net worth": 0.069963,
    "Debt ratio %": 0.278123,
    "Net worth/Assets": 0.721877,
    "Long-term fund suitability ratio (A)": 0.006611,
    "Borrowing dependency": 0.374955,
    "Contingent liabilities/Net worth": 0.005366,
    "Operating profit/Paid-in capital": 0.100572,
    "Net profit before tax/Paid-in capital": 0.174012,
    "Inventory and accounts receivable/Net value": 0.403161,
    "Total Asset Turnover": 0.112759,
    "Accounts Receivable Turnover": 0.001969,
    "Average Collection Days": 0.004713,
    "Inventory Turnover Rate (times)": 0.000175,
    "Fixed Assets Turnover Frequency": 0.000101,
    "Net Worth Turnover Rate (times)": 0.032839,
    "Revenue per person": 0.043227,
    "Operating profit per person": 0.392232,
    "Allocation rate per person": 0.037703,
    "Working Capital to Total Assets": 0.810499,
    "Quick Assets/Total Assets": 0.137213,
    "Current Assets/Total Assets": 0.198469,
    "Cash/Total Assets": 0.007717,
    "Quick Assets/Current Liability": 0.005158,
    "Cash/Current Liability": 0.000814,
    "Current Liability to Assets": 0.118987,
    "Operating Funds to Liability": 0.345483,
    "Inventory/Working Capital": 0.276464,
    "Inventory/Current Liability": 0.010387,
    "Current Liabilities/Liability": 0.949293,
    "Working Capital/Equity": 0.72175,
    "Current Liabilities/Equity": 0.379749,
    "Long-term Liability to Current Assets": 0.009535,
    "Retained Earnings to Total Assets": 0.90315,
    "Total income/Total expense": 0.00213,
    "Total expense/Assets": 0.063382,
    "Current Asset Turnover Rate": 0.000191,
    "Quick Asset Turnover Rate": 0.00013,
    "Working capitcal Turnover Rate": 0.000226,
    "Cash Turnover Rate": 0.00116,
    "Cash Flow to Sales": 0.67157,
    "Fixed Assets to Assets": 0.424734,
    "Current Liability to Liability": 0.949293,
    "Current Liability to Equity": 0.379749,
    "Equity to Long-term Liability": 0.126982,
    "Cash Flow to Total Assets": 0.644168,
    "Cash Flow to Liability": 0.458801,
    "CFO to Assets": 0.557733,
    "Cash Flow to Equity": 0.317327,
    "Current Liability to Current Assets": 0.060578,
    "Liability-Assets Flag": 0.0,
    "Net Income to Total Assets": 0.797687,
    "Total assets to GNP price": 0.009893,
    "No-credit Interval": 0.623716,
    "Gross Profit to Sales": 0.607935,
    "Net Income to Stockholder's Equity": 0.840007,
    "Liability to Equity": 0.279476,
    "Degree of Financial Leverage (DFL)": 0.026791,
    "Interest Coverage Ratio (Interest expense to EBIT)": 0.565341,
    "Net Income Flag": 1.0,
    "Equity to Liability": 0.020571
  }
}


Quick reviewer steps:

1. Open the live docs page.
2. Expand `POST /api/v1/predict`.
3. Copy the sample payload from `artifacts/sample_request.json`.
4. Paste it into the request body and click `Execute`.
5. Review the returned prediction output.
6. Open `GET /api/v1/status` to confirm the service is healthy.
7. Open `GET /api/v1/metrics` to inspect runtime monitoring data.
8. Open `GET /api/v1/admin/model` to inspect the deployed model metadata.

## Repository Layout

```text
app/                   FastAPI application code
artifacts/             Trained model, metadata, and sample payload
deployment/            Deployment environment templates
reports/figures/       Evaluation plots for the report/video
scripts/train_model.py Training pipeline
tests/                 API tests
Dockerfile             Container build
README.md              Setup, architecture, and deployment guide
```

## Endpoints

- `POST /api/v1/predict`
- `GET /api/v1/status`
- `GET /api/v1/metrics`
- `GET /api/v1/schema/input`
- `GET /api/v1/admin/model`
- `POST /api/v1/admin/reload`
- `GET /docs`

Prediction response fields:

- `bankruptcy_probability`
- `risk_label`
- `risk_band`
- `decision_threshold`
- `model_version`
- `processing_time_ms`

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the auto-generated docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Example Usage

Use the generated sample payload:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  --data @artifacts/sample_request.json
```

Check status:

```bash
curl http://127.0.0.1:8000/api/v1/status
```

Check runtime metrics:

```bash
curl http://127.0.0.1:8000/api/v1/metrics
```

## Testing

```bash
pytest -q
```

Current local result: `3 passed`

## Docker

Build:

```bash
docker build -t bankruptcy-risk-api .
```

Run:

```bash
docker run --rm -p 8080:8080 \
  -e EXPECTED_DAILY_REQUESTS=2000 \
  -e ADMIN_TOKEN=change-me \
  bankruptcy-risk-api
```

Then open [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs).

## Cloud Run Deployment

This repo includes a Dockerfile for the containerization requirement. For deployment to Cloud Run, the simplest production path is:

1. Set your Google Cloud project and region.
2. Build the image with Cloud Build.
3. Deploy the pushed image to Cloud Run.

Example:

```bash
export PROJECT_ID="your-project-id"
export REGION="northamerica-northeast1"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/assignment2/bankruptcy-risk-api:latest"

gcloud auth login
gcloud config set project "${PROJECT_ID}"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud builds submit --tag "${IMAGE_URI}"
gcloud run deploy bankruptcy-risk-api \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars EXPECTED_DAILY_REQUESTS=2000
```

Notes:

- Cloud Run injects the `PORT` environment variable automatically.
- If you want to use the admin reload endpoint, also set `ADMIN_TOKEN`.
- Cloud Run scales from zero, which matches the request-volume assumptions for this assignment.
- The deployment flow above follows Google Cloud's official guides for [building and pushing Docker images with Cloud Build](https://cloud.google.com/build/docs/build-push-docker-image) and [deploying container images to Cloud Run](https://cloud.google.com/run/docs/deploying).


## Deliverables in This Repo

- Trained model artifact: [artifacts/bankruptcy_risk_model.joblib](./artifacts/bankruptcy_risk_model.joblib)
- Model metadata: [artifacts/model_metadata.json](./artifacts/model_metadata.json)
- Sample request: [artifacts/sample_request.json](./artifacts/sample_request.json)
- ROC curve: [reports/figures/roc_curve.png](./reports/figures/roc_curve.png)
- Precision-recall curve: [reports/figures/precision_recall_curve.png](./reports/figures/precision_recall_curve.png)
- Confusion matrix: [reports/figures/confusion_matrix.png](./reports/figures/confusion_matrix.png)

Last verified deployment target: Google Cloud Run in `us-central1`.
