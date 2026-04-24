# HydroCast

HydroCast is an AI-based early warning system for waterborne disease outbreak monitoring in Maharashtra, India. The project combines epidemiological, climate, and WASH indicators with a HydroCast prediction pipeline and a modern web dashboard.

## Current Stack

- Frontend: Next.js
- API backend: FastAPI
- ML pipeline: Python
- Data outputs used by the dashboard: `results/*.json`, `results/*.csv`

## Main Features

- District-level outbreak risk monitoring
- Multi-disease forecasting for Cholera, Typhoid, and ADD
- Risk map dashboard
- SHAP-based explainability outputs
- Remedies and resource tracking dashboard views
- Render deployment support

## Project Structure

- `frontend/` - Next.js dashboard frontend and FastAPI API module
- `src/` - data pipeline, models, training, evaluation, remedy engine
- `models/` - trained model checkpoints
- `results/` - generated prediction and evaluation outputs
- `run_pipeline.py` - main project pipeline entry point
- `render.yaml` - Render deployment config

## Local Run

### 1. Backend

```powershell
cd X:\HydroCast
.venv\Scripts\python.exe -m uvicorn frontend.backend.dashboard_api:app --host 127.0.0.1 --port 8000
```

### 2. Frontend

```powershell
cd X:\HydroCast\frontend
npm.cmd install
npm.cmd run dev
```

### 3. Open in Browser

- Frontend: `http://localhost:3000/dashboard`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- FastAPI health: `http://127.0.0.1:8000/health`

## Pipeline Commands

Run the full pipeline:

```powershell
cd X:\HydroCast
python run_pipeline.py --mode all
```

Run individual steps:

```powershell
python run_pipeline.py --mode train
python run_pipeline.py --mode predict
python run_pipeline.py --mode eval
python run_pipeline.py --mode explain
python run_pipeline.py --mode dashboard
```

## Deployment

The repository includes `render.yaml` for deploying:

- `hydrocast-api`
- `hydrocast-dashboard`

## Notes

- The dashboard uses the generated outputs in `results/`.
- The current approved model checkpoint is stored in `models/`.
- No real `.env` file is tracked in the repository; only example/template env files should be committed.
