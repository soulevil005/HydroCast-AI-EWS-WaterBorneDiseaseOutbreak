# HydroCast 🌊🩺  
### AI-Based Early Warning System for Waterborne Disease Outbreak Monitoring in Maharashtra, India

HydroCast is an intelligent disease surveillance and forecasting platform designed to predict **waterborne disease outbreaks** using machine learning, climate intelligence, and WASH (Water, Sanitation, Hygiene) indicators.  

Built specifically for **Maharashtra, India**, HydroCast helps public health teams, administrators, and researchers proactively identify outbreak risks for diseases such as:

- Cholera  
- Typhoid  
- Acute Diarrheal Disease (ADD)

Instead of reacting after outbreaks occur, HydroCast enables **early intervention, smarter planning, and faster response.** :contentReference[oaicite:0]{index=0}

---

# 🚀 Key Features

## 📍 District-Level Risk Monitoring
Track disease outbreak risk across Maharashtra districts with dynamic scoring and ranking.

## 🤖 AI Disease Forecasting
Multi-disease prediction models powered by machine learning and epidemiological signals.

## 🗺️ Interactive Risk Heatmaps
Visual dashboards showing district-wise hotspots and risk severity.

## 📈 Forecast Trends
Future outbreak probability graphs for Cholera, Typhoid, and ADD.

## 🔍 Explainable AI (SHAP)
Understand **why** a district is high-risk through feature importance and local explanations.

## 🏥 Resource Readiness Tracker
Monitor medical resources, readiness %, tablets, ORS kits, response units, and field capacity.

## 💡 Remedies & Precautions
District-specific public health recommendations and mitigation protocols.

## ☁️ Cloud Deployment Ready
Configured for deployment using **Render** with separate frontend + backend services.

---

# 🧠 How HydroCast Works

HydroCast combines multiple real-world signals:

- Historical disease outbreak data  
- Rainfall patterns  
- Temperature  
- Humidity  
- Seasonal patterns  
- WASH indicators  
- Sanitation access  
- District-level trends  

These are processed through an AI prediction pipeline to generate outbreak risk alerts. :contentReference[oaicite:1]{index=1}

---

# 🏗️ Tech Stack

| Layer | Technology |
|------|------------|
| Frontend | Next.js, React |
| Styling | Tailwind CSS |
| Backend API | FastAPI |
| ML Pipeline | Python |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Modeling | Scikit-learn / Deep Learning |
| Deployment | Render |

---

# 📂 Project Structure

```bash
HydroCast/
│── frontend/                 # Next.js dashboard + FastAPI API module
│── src/                      # ML pipeline, training, evaluation, explainability
│── models/                   # Trained model checkpoints
│── results/                  # Prediction outputs (.json / .csv)
│── run_pipeline.py           # Main execution entry point
│── render.yaml               # Render deployment config
│── README.md
```
---

# ⚙️ Local Run

### 1. Backend

```cmd
cd X:\HydroCast
.venv\Scripts\python.exe -m uvicorn frontend.backend.dashboard_api:app --host 127.0.0.1 --port 8000
```

### 2. Frontend

```cmd
cd X:\HydroCast\frontend
npm.cmd install
npm.cmd run dev
```

### 3. Open in Browser

- Frontend: `http://localhost:3000/dashboard`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- FastAPI health: `http://127.0.0.1:8000/health`

---

## 🧪 Pipeline Commands

###  Run the full pipeline:

```cmd
cd X:\HydroCast
python run_pipeline.py --mode all
```

### Run individual steps:

```cmd
python run_pipeline.py --mode train
python run_pipeline.py --mode predict
python run_pipeline.py --mode eval
python run_pipeline.py --mode explain
python run_pipeline.py --mode dashboard
```

## ☁️ Deployment

The repository includes `render.yaml` for deploying:

- `hydrocast-api`
- `hydrocast-dashboard`

---

## 📊 Dashboard Modules

- Login Portal
- Dashboard Overview
- Risk Heatmap
- Forecast Charts
- AI Explainability
- Remedies & Precautions
- Resource Tracker

---

## ⭐ Support This Project
If you like HydroCast, consider giving it a ⭐ on GitHub.

---

## 🌊 HydroCast

Predicting outbreaks before they happen.
AI for Public Health.
