from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(os.environ.get("HYDROCAST_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
GEOJSON_PATH = PROJECT_ROOT / "src" / "data" / "geojson" / "maharashtra_districts.geojson"
SOURCE_PATHS = [
    PROJECT_ROOT / "results" / "baseline_comparison.csv",
    PROJECT_ROOT / "results" / "classification_metrics.csv",
    PROJECT_ROOT / "results" / "lead_time.json",
    PROJECT_ROOT / "results" / "predictions.json",
    PROJECT_ROOT / "results" / "shap_values.json",
    PROJECT_ROOT / "src" / "data" / "processed" / "epiclim_maharashtra_merged.csv",
    GEOJSON_PATH,
]


def _cors_origins() -> list[str]:
    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    configured = [
        origin.strip()
        for origin in os.environ.get("FRONTEND_ORIGIN", "").split(",")
        if origin.strip()
    ]
    return defaults + configured

app = FastAPI(
    title="HydroCast Dashboard API",
    description="FastAPI backend for the HydroCast AI early warning dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict[str, Any] = {"stamp": None, "bundle": None, "geojson": None}


def _load_bundle_helpers():
    try:
        from frontend.export_dashboard_data import build_bundle, normalize
        return build_bundle, normalize
    except ModuleNotFoundError:
        try:
            from export_dashboard_data import build_bundle, normalize
            return build_bundle, normalize
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HydroCast dashboard bundle helpers could not be imported. "
                "Check the Render Python environment and project root configuration."
            ) from exc


def _source_stamp() -> tuple[float, ...]:
    return tuple(path.stat().st_mtime for path in SOURCE_PATHS if path.exists())


def _ensure_cache() -> None:
    stamp = _source_stamp()
    if _cache["stamp"] == stamp and _cache["bundle"] is not None and _cache["geojson"] is not None:
        return

    build_bundle, normalize = _load_bundle_helpers()
    _cache["bundle"] = normalize(build_bundle())
    _cache["geojson"] = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    _cache["stamp"] = stamp


def _bundle() -> dict[str, Any]:
    try:
        _ensure_cache()
        return _cache["bundle"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HydroCast bundle load failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dashboard/summary")
def dashboard_summary() -> dict[str, Any]:
    bundle = _bundle()
    return {
        "overview": bundle["overview"],
        "districtRankings": bundle["districtRankings"],
        "districtDetails": bundle["districtDetails"],
        "baselines": bundle["baselines"],
        "alerts": bundle["alerts"],
        "metadata": bundle["metadata"],
    }


@app.get("/api/dashboard/forecast")
def dashboard_forecast() -> dict[str, Any]:
    bundle = _bundle()
    return {
        "forecasts": bundle["forecasts"],
        "districtRankings": bundle["districtRankings"],
        "districtDetails": bundle["districtDetails"],
    }


@app.get("/api/dashboard/risk-map")
def dashboard_risk_map() -> dict[str, Any]:
    bundle = _bundle()
    _ensure_cache()
    return {
        "districts": [
            {
                "district": item["district"],
                "topDisease": item["top_disease"],
                "riskLevel": item["risk_level"],
                "riskScore": item["risk_score"],
                "rainfallAnomalyPct": item["rainfall_anomaly_pct"],
                "sanitationCoveragePct": item["sanitation_coverage_pct"],
                "caseCount": item["case_count"],
                "latitude": bundle["districtDetails"].get(item["district"], {}).get("latitude", 0),
                "longitude": bundle["districtDetails"].get(item["district"], {}).get("longitude", 0),
            }
            for item in bundle["districtRankings"]
        ],
        "geojson": _cache["geojson"],
    }


@app.get("/api/dashboard/shap")
def dashboard_shap() -> dict[str, Any]:
    bundle = _bundle()
    return {
        "shapValues": bundle["shapValues"],
        "globalShap": bundle["globalShap"],
    }


@app.get("/api/dashboard/resources")
def dashboard_resources() -> dict[str, Any]:
    bundle = _bundle()
    return {"resources": bundle["resources"], "remedies": bundle["remedies"]}

