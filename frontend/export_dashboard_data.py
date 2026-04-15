from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("HYDROCAST_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DISEASES
from src.dashboard.data_adapter import (
    CASE_COLUMN_MAP,
    get_baseline_results,
    get_district_snapshot,
    load_predictions,
    load_processed_dataset,
    load_shap_values,
)
from src.remedy_engine.remedy_engine import RemedyEngine


def normalize(value):
    if is_dataclass(value):
        return {k: normalize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def optional_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def build_bundle() -> dict:
    dataset = load_processed_dataset()
    predictions = load_predictions(dataset)
    shap_values = load_shap_values(predictions, dataset)
    baselines = get_baseline_results()
    engine = RemedyEngine()

    ranked = []
    district_details = {}
    district_forecasts = {}
    district_resources = {}
    remedies = {}
    alerts = []

    for district in predictions:
        snapshot = get_district_snapshot(predictions, district)
        ranked.append(snapshot)

    ranked.sort(key=lambda item: item.risk_score, reverse=True)

    for snapshot in ranked:
        district = snapshot.district
        district_df = dataset[dataset["district"] == district].sort_values("week").copy()
        latest = district_df.iloc[-1]

        district_details[district] = {
            "district": district,
            "topDisease": snapshot.top_disease,
            "riskLevel": snapshot.risk_level,
            "riskScore": round(safe_float(snapshot.risk_score), 4),
            "rainfallAnomalyPct": optional_float(round(safe_float(snapshot.rainfall_anomaly_pct), 2) if not pd.isna(snapshot.rainfall_anomaly_pct) else None),
            "sanitationCoveragePct": round(safe_float(snapshot.sanitation_coverage_pct), 2),
            "caseCount": int(snapshot.case_count),
            "population": int(latest.get("population_2024", 0)),
            "region": str(latest.get("region", "")),
            "latitude": safe_float(latest.get("latitude", 0.0)),
            "longitude": safe_float(latest.get("longitude", 0.0)),
            "washIndex": round(safe_float(latest.get("wash_index", 0.0)), 4),
            "waterAccessPct": round(safe_float(latest.get("water_access_pct", 0.0)), 2),
            "urbanPct": round(safe_float(latest.get("urban_pct", 0.0)), 2),
        }

        disease_forecasts = {}
        for disease in DISEASES:
            case_col = CASE_COLUMN_MAP[disease]
            history = district_df[["week", case_col]].tail(12).copy()
            history["week"] = pd.to_datetime(history["week"]).dt.strftime("%d %b")
            probs = predictions[district]["predictions"][disease]["week_probs"]
            baseline = max(float(history[case_col].mean()), 1.0)
            projected_cases = [round(baseline * (1 + prob * 1.35), 1) for prob in probs]
            disease_forecasts[disease] = {
                "history": history.to_dict(orient="records"),
                "weekProbabilities": probs,
                "projectedCases": projected_cases,
                "riskLevel": predictions[district]["predictions"][disease]["risk_level"],
                "maxProbability": predictions[district]["predictions"][disease]["max_prob"],
            }
        district_forecasts[district] = disease_forecasts

        sanitation_pct = safe_float(snapshot.sanitation_coverage_pct)
        rainfall_anomaly_pct = optional_float(snapshot.rainfall_anomaly_pct)
        sanitation = sanitation_pct / 100.0 if sanitation_pct > 1 else sanitation_pct
        district_remedies = {}
        for disease in DISEASES:
            disease_prediction = predictions[district]["predictions"][disease]
            plan = engine.generate_plan(
                district=district,
                disease=disease,
                risk_score=disease_prediction["max_prob"],
                district_sanitation=sanitation,
                rainfall_anomaly=rainfall_anomaly_pct or 0.0,
            )
            district_remedies[disease] = normalize(plan)
        remedies[district] = district_remedies

        risk_weight = snapshot.risk_score
        district_resources[district] = {
            "ORS sachets": max(18, int(95 - risk_weight * 52)),
            "IV fluids": max(15, int(88 - risk_weight * 48)),
            "Chlorine tablets": max(20, int(90 - risk_weight * 55)),
            "Rapid test kits": max(12, int(84 - risk_weight * 58)),
            "Field staff": max(25, int(92 - risk_weight * 30)),
        }

        alerts.append(
            {
                "district": district,
                "severity": snapshot.risk_level,
                "title": f"{snapshot.top_disease} surveillance alert",
                "message": (
                    f"{district} is at {snapshot.risk_score:.0%} projected risk with "
                    f"{snapshot.case_count} recent cases, rainfall anomaly "
                    f"{f'{rainfall_anomaly_pct:.1f}%' if rainfall_anomaly_pct is not None else 'data unavailable'} "
                    f"and sanitation coverage {sanitation_pct:.1f}%."
                ),
            }
        )

    overview = {
        "activeOutbreaks": sum(1 for item in ranked if item.risk_level in {"critical", "high"}),
        "highRiskDistricts": sum(1 for item in ranked if item.risk_level in {"critical", "high"}),
        "criticalDistricts": sum(1 for item in ranked if item.risk_level == "critical"),
        "modelF1": round(float(baselines["F1_Macro"].max()), 3),
        "actionsIssued": sum(1 for item in ranked if item.risk_level in {"critical", "high"}) * 4,
        "observedCases": sum(item.case_count for item in ranked),
        "averageRisk": round(sum(item.risk_score for item in ranked) / max(len(ranked), 1), 4),
    }

    global_shap = {}
    for district_data in shap_values.values():
        for disease_data in district_data.values():
            for feature_name, feature_value in disease_data.get("top_features", []):
                global_shap[feature_name] = global_shap.get(feature_name, 0.0) + abs(feature_value)
    global_shap = sorted(global_shap.items(), key=lambda item: item[1], reverse=True)[:12]

    return {
        "overview": overview,
        "districtRankings": [normalize(item) for item in ranked],
        "districtDetails": district_details,
        "forecasts": district_forecasts,
        "baselines": baselines.to_dict(orient="records"),
        "shapValues": normalize(shap_values),
        "globalShap": [{"feature": feature, "value": round(float(value), 4)} for feature, value in global_shap],
        "remedies": remedies,
        "resources": district_resources,
        "alerts": alerts[:10],
        "metadata": {
            "title": "HydroCast",
            "subtitle": "AI Early Warning System",
            "state": "Maharashtra, India",
            "diseases": DISEASES,
        },
    }


def main() -> None:
    bundle = build_bundle()
    output_dir = Path(__file__).resolve().parent / "public" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dashboard-data.json").write_text(json.dumps(normalize(bundle), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
