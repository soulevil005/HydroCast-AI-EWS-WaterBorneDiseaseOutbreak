"""
HydroCast — Geographic Map Component
Folium-based interactive choropleth + bubble map of Maharashtra.
Renders real district boundaries from GeoJSON over CartoDB dark tiles.

Usage inside Streamlit:
    from src.dashboard.map_component import render_risk_map, display_in_streamlit
    m = render_risk_map(predictions, geojson_path)
    display_in_streamlit(m)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
from folium import GeoJson, GeoJsonTooltip, CircleMarker, LayerControl

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MAHARASHTRA_DISTRICTS, DISTRICT_CENTROIDS,
    RISK_COLORS, RISK_THRESHOLDS, DATA_GEOJSON_DIR,
)

logger = logging.getLogger("hydrocast.map_component")

# ── Default GeoJSON URL (fallback when local file missing)
GEOJSON_URL = (
    "https://raw.githubusercontent.com/geohacker/india/"
    "master/district/india_district.geojson"
)


# ══════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════

RISK_FILL = {
    "critical": "rgba(255,61,90,0.45)",
    "high":     "rgba(255,184,77,0.38)",
    "medium":   "rgba(77,159,255,0.30)",
    "low":      "rgba(0,229,160,0.20)",
}

RISK_FILL_HEX = {
    "critical": "#ff3d5a",
    "high":     "#ffb84d",
    "medium":   "#4d9fff",
    "low":      "#00e5a0",
}

RISK_FILL_OPACITY = {
    "critical": 0.55,
    "high":     0.42,
    "medium":   0.30,
    "low":      0.18,
}


def _get_risk_level(prob: float) -> str:
    for lvl, thr in [("critical", 0.8), ("high", 0.6), ("medium", 0.4)]:
        if prob >= thr:
            return lvl
    return "low"


def _hex_with_opacity(hex_color: str, opacity: float) -> str:
    """Convert hex color to rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


# ══════════════════════════════════════════════════════════════════
# TOOLTIP HTML BUILDER
# ══════════════════════════════════════════════════════════════════

def _build_tooltip_html(
    district:   str,
    risk_level: str,
    predictions: dict,
) -> str:
    """Build rich HTML tooltip for a district."""
    d_pred = predictions.get(district, {})
    col    = RISK_FILL_HEX.get(risk_level, "#888")

    context = d_pred.get("context", {})

    # Disease rows
    disease_rows = ""
    for dis, info in d_pred.get("predictions", {}).items():
        prob = info.get("max_prob", 0)
        disease_rows += (
            f"<tr>"
            f"<td style='color:#6b7a96;padding:1px 8px 1px 0'>{dis}</td>"
            f"<td style='color:{col};font-weight:700;font-family:monospace'>"
            f"{prob:.0%}</td>"
            f"</tr>"
        )

    max_prob = max(
        (v.get("max_prob", 0) for v in d_pred.get("predictions", {}).values()),
        default=0,
    )
    top_disease = max(
        d_pred.get("predictions", {}),
        key=lambda name: d_pred.get("predictions", {}).get(name, {}).get("max_prob", 0),
        default="Unknown",
    )

    # Progress bar width
    bar_w = int(max_prob * 100)

    return f"""
    <div style='background:#0d1420;border:1px solid rgba(255,255,255,0.12);
         border-radius:9px;padding:12px 15px;min-width:210px;
         font-family:DM Sans,sans-serif;color:#e8edf5;'>

        <div style='font-family:Space Mono,monospace;font-size:12px;
                    font-weight:700;color:{col};margin-bottom:8px'>
            {district}
            <span style='font-size:9px;background:rgba(255,255,255,0.08);
                         padding:1px 6px;border-radius:5px;margin-left:6px;
                         color:{col};border:1px solid {col}44'>
                {risk_level.upper()}
            </span>
        </div>

        <table style='width:100%;border-collapse:collapse;font-size:11px'>
            {disease_rows}
        </table>

        <div style='margin-top:8px;font-size:10px;color:#c2d1ea;line-height:1.6'>
            Disease: <span style='color:{col}'>{top_disease}</span><br>
            Risk score: <span style='color:{col}'>{max_prob:.2f}</span><br>
            Rainfall anomaly: {context.get("rainfall_anomaly_pct", 0):.1f}%<br>
            Sanitation: {context.get("sanitation_coverage_pct", 0):.1f}<br>
            Case count: {context.get("case_count", 0)}
        </div>

        <div style='height:4px;background:#131c2e;border-radius:2px;margin-top:9px'>
            <div style='width:{bar_w}%;height:100%;background:{col};
                        border-radius:2px'></div>
        </div>
        <div style='font-size:9px;color:#6b7a96;margin-top:4px;font-family:Space Mono'>
            Overall risk score: {max_prob:.2f}
        </div>

        <div style='font-size:9px;color:#4d9fff;margin-top:7px;cursor:pointer'>
            → View remedy plan
        </div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════

LEGEND_HTML = """
<div style='position:absolute;bottom:30px;left:10px;z-index:9999;
     background:#0d1420;border:1px solid rgba(255,255,255,0.1);
     border-radius:9px;padding:12px 14px;font-family:Space Mono,monospace;
     font-size:10px;color:#e8edf5;box-shadow:0 4px 20px rgba(0,0,0,0.5)'>
    <div style='font-size:9px;color:#6b7a96;letter-spacing:1px;
                text-transform:uppercase;margin-bottom:7px'>Risk Level</div>
    <div style='display:flex;align-items:center;gap:7px;margin-bottom:5px'>
        <div style='width:11px;height:11px;border-radius:50%;
                    background:#ff3d5a;border:1px solid #ff3d5a88'></div>
        <span>Critical  (≥ 0.80)</span>
    </div>
    <div style='display:flex;align-items:center;gap:7px;margin-bottom:5px'>
        <div style='width:11px;height:11px;border-radius:50%;
                    background:#ffb84d;border:1px solid #ffb84d88'></div>
        <span>High      (≥ 0.60)</span>
    </div>
    <div style='display:flex;align-items:center;gap:7px;margin-bottom:5px'>
        <div style='width:11px;height:11px;border-radius:50%;
                    background:#4d9fff;border:1px solid #4d9fff88'></div>
        <span>Medium    (≥ 0.40)</span>
    </div>
    <div style='display:flex;align-items:center;gap:7px'>
        <div style='width:11px;height:11px;border-radius:50%;
                    background:#00e5a0;border:1px solid #00e5a088'></div>
        <span>Low       (&lt; 0.40)</span>
    </div>
    <div style='margin-top:9px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.07);
                font-size:8px;color:#6b7a96'>
        Bubble size = weekly case burden
    </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════
# MAIN MAP RENDERER
# ══════════════════════════════════════════════════════════════════

def render_risk_map(
    predictions:       dict,
    geojson_path:      Optional[Path] = None,
    selected_district: str            = None,
    layer_mode:        str            = "both",
) -> folium.Map:
    """
    Build the full interactive risk map for Maharashtra.

    Parameters
    ----------
    predictions       : {district: {predictions: {disease: {...}}}}
    geojson_path      : path to GeoJSON file (auto-downloads if None)
    selected_district : highlight this district with a white ring
    layer_mode        : "choropleth", "bubbles", or "both"

    Returns
    -------
    folium.Map  ready to embed in Streamlit
    """
    # ── Create base map with dark tile layer
    m = folium.Map(
        location          = [19.5, 76.5],
        zoom_start        = 6,
        tiles             = "CartoDB dark_matter",
        prefer_canvas     = True,
        control_scale     = True,
    )

    # ── Add choropleth layer (real GeoJSON district boundaries)
    if layer_mode in ["choropleth", "both"]:
        _add_choropleth_layer(m, predictions, geojson_path)

    # ── Add bubble layer (centroid circles)
    if layer_mode in ["bubbles", "both"]:
        _add_bubble_layer(m, predictions)

    # ── Highlight selected district
    if selected_district and selected_district in DISTRICT_CENTROIDS:
        _highlight_selected(m, selected_district, predictions)

    # ── Legend
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))

    # ── Layer control
    LayerControl(collapsed=False).add_to(m)

    logger.info(f"Map rendered: mode={layer_mode} | districts={len(predictions)}")
    return m


# ══════════════════════════════════════════════════════════════════
# CHOROPLETH LAYER
# ══════════════════════════════════════════════════════════════════

def _add_choropleth_layer(
    m:            folium.Map,
    predictions:  dict,
    geojson_path: Optional[Path],
) -> None:
    """Load GeoJSON and colour each district polygon by risk level."""

    # ── Load GeoJSON (local file or fallback URL)
    geojson_data = None
    local_path   = geojson_path or (DATA_GEOJSON_DIR / "maharashtra_districts.geojson")

    if local_path.exists():
        try:
            with open(local_path) as f:
                geojson_data = json.load(f)
            logger.info(f"GeoJSON loaded from: {local_path}")
        except Exception as e:
            logger.warning(f"Local GeoJSON load failed: {e}")

    if geojson_data is None:
        # Try fetching from GitHub
        try:
            import requests
            resp = requests.get(GEOJSON_URL, timeout=10)
            if resp.status_code == 200:
                india_geojson = resp.json()
                # Filter Maharashtra districts
                mh_features = [
                    f for f in india_geojson.get("features", [])
                    if "maharashtra" in str(f.get("properties", "")).lower()
                ]
                geojson_data = {"type": "FeatureCollection", "features": mh_features}
                logger.info(f"GeoJSON fetched: {len(mh_features)} MH features")
        except Exception as e:
            logger.warning(f"GeoJSON fetch failed: {e}")

    if geojson_data is None:
        logger.warning("No GeoJSON available — skipping choropleth layer.")
        return

    # ── Add GeoJson layer
    def style_function(feature):
        props   = feature.get("properties", {})
        name    = (props.get("DISTRICT") or props.get("district") or
                   props.get("NAME_2") or "")
        matched = next(
            (d for d in MAHARASHTRA_DISTRICTS
             if name.lower() in d.lower() or d.lower() in name.lower()),
            None,
        )
        if not matched:
            return {
                "fillColor":   "#222",
                "fillOpacity": 0.2,
                "color":       "#333",
                "weight":      0.5,
                "opacity":     0.5,
            }

        d_pred = predictions.get(matched, {})
        max_prob = max(
            (v.get("max_prob", 0) for v in d_pred.get("predictions", {}).values()),
            default=0.1,
        )
        rl = _get_risk_level(max_prob)
        return {
            "fillColor":   RISK_FILL_HEX[rl],
            "fillOpacity": RISK_FILL_OPACITY[rl],
            "color":       RISK_FILL_HEX[rl],
            "weight":      1.2,
            "opacity":     0.8,
        }

    def highlight_function(feature):
        return {"weight": 3, "color": "white", "fillOpacity": 0.9}

    folium.GeoJson(
        geojson_data,
        name           = "District Risk (Choropleth)",
        style_function = style_function,
        highlight_function = highlight_function,
        tooltip        = GeoJsonTooltip(
            fields  = ["DISTRICT"],
            aliases = ["District:"],
            style   = ("background-color:#0d1420;color:#e8edf5;"
                       "font-family:Space Mono;font-size:11px;"),
        ),
    ).add_to(m)


# ══════════════════════════════════════════════════════════════════
# BUBBLE LAYER
# ══════════════════════════════════════════════════════════════════

def _add_bubble_layer(m: folium.Map, predictions: dict) -> None:
    """Add CircleMarkers at district centroids, sized by case burden."""

    bubble_group = folium.FeatureGroup(name="District Bubbles", show=True)

    for district in MAHARASHTRA_DISTRICTS:
        if district not in DISTRICT_CENTROIDS:
            continue

        lat, lon = DISTRICT_CENTROIDS[district]
        d_pred   = predictions.get(district, {})

        max_prob = max(
            (v.get("max_prob", 0) for v in d_pred.get("predictions", {}).values()),
            default=0.1,
        )
        risk_level = _get_risk_level(max_prob)
        color      = RISK_FILL_HEX[risk_level]

        # Radius proportional to weekly case count proxy
        weekly_cases = int(max_prob * 150)
        radius       = max(5, min(22, 5 + weekly_cases * 0.12))

        tooltip_html = _build_tooltip_html(district, risk_level, predictions)

        CircleMarker(
            location    = [lat, lon],
            radius      = radius,
            color       = color,
            fill        = True,
            fill_color  = color,
            fill_opacity= 0.72,
            weight      = 1.8,
            opacity     = 0.90,
            tooltip     = folium.Tooltip(tooltip_html, sticky=True,
                                         max_width=280),
        ).add_to(bubble_group)

    bubble_group.add_to(m)


# ══════════════════════════════════════════════════════════════════
# SELECTED DISTRICT HIGHLIGHT
# ══════════════════════════════════════════════════════════════════

def _highlight_selected(
    m:            folium.Map,
    district:     str,
    predictions:  dict,
) -> None:
    """Add a pulsing white-bordered marker for the selected district."""
    if district not in DISTRICT_CENTROIDS:
        return

    lat, lon   = DISTRICT_CENTROIDS[district]
    d_pred     = predictions.get(district, {})
    max_prob   = max(
        (v.get("max_prob", 0) for v in d_pred.get("predictions", {}).values()),
        default=0.5,
    )
    risk_level = _get_risk_level(max_prob)
    color      = RISK_FILL_HEX[risk_level]
    tooltip    = _build_tooltip_html(district, risk_level, predictions)

    # Outer white ring
    CircleMarker(
        location     = [lat, lon],
        radius       = 26,
        color        = "white",
        fill         = False,
        weight       = 2.5,
        opacity      = 0.9,
        tooltip      = folium.Tooltip(tooltip, sticky=True,
                                      max_width=280),
    ).add_to(m)

    # Inner filled circle
    CircleMarker(
        location     = [lat, lon],
        radius       = 18,
        color        = color,
        fill         = True,
        fill_color   = color,
        fill_opacity = 0.85,
        weight       = 2,
        tooltip      = folium.Tooltip(tooltip, sticky=True,
                                      max_width=280),
    ).add_to(m)

    # Label
    folium.Marker(
        location  = [lat + 0.3, lon],
        icon      = folium.DivIcon(
            html = f"""
            <div style='font-family:Space Mono,monospace;font-size:10px;
                        color:{color};font-weight:700;
                        background:rgba(13,20,32,0.85);
                        border:1px solid {color}55;
                        border-radius:5px;padding:2px 7px;
                        white-space:nowrap'>
                ◉ {district}
            </div>
            """,
            icon_size    = (160, 24),
            icon_anchor  = (80, 0),
        ),
    ).add_to(m)


# ══════════════════════════════════════════════════════════════════
# STREAMLIT INTEGRATION
# ══════════════════════════════════════════════════════════════════

def display_in_streamlit(
    folium_map: folium.Map,
    height:     int = 520,
) -> None:
    """
    Render a Folium map inside a Streamlit app.

    Requires: pip install streamlit-folium

    Parameters
    ----------
    folium_map : rendered folium.Map object
    height     : pixel height (default 520)
    """
    try:
        from streamlit_folium import st_folium
        st_folium(folium_map, width=None, height=height, returned_objects=[])
    except ImportError:
        import streamlit as st
        import tempfile, os

        # Fallback: save to temp HTML and embed via components
        with tempfile.NamedTemporaryFile(suffix=".html",
                                         delete=False, mode="w") as f:
            folium_map.save(f.name)
            tmp_path = f.name

        with open(tmp_path, "r") as f:
            html_content = f.read()
        os.unlink(tmp_path)

        import streamlit.components.v1 as components
        components.html(html_content, height=height, scrolling=False)


def save_map_html(folium_map: folium.Map, save_path: Path) -> None:
    """Save the Folium map as a standalone HTML file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    folium_map.save(str(save_path))
    logger.info(f"Map saved to: {save_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test (generates standalone HTML map)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import random
    random.seed(42)

    # ── Generate demo predictions
    demo_preds = {}
    for i, district in enumerate(MAHARASHTRA_DISTRICTS):
        base = max(0.05, 0.91 - i * 0.022 + random.uniform(-0.05, 0.05))
        rl   = _get_risk_level(base)
        ch   = round(max(0, min(1, base + random.uniform(-0.1, 0.1))), 3)
        ty   = round(max(0, min(1, base - 0.05 + random.uniform(-0.1, 0.1))), 3)
        ad   = round(max(0, min(1, base + 0.03 + random.uniform(-0.08, 0.08))), 3)
        demo_preds[district] = {
            "summary": f"{district}: {rl.upper()} risk",
            "predictions": {
                "Cholera": {"max_prob": ch, "risk_level": _get_risk_level(ch),
                            "week_probs": [ch]*4},
                "Typhoid": {"max_prob": ty, "risk_level": _get_risk_level(ty),
                            "week_probs": [ty]*4},
                "ADD":     {"max_prob": ad, "risk_level": _get_risk_level(ad),
                            "week_probs": [ad]*4},
            }
        }

    # ── Render map
    m = render_risk_map(
        predictions       = demo_preds,
        selected_district = "Raigad",
        layer_mode        = "both",
    )

    # ── Save standalone HTML
    out_path = Path("results/hydrocast_map.html")
    save_map_html(m, out_path)

    print(f"\n── Map rendered successfully")
    print(f"── Districts plotted: {len(demo_preds)}")
    print(f"── Map saved: {out_path}")
    print("── Open in browser to view interactive map")
    print("\n✅ map_component.py smoke test passed.")
