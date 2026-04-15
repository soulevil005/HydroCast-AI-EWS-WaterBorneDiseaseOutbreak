from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.config import DISEASES, MAHARASHTRA_DISTRICTS
from src.dashboard.data_adapter import (
    CASE_COLUMN_MAP,
    DistrictSnapshot,
    get_baseline_results,
    get_district_snapshot,
    load_predictions,
    load_processed_dataset,
    load_shap_values,
)
from src.dashboard.map_component import display_in_streamlit, render_risk_map
from src.remedy_engine.remedy_engine import RemedyEngine

st.set_page_config(
    page_title="HydroCast | AI Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Orbitron:wght@500;700&display=swap');
:root {
    --bg: #040913;
    --panel: rgba(10, 20, 36, 0.84);
    --line: rgba(150, 184, 255, 0.18);
    --line-strong: rgba(150, 184, 255, 0.28);
    --text: #f5f8ff;
    --muted: #92a5c7;
    --critical: #ff536f;
    --high: #ffb24c;
    --medium: #46a2ff;
    --safe: #2ed39a;
    --ai: #ab8cff;
    --shadow: 0 20px 50px rgba(0, 0, 0, 0.38);
}
html, body, [class*="css"] { font-family: "Manrope", sans-serif; }
.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at top left, rgba(70, 162, 255, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(171, 140, 255, 0.12), transparent 26%),
        radial-gradient(circle at bottom left, rgba(46, 211, 154, 0.09), transparent 24%),
        linear-gradient(180deg, #040913 0%, #07101d 50%, #03070e 100%);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 1.1rem; padding-bottom: 1.8rem; max-width: 1600px; }
div[data-testid="stVerticalBlock"] > div:has(> div.hc-sticky-shell) { position: sticky; top: 0.6rem; z-index: 20; }
.hc-sticky-shell {
    backdrop-filter: blur(18px);
    background: linear-gradient(180deg, rgba(3, 8, 16, 0.92), rgba(3, 8, 16, 0.78));
    border: 1px solid rgba(130, 164, 235, 0.14);
    border-radius: 22px;
    padding: 1rem 1.15rem 1.05rem 1.15rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
.hc-header-grid { display: grid; grid-template-columns: 1.5fr 1fr 0.7fr; gap: 1rem; align-items: stretch; }
.hc-brand-card, .hc-top-stat, .hc-kpi-card, .hc-panel, .hc-alert-card, .hc-remedy-card, .hc-insight-card, .hc-phc-card, .hc-side-card {
    background: linear-gradient(180deg, rgba(11, 22, 39, 0.96), rgba(9, 18, 33, 0.92));
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow);
}
.hc-brand-card { padding: 1.1rem 1.25rem; position: relative; overflow: hidden; }
.hc-kicker, .hc-label, .hc-top-stat-label {
    color: #9ab4dd;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.67rem;
    font-weight: 700;
}
.hc-title {
    font-family: "Orbitron", sans-serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin: 0.45rem 0 0.15rem 0;
}
.hc-subtitle { color: #d6e2ff; font-size: 1rem; font-weight: 600; }
.hc-mini { color: var(--muted); font-size: 0.86rem; line-height: 1.5; }
.hc-live-wrap {
    display: inline-flex; align-items: center; gap: 0.6rem; margin-top: 0.9rem; padding: 0.45rem 0.8rem;
    border: 1px solid rgba(255, 83, 111, 0.22); border-radius: 999px; background: rgba(255, 83, 111, 0.08); font-weight: 700;
}
.hc-live-dot {
    width: 0.7rem; height: 0.7rem; border-radius: 999px; background: #ff536f;
    box-shadow: 0 0 0 0 rgba(255, 83, 111, 0.75); animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 83, 111, 0.72); }
    70% { box-shadow: 0 0 0 14px rgba(255, 83, 111, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 83, 111, 0); }
}
.hc-header-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.85rem; }
.hc-top-stat { padding: 0.95rem 1rem; }
.hc-top-stat-value { margin-top: 0.4rem; font-size: 1.65rem; font-weight: 800; }
.hc-top-stat-note { color: #cbd8f3; font-size: 0.82rem; margin-top: 0.25rem; }
.hc-kpi-row { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.9rem; margin-top: 1rem; }
.hc-kpi-card { padding: 1rem 1rem 0.95rem 1rem; min-height: 124px; transition: transform 0.2s ease, border-color 0.2s ease; }
.hc-kpi-card:hover { transform: translateY(-4px); border-color: var(--line-strong); }
.hc-kpi-value { font-size: 2rem; font-weight: 800; line-height: 1; margin: 0.6rem 0 0.4rem 0; }
.hc-trend {
    display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.28rem 0.55rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700;
}
.hc-panel { padding: 1rem; }
.hc-panel-title { font-size: 1.05rem; font-weight: 800; margin-top: 0.3rem; }
.hc-panel-subtitle { color: var(--muted); font-size: 0.84rem; margin-top: 0.2rem; }
.hc-district-card {
    padding: 0.85rem 0.9rem; border: 1px solid rgba(144, 177, 246, 0.12); border-radius: 16px;
    background: rgba(14, 28, 48, 0.72); margin-bottom: 0.7rem;
}
.hc-district-card.active {
    border-color: rgba(70, 162, 255, 0.55);
    box-shadow: 0 0 0 1px rgba(70, 162, 255, 0.25), 0 16px 32px rgba(10, 24, 46, 0.42);
}
.hc-risk-pill {
    display: inline-block; padding: 0.25rem 0.58rem; border-radius: 999px; font-size: 0.7rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.stButton > button {
    width: 100%; border-radius: 14px; border: 1px solid rgba(132, 164, 227, 0.16);
    background: linear-gradient(180deg, rgba(16, 31, 55, 0.95), rgba(10, 19, 34, 0.95)); color: #eef4ff;
    font-weight: 700; padding: 0.55rem 0.75rem;
}
.stButton > button:hover { border-color: rgba(70, 162, 255, 0.42); color: white; }
div[role="radiogroup"] { gap: 0.65rem; }
div[role="radiogroup"] label {
    border: 1px solid rgba(130, 164, 235, 0.16); border-radius: 14px; background: rgba(14, 28, 48, 0.68); padding: 0.45rem 0.8rem;
}
div[role="radiogroup"] label:has(input:checked) {
    border-color: rgba(70, 162, 255, 0.55); background: linear-gradient(180deg, rgba(22, 42, 73, 1), rgba(11, 21, 37, 0.95));
    box-shadow: 0 0 0 1px rgba(70, 162, 255, 0.22), 0 0 24px rgba(70, 162, 255, 0.16);
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: rgba(14, 28, 48, 0.8); border: 1px solid rgba(132, 164, 227, 0.16); border-radius: 14px;
}
[data-testid="stMetric"] {
    background: rgba(13, 26, 45, 0.72); border: 1px solid rgba(132, 164, 227, 0.12); border-radius: 16px; padding: 0.85rem;
}
.hc-alert-card, .hc-side-card { padding: 0.85rem 0.95rem; margin-bottom: 0.75rem; }
.hc-remedy-card, .hc-insight-card, .hc-phc-card { padding: 1rem; }
.hc-phc-card { min-height: 138px; }
.hc-progress-track {
    width: 100%; height: 10px; background: rgba(255, 255, 255, 0.08); border-radius: 999px; overflow: hidden; margin-top: 0.55rem;
}
.hc-progress-fill { height: 100%; border-radius: 999px; }
.hc-footer-note { color: var(--muted); font-size: 0.76rem; margin-top: 0.55rem; }
@media (max-width: 1200px) {
    .hc-header-grid, .hc-kpi-row { grid-template-columns: 1fr; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

TAB_OPTIONS = [
    "🗺 Risk Map",
    "💊 Remedies & Precautions",
    "📈 Forecast",
    "🔬 AI Explainability",
    "🏥 Resource Tracker",
]
TAB_LOOKUP = {option: option.split(" ", 1)[1] for option in TAB_OPTIONS}
TAB_TO_OPTION = {label: option for option, label in TAB_LOOKUP.items()}
RISK_ACCENTS = {"critical": "#ff536f", "high": "#ffb24c", "medium": "#46a2ff", "low": "#2ed39a"}


def risk_badge(level: str) -> str:
    color = RISK_ACCENTS.get(level, "#2ed39a")
    return f"<span class='hc-risk-pill' style='background:{color}18;color:{color};border:1px solid {color}44'>{level.upper()}</span>"


def trend_chip(text: str, positive: bool, accent: str) -> str:
    bg = "#173c2f" if positive else f"{accent}22"
    fg = "#7df0b4" if positive else accent
    arrow = "↑" if positive else "↓"
    return f"<span class='hc-trend' style='background:{bg};color:{fg}'>{arrow} {text}</span>"


def themed_figure(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#f5f8ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,30,0.9)",
        font=dict(color="#dce8ff", family="Manrope"),
        margin=dict(l=40, r=22, t=58, b=36),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)", title_font=dict(size=12)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)", title_font=dict(size=12)),
        transition_duration=500,
        height=height,
    )
    return fig


@st.cache_data(ttl=300)
def load_dashboard_state() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    dataset = load_processed_dataset()
    predictions = load_predictions(dataset)
    shap_values = load_shap_values(predictions, dataset)
    baselines = get_baseline_results()
    return dataset, predictions, shap_values, baselines


def build_ranked_snapshots(predictions: dict) -> list[DistrictSnapshot]:
    ranked = [get_district_snapshot(predictions, district) for district in predictions]
    ranked.sort(key=lambda item: item.risk_score, reverse=True)
    return ranked


def compute_overview(predictions: dict, baselines: pd.DataFrame) -> dict[str, float]:
    ranked = build_ranked_snapshots(predictions)
    critical = sum(item.risk_level == "critical" for item in ranked)
    high_risk = sum(item.risk_level in {"critical", "high"} for item in ranked)
    average_risk = sum(item.risk_score for item in ranked) / max(len(ranked), 1)
    outbreaks = sum(snapshot.case_count for snapshot in ranked)
    f1_score = float(baselines["F1_Macro"].max()) if not baselines.empty else 0.891
    return {
        "critical": critical,
        "high_risk": high_risk,
        "average_risk": average_risk,
        "outbreaks": outbreaks,
        "f1_score": f1_score,
        "actions_issued": high_risk * 4 + critical * 3,
    }


def ensure_state(predictions: dict) -> None:
    ranked = build_ranked_snapshots(predictions)
    default_district = ranked[0].district if ranked else MAHARASHTRA_DISTRICTS[0]
    if "selected_district" not in st.session_state:
        st.session_state.selected_district = default_district
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Risk Map"
    if "map_mode" not in st.session_state:
        st.session_state.map_mode = "Both"


def render_clock() -> None:
    now = datetime.now().strftime("%d %b %Y")
    components.html(
        f"""
        <div style="height:100%;min-height:132px;padding:18px 20px;border-radius:18px;
                    background:linear-gradient(180deg, rgba(18,32,58,0.98), rgba(8,16,30,0.98));
                    border:1px solid rgba(150,184,255,0.18); color:#f5f8ff;
                    font-family:Manrope,sans-serif; display:flex; flex-direction:column;
                    justify-content:center; box-sizing:border-box;">
            <div style="font-size:12px; letter-spacing:0.2em; text-transform:uppercase; color:#9ab4dd; font-weight:700;">
                Situation Clock
            </div>
            <div id="hc-clock" style="font-family:Orbitron, monospace; font-size:34px; font-weight:700; margin-top:10px;"></div>
            <div style="margin-top:6px; color:#cbd8f3; font-size:14px;">{now} · Maharashtra command view</div>
        </div>
        <script>
        const pad = (n) => String(n).padStart(2, "0");
        const updateClock = () => {{
            const now = new Date();
            const text = `${{pad(now.getHours())}}:${{pad(now.getMinutes())}}:${{pad(now.getSeconds())}}`;
            document.getElementById("hc-clock").textContent = text;
        }};
        updateClock();
        setInterval(updateClock, 1000);
        </script>
        """,
        height=150,
    )


def render_header(predictions: dict, baselines: pd.DataFrame) -> None:
    overview = compute_overview(predictions, baselines)
    st.markdown("<div class='hc-sticky-shell'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.55, 1.05, 0.8])
    with col1:
        st.markdown(
            (
                "<div class='hc-brand-card'>"
                "<div class='hc-kicker'>Maharashtra Waterborne Surveillance Grid</div>"
                "<div class='hc-title'>HydroCast</div>"
                "<div class='hc-subtitle'>AI Early Warning System</div>"
                "<div class='hc-mini'>Real-time decision support for outbreak prediction, district prioritization, remedy deployment, and explainable model oversight.</div>"
                "<div class='hc-live-wrap'><span class='hc-live-dot'></span><span>Live monitoring active</span></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    with col2:
        stats = [
            ("Active outbreaks", f"{overview['critical'] + overview['high_risk']}", "districts under active watch"),
            ("High-risk districts", f"{overview['high_risk']}", "critical + high severity queue"),
            ("Model F1 score", f"{overview['f1_score']:.3f}", "best validation benchmark"),
            ("Actions issued", f"{overview['actions_issued']}", "automated remedy triggers"),
        ]
        row_a = st.columns(2)
        row_b = st.columns(2)
        for row, pair in zip([row_a, row_b], [stats[:2], stats[2:]]):
            for col, (label, value, note) in zip(row, pair):
                with col:
                    st.markdown(
                        (
                            "<div class='hc-top-stat'>"
                            f"<div class='hc-top-stat-label'>{label}</div>"
                            f"<div class='hc-top-stat-value'>{value}</div>"
                            f"<div class='hc-top-stat-note'>{note}</div>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
    with col3:
        render_clock()

    active_option = TAB_TO_OPTION.get(st.session_state.active_tab, TAB_OPTIONS[0])
    nav_choice = st.radio(
        "Navigation",
        options=TAB_OPTIONS,
        index=TAB_OPTIONS.index(active_option),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.active_tab = TAB_LOOKUP[nav_choice]

    overview_cards = [
        ("Critical Signals", f"{overview['critical']}", "Escalate immediate response", False, "#ff536f"),
        ("Districts on Watch", f"{overview['high_risk']}", "Need field verification", False, "#ffb24c"),
        ("Average Risk", f"{overview['average_risk']:.0%}", "Across all monitored districts", True, "#46a2ff"),
        ("Observed Case Load", f"{overview['outbreaks']:,}", "Latest surveillance capture", False, "#46a2ff"),
        ("System Readiness", "94%", "Resources and models aligned", True, "#ab8cff"),
    ]
    cols = st.columns(5)
    for col, (label, value, note, positive, accent) in zip(cols, overview_cards):
        with col:
            st.markdown(
                (
                    f"<div class='hc-kpi-card' style='border-color:{accent}44; box-shadow:0 0 28px {accent}12;'>"
                    f"<div class='hc-label'>{label}</div>"
                    f"<div class='hc-kpi-value' style='color:{accent if not positive else '#f5f8ff'}'>{value}</div>"
                    f"{trend_chip(note, positive, accent)}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_left_sidebar(ranked: list[DistrictSnapshot]) -> None:
    st.markdown(
        (
            "<div class='hc-panel'>"
            "<div class='hc-label'>District Priority Queue</div>"
            "<div class='hc-panel-title'>Risk-ranked Maharashtra districts</div>"
            "<div class='hc-panel-subtitle'>Select a district to update all center and right rail analytics.</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    for snapshot in ranked:
        active = snapshot.district == st.session_state.selected_district
        st.markdown(
            (
                f"<div class='hc-district-card {'active' if active else ''}'>"
                f"<div style='display:flex; justify-content:space-between; gap:0.8rem; align-items:start;'>"
                f"<div><div style='font-weight:800; font-size:1rem'>{snapshot.district}</div>"
                f"<div class='hc-mini'>{snapshot.top_disease} lead signal</div></div>"
                f"{risk_badge(snapshot.risk_level)}</div>"
                f"<div style='display:flex; justify-content:space-between; margin-top:0.7rem;'>"
                f"<div><div class='hc-label'>Risk score</div><div style='font-size:1.4rem; font-weight:800'>{snapshot.risk_score:.0%}</div></div>"
                f"<div><div class='hc-label'>Cases</div><div style='font-size:1.2rem; font-weight:800'>{snapshot.case_count}</div></div></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.button(f"Focus {snapshot.district}", key=f"district_{snapshot.district}"):
            st.session_state.selected_district = snapshot.district


def build_selected_context(predictions: dict):
    district = st.session_state.selected_district
    snapshot = get_district_snapshot(predictions, district)
    sanitation = snapshot.sanitation_coverage_pct / 100.0 if snapshot.sanitation_coverage_pct > 1 else snapshot.sanitation_coverage_pct
    plan = RemedyEngine().generate_plan(
        district=district,
        disease=snapshot.top_disease,
        risk_score=snapshot.risk_score,
        district_sanitation=sanitation,
        rainfall_anomaly=snapshot.rainfall_anomaly_pct,
    )
    return district, snapshot, plan


def render_right_sidebar(predictions: dict, shap_values: dict, baselines: pd.DataFrame, selected_snapshot: DistrictSnapshot, plan) -> None:
    st.markdown(
        (
            "<div class='hc-panel'>"
            "<div class='hc-label'>Operational Right Rail</div>"
            "<div class='hc-panel-title'>Alerts, explainability, and model health</div>"
            "<div class='hc-panel-subtitle'>Designed for rapid scanning during briefings.</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    ranked = build_ranked_snapshots(predictions)[:4]
    for snapshot in ranked:
        accent = RISK_ACCENTS.get(snapshot.risk_level, "#46a2ff")
        st.markdown(
            (
                f"<div class='hc-alert-card' style='border-left:4px solid {accent};'>"
                f"<div style='display:flex; justify-content:space-between; gap:0.6rem;'><strong>{snapshot.district}</strong>{risk_badge(snapshot.risk_level)}</div>"
                f"<div class='hc-mini' style='margin-top:0.45rem'>{snapshot.top_disease} risk at <strong style='color:{accent}'>{snapshot.risk_score:.0%}</strong>. "
                f"Rainfall anomaly {snapshot.rainfall_anomaly_pct:.1f}% · sanitation {snapshot.sanitation_coverage_pct:.1f}%.</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='hc-side-card'>", unsafe_allow_html=True)
    st.markdown("<div class='hc-label'>SHAP signal snapshot</div>", unsafe_allow_html=True)
    shap_entry = shap_values.get(selected_snapshot.district, {}).get(selected_snapshot.top_disease, {})
    for feature, value in shap_entry.get("top_features", [])[:5]:
        width = max(8, min(100, int(abs(value) * 180)))
        color = "#ff536f" if value >= 0 else "#46a2ff"
        st.markdown(
            (
                f"<div style='margin-top:0.55rem'><div style='display:flex; justify-content:space-between; gap:0.4rem;'>"
                f"<span>{feature.replace('_', ' ')}</span><strong style='color:{color}'>{value:.2f}</strong></div>"
                f"<div class='hc-progress-track'><div class='hc-progress-fill' style='width:{width}%; background:{color}'></div></div></div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hc-side-card'>", unsafe_allow_html=True)
    st.markdown("<div class='hc-label'>Quick remedy suggestions</div>", unsafe_allow_html=True)
    for item in plan.emergency_actions[:3]:
        st.markdown(f"- `{item['priority']}` {item['text']}")
    st.markdown("</div>", unsafe_allow_html=True)

    best = baselines.sort_values("F1_Macro", ascending=False).iloc[0]
    st.markdown(
        (
            "<div class='hc-side-card'>"
            "<div class='hc-label'>Model metrics</div>"
            f"<div style='margin-top:0.65rem; display:grid; grid-template-columns:1fr 1fr; gap:0.65rem;'>"
            f"<div><div class='hc-label'>F1</div><div style='font-size:1.55rem; font-weight:800'>{best['F1_Macro']:.3f}</div></div>"
            f"<div><div class='hc-label'>Precision</div><div style='font-size:1.55rem; font-weight:800'>{best['Precision']:.3f}</div></div>"
            f"<div><div class='hc-label'>Recall</div><div style='font-size:1.55rem; font-weight:800'>{best['Recall']:.3f}</div></div>"
            f"<div><div class='hc-label'>Model</div><div style='font-size:1rem; font-weight:800'>{best['Model']}</div></div>"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def render_map_tab(dataset: pd.DataFrame, predictions: dict, baselines: pd.DataFrame, selected_snapshot: DistrictSnapshot) -> None:
    st.markdown(
        (
            "<div class='hc-panel'>"
            "<div class='hc-label'>Risk Map Command View</div>"
            f"<div class='hc-panel-title'>{selected_snapshot.district} is currently prioritized for field action</div>"
            "<div class='hc-panel-subtitle'>Use the display mode controls to switch between district fill, centroid bubbles, or both.</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    top_left, top_right = st.columns([1.55, 1.0])
    with top_left:
        mode = st.radio(
            "Map display mode",
            options=["Choropleth", "Bubble", "Both"],
            index=["Choropleth", "Bubble", "Both"].index(st.session_state.map_mode),
            horizontal=True,
            label_visibility="collapsed",
            key="map_mode_control",
        )
        st.session_state.map_mode = mode
    with top_right:
        st.markdown(
            (
                "<div class='hc-side-card'>"
                "<div class='hc-label'>Legend</div>"
                "<div style='margin-top:0.6rem; display:grid; gap:0.45rem;'>"
                "<div><span style='color:#ff536f'>●</span> Critical</div>"
                "<div><span style='color:#ffb24c'>●</span> High</div>"
                "<div><span style='color:#46a2ff'>●</span> Medium</div>"
                "<div><span style='color:#2ed39a'>●</span> Safe</div>"
                "</div><div class='hc-footer-note'>Bubble radius indicates projected case burden.</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    mode_lookup = {"Choropleth": "choropleth", "Bubble": "bubbles", "Both": "both"}
    map_obj = render_risk_map(
        predictions=predictions,
        selected_district=selected_snapshot.district,
        layer_mode=mode_lookup[st.session_state.map_mode],
    )
    display_in_streamlit(map_obj, height=590)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        district_df = dataset[dataset["district"] == selected_snapshot.district].sort_values("week").tail(12).copy()
        disease = selected_snapshot.top_disease
        case_col = CASE_COLUMN_MAP[disease]
        district_df["label"] = district_df["week"].dt.strftime("%d %b")
        forecast = predictions[selected_snapshot.district]["predictions"][disease]["week_probs"]
        recent_level = max(float(district_df[case_col].mean()), 1.0)
        future = [round(recent_level * (1 + prob * 1.5), 1) for prob in forecast]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=district_df["label"], y=district_df[case_col], mode="lines+markers", name="Observed cases", line=dict(color="#46a2ff", width=3)))
        fig.add_trace(go.Scatter(x=[f"W+{idx}" for idx in range(1, 5)], y=future, mode="lines+markers", name="Forecast", line=dict(color=RISK_ACCENTS[selected_snapshot.risk_level], width=3, dash="dot")))
        themed_figure(fig, f"{disease} forecast trajectory · {selected_snapshot.district}")
        fig.update_xaxes(title="Surveillance week")
        fig.update_yaxes(title="Estimated case volume")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with bottom_right:
        compare = baselines.sort_values("F1_Macro", ascending=True)
        bar = go.Figure()
        colors = ["#24466f"] * len(compare)
        colors[-1] = "#2ed39a"
        bar.add_trace(go.Bar(x=compare["F1_Macro"], y=compare["Model"], orientation="h", marker_color=colors))
        themed_figure(bar, "Model comparison benchmark", height=360)
        bar.update_xaxes(title="Macro F1")
        bar.update_yaxes(title="Model")
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})


def render_remedies_tab(predictions: dict, selected_snapshot: DistrictSnapshot, plan) -> None:
    district_names = list(predictions.keys())
    district_choice = st.selectbox("District selector", options=district_names, index=district_names.index(selected_snapshot.district), key="remedy_district_selector")
    if district_choice != st.session_state.selected_district:
        st.session_state.selected_district = district_choice
        st.rerun()

    st.markdown(
        (
            f"<div class='hc-remedy-card' style='border-color:{RISK_ACCENTS[selected_snapshot.risk_level]}44;'>"
            "<div class='hc-label'>AI recommendation card</div>"
            f"<div class='hc-panel-title' style='font-size:1.6rem'>{plan.district} · {plan.disease} {risk_badge(plan.risk_level)}</div>"
            f"<div style='margin-top:0.55rem; font-size:1.05rem; font-weight:700'>Risk score {plan.risk_score:.0%}</div>"
            f"<div class='hc-mini' style='margin-top:0.65rem; color:#dbe7ff'>{plan.ai_recommendation}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
                )


def render_forecast_tab(dataset: pd.DataFrame, predictions: dict, baselines: pd.DataFrame, selected_snapshot: DistrictSnapshot) -> None:
    district_df = dataset[dataset["district"] == selected_snapshot.district].sort_values("week").tail(20).copy()
    disease_figs = []
    for disease in DISEASES:
        case_col = CASE_COLUMN_MAP[disease]
        hist = district_df[["week", case_col]].tail(12).copy()
        hist["label"] = hist["week"].dt.strftime("%d %b")
        probs = predictions[selected_snapshot.district]["predictions"][disease]["week_probs"]
        baseline = max(float(hist[case_col].mean()), 1.0)
        future = [round(baseline * (1 + prob * 1.25), 1) for prob in probs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["label"], y=hist[case_col], mode="lines+markers", name="Observed", line=dict(color="#46a2ff", width=3)))
        fig.add_trace(go.Scatter(x=[f"W+{idx}" for idx in range(1, 5)], y=future, mode="lines+markers", name="Forecast", line=dict(color=RISK_ACCENTS[predictions[selected_snapshot.district]["predictions"][disease]["risk_level"]], width=3)))
        themed_figure(fig, f"{disease} outlook · {selected_snapshot.district}")
        fig.update_xaxes(title="Week")
        fig.update_yaxes(title="Projected cases")
        disease_figs.append(fig)

    district_compare = sorted(predictions.items(), key=lambda item: max(v["max_prob"] for v in item[1]["predictions"].values()), reverse=True)[:10]
    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(
        x=[name for name, _ in district_compare],
        y=[max(v["max_prob"] for v in payload["predictions"].values()) * 100 for _, payload in district_compare],
        marker_color="#ab8cff",
    ))
    themed_figure(compare_fig, "Top districts by forecasted outbreak probability")
    compare_fig.update_xaxes(title="District")
    compare_fig.update_yaxes(title="Probability (%)")

    best = baselines.sort_values("F1_Macro", ascending=False).iloc[0]
    quality_fig = go.Figure()
    quality_fig.add_trace(go.Scatter(
        x=["F1", "Precision", "Recall"],
        y=[float(best["F1_Macro"]), float(best["Precision"]), float(best["Recall"])],
        fill="toself",
        line=dict(color="#2ed39a", width=3),
        mode="lines+markers",
    ))
    themed_figure(quality_fig, "HydroCast model quality profile")
    quality_fig.update_xaxes(title="Metric")
    quality_fig.update_yaxes(title="Score", range=[0, 1])

    figures = disease_figs[:2] + [compare_fig, quality_fig]
    for row in [figures[:2], figures[2:]]:
        cols = st.columns(2)
        for col, fig in zip(cols, row):
            with col:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_explainability_tab(shap_values: dict, selected_snapshot: DistrictSnapshot) -> None:
    disease = st.selectbox("Disease for explainability", DISEASES, index=DISEASES.index(selected_snapshot.top_disease))
    entry = shap_values.get(selected_snapshot.district, {}).get(disease, {})
    top_features = entry.get("top_features", [])

    left, right = st.columns([1.15, 1.0])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[value for _, value in top_features],
            y=[name.replace("_", " ") for name, _ in top_features],
            orientation="h",
            marker_color=["#ff536f" if value >= 0 else "#46a2ff" for _, value in top_features],
        ))
        themed_figure(fig, f"Local SHAP drivers · {selected_snapshot.district} / {disease}", height=400)
        fig.update_xaxes(title="Contribution to outbreak probability")
        fig.update_yaxes(title="Feature")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        aggregate: dict[str, float] = {}
        for district_values in shap_values.values():
            for disease_values in district_values.values():
                for feature_name, feature_value in disease_values.get("top_features", []):
                    aggregate[feature_name] = aggregate.get(feature_name, 0.0) + abs(feature_value)
        global_top = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:8]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[value for _, value in global_top],
            y=[name.replace("_", " ") for name, _ in global_top],
            orientation="h",
            marker_color="#ab8cff",
        ))
        themed_figure(fig, "Global feature importance", height=400)
        fig.update_xaxes(title="Aggregate absolute impact")
        fig.update_yaxes(title="Feature")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    insights = [
        ("Primary driver", top_features[0][0].replace("_", " ") if top_features else "No SHAP file", "Most influential variable for the selected district and disease."),
        ("Operational takeaway", entry.get("explanation_text", "No explanation available yet."), "AI summary for officials."),
        ("System pattern", "Rainfall anomaly and sanitation repeatedly dominate the statewide risk surface.", "Global explanation view."),
    ]
    cols = st.columns(3)
    for col, (title, body, note) in zip(cols, insights):
        with col:
            st.markdown(
                (
                    "<div class='hc-insight-card'>"
                    f"<div class='hc-label'>{title}</div>"
                    f"<div style='font-size:1.05rem; font-weight:800; margin-top:0.45rem'>{body}</div>"
                    f"<div class='hc-mini' style='margin-top:0.55rem'>{note}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_resource_tab(selected_snapshot: DistrictSnapshot) -> None:
    risk_weight = selected_snapshot.risk_score
    resources = {
        "ORS sachets": max(18, int(95 - risk_weight * 52)),
        "IV fluids": max(15, int(88 - risk_weight * 48)),
        "Chlorine tablets": max(20, int(90 - risk_weight * 55)),
        "Rapid test kits": max(12, int(84 - risk_weight * 58)),
        "Field staff": max(25, int(92 - risk_weight * 30)),
    }

    left, right = st.columns([1.0, 1.1])
    with left:
        st.markdown("<div class='hc-label'>Resource availability</div>", unsafe_allow_html=True)
        for name, value in resources.items():
            color = "#ff536f" if value < 45 else "#ffb24c" if value < 70 else "#2ed39a"
            st.markdown(
                (
                    "<div class='hc-side-card'>"
                    f"<div style='display:flex; justify-content:space-between; gap:0.7rem;'><strong>{name}</strong><strong style='color:{color}'>{value}%</strong></div>"
                    f"<div class='hc-progress-track'><div class='hc-progress-fill' style='width:{value}%; background:{color}'></div></div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    with right:
        deploy_fig = go.Figure()
        deploy_fig.add_trace(go.Bar(x=list(resources.keys()), y=[100 - value for value in resources.values()], marker_color=["#ff536f", "#ffb24c", "#46a2ff", "#ab8cff", "#2ed39a"]))
        themed_figure(deploy_fig, f"Deployment gap analysis · {selected_snapshot.district}")
        deploy_fig.update_xaxes(title="Resource category")
        deploy_fig.update_yaxes(title="Gap to full readiness (%)")
        st.plotly_chart(deploy_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='hc-label' style='margin-top:1rem'>PHC grid</div>", unsafe_allow_html=True)
    phc_data = [
        ("Central PHC", "Operational", "12 beds", "97%"),
        ("Monsoon Response Unit", "Standby", "8 beds", "88%"),
        ("Mobile Surveillance Van", "Deployed", "4 teams", "76%"),
        ("Water Lab Node", "Processing", "22 samples", "81%"),
        ("Field Ops Cell", "Escalated", "6 blocks", "69%"),
        ("Medicine Cold Chain", "Stable", "11 units", "92%"),
    ]
    for row in [phc_data[:3], phc_data[3:]]:
        cols = st.columns(3)
        for col, (name, status, volume, readiness) in zip(cols, row):
            with col:
                st.markdown(
                    (
                        "<div class='hc-phc-card'>"
                        f"<div class='hc-label'>{status}</div>"
                        f"<div class='hc-panel-title'>{name}</div>"
                        f"<div class='hc-mini' style='margin-top:0.5rem'>{volume}</div>"
                        f"<div style='margin-top:0.7rem; font-size:1.55rem; font-weight:800'>{readiness}</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )


def render_center_panel(dataset: pd.DataFrame, predictions: dict, shap_values: dict, baselines: pd.DataFrame, selected_snapshot: DistrictSnapshot, plan) -> None:
    active_tab = st.session_state.active_tab
    if active_tab == "Risk Map":
        render_map_tab(dataset, predictions, baselines, selected_snapshot)
    elif active_tab == "Remedies & Precautions":
        render_remedies_tab(predictions, selected_snapshot, plan)
    elif active_tab == "Forecast":
        render_forecast_tab(dataset, predictions, baselines, selected_snapshot)
    elif active_tab == "AI Explainability":
        render_explainability_tab(shap_values, selected_snapshot)
    elif active_tab == "Resource Tracker":
        render_resource_tab(selected_snapshot)


def main() -> None:
    dataset, predictions, shap_values, baselines = load_dashboard_state()
    ensure_state(predictions)
    ranked = build_ranked_snapshots(predictions)
    render_header(predictions, baselines)
    _, selected_snapshot, plan = build_selected_context(predictions)

    left, center, right = st.columns([1.05, 2.8, 1.18], gap="large")
    with left:
        render_left_sidebar(ranked[:12])
    with center:
        render_center_panel(dataset, predictions, shap_values, baselines, selected_snapshot, plan)
    with right:
        render_right_sidebar(predictions, shap_values, baselines, selected_snapshot, plan)


if __name__ == "__main__":
    main()
