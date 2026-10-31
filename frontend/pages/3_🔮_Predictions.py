"""Predictive maintenance analytics and prognostics page.

Displays deep learning model inference including autoencoder anomaly detection,
multi-class fault classification probabilities, Remaining Useful Life (RUL) estimation,
and health degradation trajectories.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from typing import Any, Dict, Tuple

import pandas as pd
import requests
import streamlit as st

from components.charts import create_bar_chart, create_gauge_chart, create_time_series_chart
from components.sidebar import render_sidebar

# Page Configuration
st.set_page_config(page_title="Predictions | AI Maintenance", page_icon="🔮", layout="wide")

# Load CSS
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar
selected_machine, api_url, is_online = render_sidebar()

# Page Header
st.title("🔮 Predictive Analytics & Health Prognostics")
st.markdown(
    f"Inference outputs from deep learning autoencoders, multi-class classifiers, "
    f"and RUL degradation models for **{selected_machine}**."
)


def get_local_demo_predictions(machine_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Generate realistic predictive diagnostics when backend is offline."""
    configs = {
        "MCH-001": {
            "anomaly": {"is_anomaly": False, "anomaly_score": 0.12, "details": "All sensors within normal limits."},
            "fault": {
                "predicted_fault": "Normal Operation",
                "confidence": 0.94,
                "probabilities": {
                    "Normal Operation": 0.94,
                    "Bearing Degradation": 0.03,
                    "Cavitation": 0.01,
                    "Motor Overheating": 0.01,
                    "Shaft Misalignment": 0.01,
                },
            },
            "rul": {"estimated_rul": 680.0, "unit": "operating hours", "confidence_interval": {"lower": 620.0, "upper": 740.0}},
            "base_health": 95.0,
            "slope": -0.02,
        },
        "MCH-002": {
            "anomaly": {"is_anomaly": True, "anomaly_score": 0.78, "details": "Critical vibration (4.82 mm/s RMS); High acoustic noise (86.5 dB)."},
            "fault": {
                "predicted_fault": "Bearing Degradation",
                "confidence": 0.88,
                "probabilities": {
                    "Normal Operation": 0.04,
                    "Bearing Degradation": 0.88,
                    "Cavitation": 0.02,
                    "Motor Overheating": 0.03,
                    "Shaft Misalignment": 0.03,
                },
            },
            "rul": {"estimated_rul": 140.0, "unit": "operating hours", "confidence_interval": {"lower": 115.0, "upper": 165.0}},
            "base_health": 68.0,
            "slope": -0.15,
        },
        "MCH-003": {
            "anomaly": {"is_anomaly": True, "anomaly_score": 0.82, "details": "Severe discharge pressure drop (3.1 bar); High cavitation noise (89.0 dB)."},
            "fault": {
                "predicted_fault": "Cavitation / Flow Instability",
                "confidence": 0.82,
                "probabilities": {
                    "Normal Operation": 0.05,
                    "Bearing Degradation": 0.08,
                    "Cavitation / Flow Instability": 0.82,
                    "Motor Overheating": 0.02,
                    "Shaft Misalignment": 0.03,
                },
            },
            "rul": {"estimated_rul": 95.0, "unit": "operating hours", "confidence_interval": {"lower": 75.0, "upper": 115.0}},
            "base_health": 54.0,
            "slope": -0.22,
        },
        "MCH-004": {
            "anomaly": {"is_anomaly": True, "anomaly_score": 0.86, "details": "Thermal alarm: Stator housing temperature 98.2 °C; High current draw."},
            "fault": {
                "predicted_fault": "Motor Overheating",
                "confidence": 0.89,
                "probabilities": {
                    "Normal Operation": 0.02,
                    "Bearing Degradation": 0.03,
                    "Cavitation": 0.01,
                    "Motor Overheating": 0.89,
                    "Shaft Misalignment": 0.05,
                },
            },
            "rul": {"estimated_rul": 62.0, "unit": "operating hours", "confidence_interval": {"lower": 48.0, "upper": 78.0}},
            "base_health": 42.0,
            "slope": -0.28,
        },
        "MCH-005": {
            "anomaly": {"is_anomaly": False, "anomaly_score": 0.08, "details": "Nominal spindle performance."},
            "fault": {
                "predicted_fault": "Normal Operation",
                "confidence": 0.97,
                "probabilities": {
                    "Normal Operation": 0.97,
                    "Bearing Degradation": 0.01,
                    "Cavitation": 0.01,
                    "Motor Overheating": 0.005,
                    "Shaft Misalignment": 0.005,
                },
            },
            "rul": {"estimated_rul": 710.0, "unit": "operating hours", "confidence_interval": {"lower": 650.0, "upper": 760.0}},
            "base_health": 98.0,
            "slope": -0.01,
        },
    }
    cfg = configs.get(machine_id, configs["MCH-001"])

    now = datetime.now(timezone.utc)
    rows = []
    for i in range(50, -1, -1):
        ts = now - timedelta(hours=i * 2)
        health = max(10.0, round(cfg["base_health"] + (50 - i) * cfg["slope"] + random.gauss(0, 0.4), 1))
        rows.append({"timestamp": ts.strftime("%m-%d %H:%M"), "health_score": health})

    return cfg["anomaly"], cfg["fault"], cfg["rul"], pd.DataFrame(rows)


def fetch_all_predictions(
    api_url: str, machine_id: str, is_online: bool
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Retrieve predictions from backend API or local fallback."""
    if is_online:
        try:
            req_data = {"machine_id": machine_id}
            anom_res = requests.post(f"{api_url.rstrip('/')}/api/predict/anomaly", json=req_data, timeout=3.0)
            fault_res = requests.post(f"{api_url.rstrip('/')}/api/predict/fault", json=req_data, timeout=3.0)
            rul_res = requests.post(f"{api_url.rstrip('/')}/api/predict/rul", json=req_data, timeout=3.0)
            hist_res = requests.get(f"{api_url.rstrip('/')}/api/sensors/history/{machine_id}?limit=50", timeout=3.0)

            if anom_res.status_code == 200 and fault_res.status_code == 200 and rul_res.status_code == 200:
                hist_data = hist_res.json() if hist_res.status_code == 200 else []
                if hist_data:
                    df = pd.DataFrame(hist_data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
                else:
                    _, _, _, df = get_local_demo_predictions(machine_id)

                return anom_res.json(), fault_res.json(), rul_res.json(), df
        except Exception:
            pass

    return get_local_demo_predictions(machine_id)


anomaly, fault, rul, df_trend = fetch_all_predictions(api_url, selected_machine, is_online)

# --- 3-Column Predictive Layout ---
col1, col2, col3 = st.columns(3)

# Column 1: Anomaly Detection
with col1:
    st.markdown("### 🚨 Anomaly Detection")
    is_anom = anomaly.get("is_anomaly", False)
    score = anomaly.get("anomaly_score", 0.0)

    if is_anom:
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #f87171;">⚠️ ANOMALY DETECTED</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 6px 0;">{score:.2f} <span style="font-size: 1rem; color: #94a3b8;">/ 1.00</span></div>
                <div style="font-size: 0.85rem; color: #fca5a5; line-height: 1.4;">{anomaly.get('details', 'Exceeds threshold limits.')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #34d399;">✅ NORMAL OPERATION</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 6px 0;">{score:.2f} <span style="font-size: 1rem; color: #94a3b8;">/ 1.00</span></div>
                <div style="font-size: 0.85rem; color: #a7f3d0; line-height: 1.4;">{anomaly.get('details', 'All sensors within ISO standard envelope.')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Column 2: Fault Classification
with col2:
    st.markdown("### 🔬 Fault Classification")
    probs = fault.get("probabilities", {})
    categories = list(probs.keys())
    values = [round(v * 100, 1) for v in probs.values()]

    fig_bar = create_bar_chart(
        categories=categories,
        values=values,
        title="Fault Class Probabilities (%)",
        horizontal=True,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(f"Top Diagnostic: **{fault.get('predicted_fault')}** ({fault.get('confidence', 0.0) * 100:.1f}% Confidence)")

# Column 3: RUL Estimation
with col3:
    st.markdown("### ⏳ Remaining Useful Life (RUL)")
    rul_hours = rul.get("estimated_rul", 720.0)
    rul_percent = min(100.0, round((rul_hours / 720.0) * 100, 1))

    fig_rul = create_gauge_chart(
        value=rul_percent,
        title="Remaining Life %",
        min_val=0.0,
        max_val=100.0,
        thresholds={"danger": 25.0, "warning": 50.0},
    )
    st.plotly_chart(fig_rul, use_container_width=True)

    ci = rul.get("confidence_interval", {})
    lower = ci.get("lower", rul_hours * 0.85)
    upper = ci.get("upper", rul_hours * 1.15)
    st.metric(
        label="Estimated Remaining Operational Life",
        value=f"{rul_hours:.0f} Hours ({rul_hours / 24:.1f} Days)",
        delta=f"90% CI: [{lower:.0f} - {upper:.0f} hrs]",
    )

st.markdown("---")

# Degradation Trend Chart
st.subheader("📉 Machine Health Degradation Trajectory")
fig_degrad = create_time_series_chart(
    df=df_trend,
    x_col="timestamp",
    y_cols=["health_score"],
    title="Historical Health Score (%) Degradation Path",
    unit="%",
)
st.plotly_chart(fig_degrad, use_container_width=True)

# Maintenance Recommendation Box
st.subheader("📋 Prescriptive Maintenance Recommendation")

if rul_hours < 100.0:
    st.markdown(
        """
        <div style="background: rgba(239, 68, 68, 0.12); border-left: 5px solid #ef4444; border-radius: 6px; padding: 18px 22px;">
            <h4 style="color: #ef4444; margin: 0 0 8px 0;">⚡ URGENT ACTION REQUIRED (< 100 Hours Remaining)</h4>
            <p style="color: #fca5a5; margin: 0 0 8px 0; font-size: 0.95rem;">
                Machine is operating in late-stage degradation (Zone D). High probability of functional failure within 4 to 6 operational shifts.
            </p>
            <ul style="color: #e2e8f0; font-size: 0.9rem; margin: 0; padding-left: 20px;">
                <li>Immediately dispatch predictive maintenance technician to verify vibration spectra (1X, 2X, BPFI).</li>
                <li>Verify spare part inventory for bearing assembly and mechanical seals.</li>
                <li>Schedule emergency planned overhaul during the next available shift change.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif rul_hours < 300.0:
    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.12); border-left: 5px solid #f59e0b; border-radius: 6px; padding: 18px 22px;">
            <h4 style="color: #f59e0b; margin: 0 0 8px 0;">⚠️ SCHEDULE MAINTENANCE TURNAROUND (100 - 300 Hours Remaining)</h4>
            <p style="color: #fde68a; margin: 0 0 8px 0; font-size: 0.95rem;">
                Early mechanical wear detected. Degradation trajectory indicates planned intervention recommended within 7 to 12 days.
            </p>
            <ul style="color: #e2e8f0; font-size: 0.9rem; margin: 0; padding-left: 20px;">
                <li>Perform acoustic lubrication test and grease replenishment.</li>
                <li>Inspect shaft alignment and torque motor mounting bolts.</li>
                <li>Add asset to high-frequency monitoring watchlist.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="background: rgba(16, 185, 129, 0.12); border-left: 5px solid #10b981; border-radius: 6px; padding: 18px 22px;">
            <h4 style="color: #10b981; margin: 0 0 8px 0;">✅ HEALTHY OPERATING ENVELOPE (> 300 Hours Remaining)</h4>
            <p style="color: #a7f3d0; margin: 0; font-size: 0.95rem;">
                Asset health is satisfactory. All components exhibit low wear rates. Continue standard preventive maintenance schedule and continuous monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
