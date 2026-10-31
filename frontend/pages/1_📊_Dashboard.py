"""Real-time industrial sensor telemetry monitoring dashboard.

Displays live operational metrics, health indices, ISO-10816 anomaly alerts,
and dynamic time-series charts with auto-refresh capability and local demo fallback.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

from components.charts import create_gauge_chart, create_time_series_chart
from components.sidebar import render_sidebar

# Page Configuration
st.set_page_config(page_title="Dashboard | AI Maintenance", page_icon="📊", layout="wide")

# Load CSS
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
selected_machine, api_url, is_online = render_sidebar()

# Page Title & Description
st.title("📊 Real-Time Sensor Telemetry Dashboard")
st.markdown(
    f"Continuous live telemetry monitoring and ISO-10816 threshold surveillance for **{selected_machine}**."
)


def generate_local_demo_data(machine_id: str, count: int = 60) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate realistic synthetic telemetry when API is unreachable."""
    now = datetime.now(timezone.utc)
    base_offsets = {
        "MCH-001": {"temp": 68.0, "vib": 1.4, "press": 5.2, "health": 95.0, "rpm": 2980.0, "noise": 71.0},
        "MCH-002": {"temp": 79.0, "vib": 4.6, "press": 4.8, "health": 65.0, "rpm": 2940.0, "noise": 86.0},
        "MCH-003": {"temp": 74.0, "vib": 3.8, "press": 3.2, "health": 55.0, "rpm": 2890.0, "noise": 88.0},
        "MCH-004": {"temp": 97.0, "vib": 2.5, "press": 6.7, "health": 44.0, "rpm": 3040.0, "noise": 80.0},
        "MCH-005": {"temp": 64.0, "vib": 1.1, "press": 5.0, "health": 98.0, "rpm": 3010.0, "noise": 69.0},
    }
    cfg = base_offsets.get(machine_id, base_offsets["MCH-001"])

    rows = []
    for i in range(count, -1, -1):
        ts = now - timedelta(seconds=i * 5)
        temp = round(cfg["temp"] + random.gauss(0, 0.7), 2)
        vib = round(max(0.2, cfg["vib"] + random.gauss(0, 0.18)), 3)
        press = round(max(0.5, cfg["press"] + random.gauss(0, 0.12)), 2)
        rpm = round(cfg["rpm"] + random.gauss(0, 10.0), 1)
        noise = round(cfg["noise"] + random.gauss(0, 0.8), 1)
        health = round(max(5.0, min(100.0, cfg["health"] + random.gauss(0, 0.4))), 1)

        rows.append({
            "timestamp": ts.strftime("%H:%M:%S"),
            "temperature": temp,
            "vibration": vib,
            "pressure": press,
            "rpm": rpm,
            "noise_db": noise,
            "health_score": health,
        })

    df = pd.DataFrame(rows)
    latest = rows[-1]
    return df, latest


def fetch_telemetry(api_url: str, machine_id: str, is_online: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Retrieve telemetry from API or fallback to local generator."""
    if is_online:
        try:
            hist_res = requests.get(f"{api_url.rstrip('/')}/api/sensors/history/{machine_id}?limit=60", timeout=1.8)
            if hist_res.status_code == 200:
                data = hist_res.json()
                if data:
                    df = pd.DataFrame(data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
                    latest = data[-1]
                    return df, latest
        except Exception:
            pass

    return generate_local_demo_data(machine_id)


# Fetch Data
df_history, latest = fetch_telemetry(api_url, selected_machine, is_online)

# Calculate Deltas
prev_row = df_history.iloc[-2] if len(df_history) >= 2 else df_history.iloc[-1]
delta_temp = round(latest["temperature"] - prev_row["temperature"], 2)
delta_vib = round(latest["vibration"] - prev_row["vibration"], 3)
delta_press = round(latest["pressure"] - prev_row["pressure"], 2)
delta_health = round(latest["health_score"] - prev_row["health_score"], 1)

# Metric Cards Row
st.subheader("⚡ Operational Telemetry Snapshot")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🌡️ Temperature",
        value=f"{latest['temperature']:.1f} °C",
        delta=f"{delta_temp:+.1f} °C",
        delta_color="inverse",
    )

with col2:
    st.metric(
        label="〰️ Vibration (RMS)",
        value=f"{latest['vibration']:.2f} mm/s",
        delta=f"{delta_vib:+.2f} mm/s",
        delta_color="inverse",
    )

with col3:
    st.metric(
        label="🔄 Pressure",
        value=f"{latest['pressure']:.2f} bar",
        delta=f"{delta_press:+.2f} bar",
    )

with col4:
    st.metric(
        label="🛡️ Asset Health Score",
        value=f"{latest['health_score']:.1f}%",
        delta=f"{delta_health:+.1f}%",
        delta_color="normal",
    )

st.markdown("---")

# Health Gauge & Anomaly Alert Panel
row2_col1, row2_col2 = st.columns([1, 2])

with row2_col1:
    gauge_fig = create_gauge_chart(
        value=latest["health_score"],
        title="Asset Health Index",
        min_val=0.0,
        max_val=100.0,
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

with row2_col2:
    st.markdown("#### 🚨 Real-Time Diagnostic Alerts & Thresholds")

    # Evaluate ISO-10816 conditions
    alerts = []
    vib_val = latest["vibration"]
    temp_val = latest["temperature"]
    press_val = latest["pressure"]

    if vib_val > 4.5:
        alerts.append(("danger", f"CRITICAL VIBRATION: {vib_val:.2f} mm/s exceeds ISO 10816 Zone D limit (4.5 mm/s). Immediate shutdown recommended."))
    elif vib_val > 2.8:
        alerts.append(("warning", f"ELEVATED VIBRATION: {vib_val:.2f} mm/s in ISO Zone C. Schedule bearing inspection."))

    if temp_val > 90.0:
        alerts.append(("danger", f"CRITICAL TEMPERATURE: {temp_val:.1f} °C exceeds maximum thermal threshold (90 °C). Thermal cutoff hazard."))
    elif temp_val > 78.0:
        alerts.append(("warning", f"HIGH TEMPERATURE: {temp_val:.1f} °C approaching allowable thermal insulation limit."))

    if press_val < 3.5:
        alerts.append(("warning", f"LOW DISCHARGE PRESSURE: {press_val:.2f} bar indicates possible cavitation or intake blockage."))
    elif press_val > 6.2:
        alerts.append(("warning", f"OVER-PRESSURE CONDITION: {press_val:.2f} bar exceeds nominal pipeline design pressure."))

    if not alerts:
        alerts.append(("success", f"Asset operating within standard design envelope. All ISO-10816 tolerances satisfied."))

    for alert_type, alert_msg in alerts:
        css_class = f"custom-alert custom-alert-{alert_type}"
        icon = "🔴" if alert_type == "danger" else ("🟡" if alert_type == "warning" else "🟢")
        st.markdown(
            f"""
            <div class="{css_class}">
                <span style="font-size: 1.1rem;">{icon}</span>
                <div>{alert_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# Telemetry Time-Series Charts
st.subheader("📈 Live Telemetry Waveforms")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_vib = create_time_series_chart(
        df=df_history,
        x_col="timestamp",
        y_cols=["vibration"],
        title="Vibration Velocity (mm/s RMS)",
        unit="mm/s",
    )
    st.plotly_chart(fig_vib, use_container_width=True)

with chart_col2:
    fig_temp = create_time_series_chart(
        df=df_history,
        x_col="timestamp",
        y_cols=["temperature"],
        title="Operating Temperature (°C)",
        unit="°C",
    )
    st.plotly_chart(fig_temp, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_press = create_time_series_chart(
        df=df_history,
        x_col="timestamp",
        y_cols=["pressure"],
        title="Operational Pressure (bar)",
        unit="bar",
    )
    st.plotly_chart(fig_press, use_container_width=True)

with chart_col4:
    fig_health = create_time_series_chart(
        df=df_history,
        x_col="timestamp",
        y_cols=["health_score"],
        title="Composite Health Degradation (%)",
        unit="%",
    )
    st.plotly_chart(fig_health, use_container_width=True)

# Auto-refresh control in sidebar
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("⏱️ Auto-Refresh (5s)", value=False)
if auto_refresh:
    time.sleep(5)
    st.rerun()
