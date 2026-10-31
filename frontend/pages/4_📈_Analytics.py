"""Historical analytics, reliability metrics, and maintenance event logs page.

Provides longitudinal sensor trend analysis, MTBF calculations, fault distribution Pareto charts,
and maintenance work order records.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import random
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import (
    create_bar_chart,
    create_heatmap,
    create_pie_chart,
    create_scatter_plot,
    create_time_series_chart,
)
from components.sidebar import render_sidebar

# Page Configuration
st.set_page_config(page_title="Analytics | AI Maintenance", page_icon="📈", layout="wide")

# Load CSS
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar
selected_machine, api_url, is_online = render_sidebar()

# Page Header
st.title("📈 Fleet Reliability & Historical Analytics")
st.markdown("Longitudinal equipment reliability metrics, sensor cross-correlation, and maintenance logs.")

# Date Range Picker
today = date.today()
thirty_days_ago = today - timedelta(days=30)

filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
with filter_col1:
    date_range = st.date_input(
        "Select Historical Date Range",
        value=(thirty_days_ago, today),
        max_value=today,
    )
with filter_col2:
    view_scope = st.selectbox("Scope", ["Selected Machine (" + selected_machine + ")", "Entire Fleet (All Machines)"])
with filter_col3:
    st.write("")
    st.write("")
    export_clicked = st.button("📥 Export Report", use_container_width=True)

st.markdown("---")

# Key Reliability Metric Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric(label="⏱️ Total Downtime", value="28.4 hrs", delta="-4.2 hrs (MoM)", delta_color="inverse")
with kpi_col2:
    st.metric(label="⚙️ MTBF (Mean Time Between Failures)", value="524 hrs", delta="+38 hrs", delta_color="normal")
with kpi_col3:
    st.metric(label="⚠️ Fault Incident Count", value="6 events", delta="-2 events", delta_color="inverse")
with kpi_col4:
    st.metric(label="🛡️ Fleet Health Average", value="87.6%", delta="+1.8%", delta_color="normal")

st.markdown("---")


def generate_historical_series(days: int = 30) -> pd.DataFrame:
    """Generate multi-day historical telemetry data."""
    now = datetime.now(timezone.utc)
    records = []
    for i in range(days * 4):  # 6-hour intervals
        ts = now - timedelta(hours=i * 6)
        temp = 68.0 + 8.0 * np.sin(i * 0.1) + random.gauss(0, 1.2)
        vib = 1.3 + max(0.0, 1.5 * np.sin(i * 0.08)) + random.gauss(0, 0.2)
        press = 5.1 + 0.5 * np.cos(i * 0.12) + random.gauss(0, 0.15)
        rpm = 2990.0 + random.gauss(0, 18.0)
        current = 18.5 + (vib * 0.8) + random.gauss(0, 0.4)
        health = max(20.0, min(100.0, 98.0 - (vib * 8.0) - max(0.0, temp - 75.0)))

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "temperature": round(temp, 1),
            "vibration": round(vib, 2),
            "pressure": round(press, 2),
            "rpm": round(rpm, 0),
            "current": round(current, 1),
            "health_score": round(health, 1),
        })

    df = pd.DataFrame(records[::-1])
    return df


df_hist = generate_historical_series(30)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Sensor Trends", "🔬 Fault History & Pareto", "🛠️ Maintenance Log"])

# Tab 1: Sensor Trends
with tab1:
    st.markdown("#### Longitudinal Sensor Comparative Analysis")
    metric_options = ["temperature", "vibration", "pressure", "current", "rpm", "health_score"]
    chosen_metrics = st.multiselect(
        "Select Telemetry Variables to Plot",
        options=metric_options,
        default=["vibration", "temperature"],
    )

    if chosen_metrics:
        fig_trend = create_time_series_chart(
            df=df_hist,
            x_col="timestamp",
            y_cols=chosen_metrics,
            title="Multi-Parameter Sensor Trends Over Time",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    corr_col1, corr_col2 = st.columns(2)
    with corr_col1:
        st.markdown("##### Vibration vs Temperature Correlation")
        fig_scatter = create_scatter_plot(
            x=df_hist["vibration"].tolist(),
            y=df_hist["temperature"].tolist(),
            title="Vibration vs Temperature Correlation",
            color_by=df_hist["health_score"].tolist(),
            x_label="Vibration (mm/s RMS)",
            y_label="Temperature (°C)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with corr_col2:
        st.markdown("##### Cross-Sensor Correlation Matrix")
        corr_matrix = df_hist[["temperature", "vibration", "pressure", "current", "rpm", "health_score"]].corr()
        fig_heat = create_heatmap(
            data=corr_matrix.values.round(2).tolist(),
            x_labels=["Temp", "Vib", "Press", "Curr", "RPM", "Health"],
            y_labels=["Temp", "Vib", "Press", "Curr", "RPM", "Health"],
            title="Telemetry Correlation Heatmap",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# Tab 2: Fault History
with tab2:
    st.markdown("#### Historical Failure Modes & Distribution")

    fault_counts = {
        "Bearing Degradation": 18,
        "Motor Overheating": 9,
        "Cavitation / Flow": 7,
        "Shaft Misalignment": 5,
        "Electrical Unbalance": 3,
    }

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        fig_fault_bar = create_bar_chart(
            categories=list(fault_counts.keys()),
            values=list(fault_counts.values()),
            title="Total Fault Incidents by Failure Mode (Last 90 Days)",
            colors=["#ef4444", "#f59e0b", "#3b82f6", "#8b5cf6", "#10b981"],
        )
        st.plotly_chart(fig_fault_bar, use_container_width=True)

    with f_col2:
        fig_pie = create_pie_chart(
            labels=list(fault_counts.keys()),
            values=list(fault_counts.values()),
            title="Fault Category Breakdown (%)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Tab 3: Maintenance Log
with tab3:
    st.markdown("#### Preventive & Corrective Maintenance History")

    maintenance_records = [
        {
            "Work Order": "WO-2026-089",
            "Date": "2026-10-24",
            "Asset": "MCH-002",
            "Component": "Drive End Roller Bearing",
            "Fault Detected": "Bearing Spalling (ISO Zone C)",
            "Action Taken": "Replaced roller bearing, flushed grease, re-aligned shaft",
            "Downtime (hrs)": 4.5,
            "Technician": "Alex Mercer",
            "Status": "Verified Closed",
        },
        {
            "Work Order": "WO-2026-077",
            "Date": "2026-10-18",
            "Asset": "MCH-004",
            "Component": "Cooling Radiator & Fan Cowl",
            "Fault Detected": "Thermal Overload (96°C)",
            "Action Taken": "Cleaned clogged radiator fins, replaced cooling fan belt",
            "Downtime (hrs)": 2.0,
            "Technician": "Sarah Jenkins",
            "Status": "Verified Closed",
        },
        {
            "Work Order": "WO-2026-065",
            "Date": "2026-10-10",
            "Asset": "MCH-003",
            "Component": "Suction Strainer & Impeller",
            "Fault Detected": "Cavitation Noise & Pressure Drop",
            "Action Taken": "Cleared intake blockage, replaced worn mechanical seal",
            "Downtime (hrs)": 6.0,
            "Technician": "David Chen",
            "Status": "Verified Closed",
        },
        {
            "Work Order": "WO-2026-052",
            "Date": "2026-09-28",
            "Asset": "MCH-001",
            "Component": "Flexible Coupling",
            "Fault Detected": "Angular Misalignment (2X RPM)",
            "Action Taken": "Laser realignment of pump-motor shaft to 0.03 mm",
            "Downtime (hrs)": 3.0,
            "Technician": "Alex Mercer",
            "Status": "Verified Closed",
        },
        {
            "Work Order": "WO-2026-041",
            "Date": "2026-09-15",
            "Asset": "MCH-005",
            "Component": "Spindle Lube System",
            "Fault Detected": "Routine Preventive Maintenance",
            "Action Taken": "Oil sampling, ISO 4406 particle count, filter replacement",
            "Downtime (hrs)": 1.5,
            "Technician": "Sarah Jenkins",
            "Status": "Verified Closed",
        },
    ]

    df_maint = pd.DataFrame(maintenance_records)
    st.dataframe(df_maint, use_container_width=True, hide_index=True)

    csv_data = df_maint.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Maintenance Records (CSV)",
        data=csv_data,
        file_name="maintenance_log_export.csv",
        mime="text/csv",
    )
