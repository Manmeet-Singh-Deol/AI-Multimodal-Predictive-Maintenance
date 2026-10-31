"""Main entry point for AI Maintenance Assistant Streamlit application.

Renders the portal homepage, system overview, feature showcase, and global styling.
"""

from pathlib import Path
import streamlit as st

from components.sidebar import render_sidebar

# Page Configuration
st.set_page_config(
    page_title="AI Maintenance Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Global Sidebar
selected_machine, api_url, is_online = render_sidebar()

# Hero Header
st.markdown(
    """
    <div style="padding: 20px 0 10px 0;">
        <h1 style="font-size: 2.6rem; font-weight: 800; color: #ffffff; margin-bottom: 8px;">
            AI-Powered Industrial Maintenance Assistant
        </h1>
        <p style="font-size: 1.15rem; color: #94a3b8; max-width: 900px; line-height: 1.6;">
            Next-generation predictive maintenance and multimodal troubleshooting platform combining
            real-time machine telemetry, deep learning diagnostics, Remaining Useful Life (RUL) prognostics,
            and RAG-augmented technical manual assistance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# System Architecture & Capability Badges
st.markdown(
    """
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 25px;">
        <span class="badge-tag">💎 Glassmorphism Theme</span>
        <span class="badge-tag">⚡ FastAPI Backend</span>
        <span class="badge-tag">🔥 PyTorch Deep Learning</span>
        <span class="badge-tag">🧠 Gemini Multimodal AI</span>
        <span class="badge-tag">📚 ChromaDB Vector RAG</span>
        <span class="badge-tag">📊 Plotly Interactive Telemetry</span>
        <span class="badge-tag">🛡️ ISO 10816 Vibration Standards</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Feature Showcase Grid
st.subheader("🚀 Platform Modules & Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Real-Time Monitoring Dashboard</div>
            <div class="feature-desc">
                Continuous high-frequency telemetry streaming across temperature, vibration, pressure,
                rotational speed, and acoustic noise with real-time ISO tolerance monitoring and anomaly alerts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Troubleshooting Chatbot</div>
            <div class="feature-desc">
                Multimodal industrial technician assistant supporting conversational queries, component
                image defect analysis (bearing spalling, thermal bluing), and spoken voice commands.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <div class="feature-title">Predictive Diagnostics & RUL</div>
            <div class="feature-desc">
                Deep learning inference combining autoencoder anomaly detection, multi-class fault
                classification (bearing, cavitation, misalignment), and Remaining Useful Life (RUL) curve estimation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

col4, col5 = st.columns(2)

with col4:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Historical Trends & Fleet Analytics</div>
            <div class="feature-desc">
                Longitudinal telemetry tracking, Mean Time Between Failures (MTBF), failure mode distribution,
                and correlation scatter matrices for root-cause failure analysis (RCFA).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <div class="feature-title">RAG Knowledge Base & Manuals</div>
            <div class="feature-desc">
                Automated PDF ingestion and chunking with vector retrieval over OEM equipment manuals,
                providing cited maintenance procedures, torque specifications, and safety precautions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Quick Start & Navigation Instructions
st.subheader("🧭 Getting Started")
info_col1, info_col2 = st.columns([2, 1])

with info_col1:
    st.markdown(
        """
        1. **Select an Asset:** Use the sidebar dropdown to choose between monitored assets (`MCH-001` through `MCH-005`).
        2. **Inspect Telemetry:** Navigate to **📊 Dashboard** to view live sensor readings, health index, and active alert statuses.
        3. **Diagnose Failures:** Open **🔮 Predictions** for deep learning fault probability breakdown and RUL timeline.
        4. **Troubleshoot Interactively:** Open **🤖 Chatbot** to ask questions, upload component photos, or speak commands.
        5. **Search OEM Manuals:** Use **📚 Knowledge Base** to index new PDF manuals or search technical specifications.
        """
    )

with info_col2:
    st.info(
        f"""
        **System Status:**
        - **Selected Asset:** `{selected_machine}`
        - **Backend Host:** `{api_url}`
        - **Status:** {'🟢 Online' if is_online else '🟡 Offline (Demo Mode)'}
        """
    )
