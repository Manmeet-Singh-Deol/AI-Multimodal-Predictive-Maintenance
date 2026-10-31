"""Sidebar component for global navigation, machine selection, and backend connectivity status.

Provides the standardized sidebar interface for all Streamlit frontend pages.
"""

from typing import Tuple
import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
MACHINE_LIST = ["MCH-001", "MCH-002", "MCH-003", "MCH-004", "MCH-005"]


def check_backend_health(api_url: str) -> bool:
    """Check connectivity to FastAPI backend with short timeout."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=1.2)
        return response.status_code == 200
    except Exception:
        return False


def render_sidebar() -> Tuple[str, str, bool]:
    """Render the standard sidebar and return (selected_machine_id, api_url, is_api_online).

    Returns:
        Tuple[str, str, bool]: (machine_id, api_url, is_online)
    """
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL

    if "selected_machine" not in st.session_state:
        st.session_state.selected_machine = MACHINE_LIST[0]

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0 20px 0;">
                <h2 style="margin: 0; color: #ffffff; font-size: 1.5rem; letter-spacing: -0.02em;">
                    🔧 <span style="color: #00d2ff;">Jarvis</span> Industrial
                </h2>
                <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;">
                    Predictive Maintenance AI
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Glass UI Badge
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin: 4px 0 16px 0; padding: 6px 12px; background: rgba(0, 210, 255, 0.1); border: 1px solid rgba(0, 210, 255, 0.3); border-radius: 999px; font-size: 0.76rem; color: #38bdf8; backdrop-filter: blur(8px);">
                <span>💎</span> <b>Glassmorphic UI Active</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Machine Selector
        st.subheader("⚙️ Monitored Asset")
        asset_labels = {
            "MCH-001": "MCH-001 • Centrifugal Compressor",
            "MCH-002": "MCH-002 • Slurry Pump (Bearing Wear)",
            "MCH-003": "MCH-003 • Feedwater Pump (Cavitation)",
            "MCH-004": "MCH-004 • Glass Lehr Thermal Furnace",
            "MCH-005": "MCH-005 • Glass Bottle IS Machine",
        }
        current_idx = (
            MACHINE_LIST.index(st.session_state.selected_machine)
            if st.session_state.selected_machine in MACHINE_LIST
            else 0
        )
        selected_machine = st.selectbox(
            "Select Machine ID",
            options=MACHINE_LIST,
            index=current_idx,
            format_func=lambda x: asset_labels.get(x, x),
            help="Choose the industrial asset to inspect telemetry and diagnostics for.",
        )
        st.session_state.selected_machine = selected_machine

        st.markdown("---")


        # Backend Health Status
        is_online = check_backend_health(st.session_state.api_url)
        st.subheader("🌐 System Connectivity")
        if is_online:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></span>
                    <span style="font-size: 0.88rem; color: #34d399; font-weight: 600;">Backend Connected</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="height: 10px; width: 10px; background-color: #ef4444; border-radius: 50%; box-shadow: 0 0 8px #ef4444;"></span>
                    <span style="font-size: 0.88rem; color: #f87171; font-weight: 600;">API Offline (Demo Mode)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Settings Expander
        with st.expander("🛠️ API & Telemetry Settings", expanded=False):
            api_input = st.text_input(
                "API Base URL",
                value=st.session_state.api_url,
                help="Host and port where the FastAPI backend is running.",
            )
            if api_input != st.session_state.api_url:
                st.session_state.api_url = api_input
                st.rerun()

            if st.button("🔄 Test Backend Ping", use_container_width=True):
                test_online = check_backend_health(st.session_state.api_url)
                if test_online:
                    st.success("API reachable!")
                else:
                    st.warning("API not responding. Demo fallback active.")

        st.markdown(
            """
            <div style="margin-top: 30px; padding: 12px; background: rgba(15, 23, 42, 0.4); border-radius: 8px; font-size: 0.75rem; color: #64748b; text-align: center;">
                Industrial AI Assistant v1.0.0<br/>
                Multimodal Diagnostics & Prognostics
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_machine, st.session_state.api_url, is_online
