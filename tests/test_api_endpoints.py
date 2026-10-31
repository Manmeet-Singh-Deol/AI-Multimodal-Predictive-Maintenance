"""Unit and integration tests for FastAPI backend routes.

Verifies health endpoints, sensor telemetry streaming, predictive inference,
chat conversation handling, and upload status.
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Verify root and health check endpoints."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"

    res_h = client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "healthy"


def test_sensors_routes():
    """Verify sensor stream and historical telemetry endpoints."""
    # Latest for all
    res = client.get("/api/sensors/latest")
    assert res.status_code == 200
    telemetry = res.json()
    assert len(telemetry) >= 5
    first = telemetry[0]
    assert "machine_id" in first
    assert "temperature" in first
    assert "vibration" in first
    assert "health_score" in first

    # Latest for specific machine
    res_mch = client.get("/api/sensors/latest/MCH-001")
    assert res_mch.status_code == 200
    assert res_mch.json()["machine_id"] == "MCH-001"

    # History
    res_hist = client.get("/api/sensors/history/MCH-001?limit=25")
    assert res_hist.status_code == 200
    history = res_hist.json()
    assert len(history) == 25

    # Simulate injection
    res_sim = client.post("/api/sensors/simulate?num_points=5&machine_id=MCH-001")
    assert res_sim.status_code == 200
    assert res_sim.json()["status"] == "success"


def test_predict_routes():
    """Verify predictive inference endpoints."""
    payload = {"machine_id": "MCH-002"}

    # Anomaly
    res_anom = client.post("/api/predict/anomaly", json=payload)
    assert res_anom.status_code == 200
    anom_data = res_anom.json()
    assert "is_anomaly" in anom_data
    assert "anomaly_score" in anom_data
    assert 0.0 <= anom_data["anomaly_score"] <= 1.0

    # Fault
    res_fault = client.post("/api/predict/fault", json=payload)
    assert res_fault.status_code == 200
    fault_data = res_fault.json()
    assert "predicted_fault" in fault_data
    assert "probabilities" in fault_data
    assert "confidence" in fault_data

    # RUL
    res_rul = client.post("/api/predict/rul", json=payload)
    assert res_rul.status_code == 200
    rul_data = res_rul.json()
    assert "estimated_rul" in rul_data
    assert rul_data["estimated_rul"] > 0
    assert "confidence_interval" in rul_data

    # Dashboard consolidated
    res_dash = client.get("/api/predict/dashboard")
    assert res_dash.status_code == 200
    dash_data = res_dash.json()
    assert "fleet_overview" in dash_data
    assert "machines" in dash_data


def test_chat_routes():
    """Verify chatbot endpoints."""
    chat_payload = {
        "message": "What are the vibration limits for MCH-002 under ISO 10816?",
        "machine_id": "MCH-002",
        "session_id": "test_session_1",
    }
    res = client.post("/api/chat", json=chat_payload)
    assert res.status_code == 200
    chat_res = res.json()
    assert "response" in chat_res
    assert len(chat_res["response"]) > 0
    assert "sources" in chat_res
    assert "safety_warnings" in chat_res
    assert "suggestions" in chat_res

    # History
    res_hist = client.get("/api/chat/history?session_id=test_session_1")
    assert res_hist.status_code == 200
    hist = res_hist.json()
    assert hist["session_id"] == "test_session_1"
    assert hist["message_count"] >= 2


def test_upload_status_route():
    """Verify upload knowledge base status endpoint."""
    res = client.get("/api/upload/status")
    assert res.status_code == 200
    status_data = res.json()
    assert "status" in status_data
    assert "indexed_documents_count" in status_data
