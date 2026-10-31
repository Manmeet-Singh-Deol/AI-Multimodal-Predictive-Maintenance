"""Sensor telemetry and simulation API routes.

Provides endpoints for real-time sensor streams, historical telemetry retrieval,
and synthetic sensor data injection.
"""

from datetime import datetime, timedelta, timezone
import math
import random
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from config.settings import get_settings
from ..schemas import SensorData

router = APIRouter(prefix="/api/sensors", tags=["Sensors"])
settings = get_settings()

# In-memory time-series store: machine_id -> List[SensorData]
_SENSOR_STORE: Dict[str, List[SensorData]] = {}

# Machine operating profile baseline states
MACHINE_PROFILES: Dict[str, Dict[str, float]] = {
    "MCH-001": {  # Normal healthy compressor
        "base_temp": 68.0,
        "base_vib": 1.4,
        "base_press": 5.2,
        "base_rpm": 2980.0,
        "base_current": 18.2,
        "base_noise": 72.0,
        "base_health": 95.0,
    },
    "MCH-002": {  # Motor with early bearing wear (elevated vibration & noise)
        "base_temp": 79.5,
        "base_vib": 4.8,
        "base_press": 4.9,
        "base_rpm": 2940.0,
        "base_current": 22.4,
        "base_noise": 86.5,
        "base_health": 68.0,
    },
    "MCH-003": {  # Pump with cavitation & pressure instability
        "base_temp": 73.0,
        "base_vib": 3.9,
        "base_press": 3.1,
        "base_rpm": 2890.0,
        "base_current": 20.8,
        "base_noise": 89.0,
        "base_health": 54.0,
    },
    "MCH-004": {  # Glass Lehr Thermal Annealing Furnace (elevated heat zone)
        "base_temp": 98.2,
        "base_vib": 2.6,
        "base_press": 6.8,
        "base_rpm": 3050.0,
        "base_current": 26.5,
        "base_noise": 81.0,
        "base_health": 42.0,
    },
    "MCH-005": {  # Glass Bottle Forming IS Machine (precision pneumatic molding)
        "base_temp": 64.5,
        "base_vib": 1.1,
        "base_press": 5.0,
        "base_rpm": 3010.0,
        "base_current": 17.5,
        "base_noise": 69.5,
        "base_health": 98.0,
    },
}


def _generate_point(machine_id: str, timestamp: datetime, step_index: int = 0) -> SensorData:
    """Generate a realistic physics-guided sensor telemetry point."""
    profile = MACHINE_PROFILES.get(
        machine_id,
        {
            "base_temp": 70.0,
            "base_vib": 2.0,
            "base_press": 5.0,
            "base_rpm": 3000.0,
            "base_current": 19.0,
            "base_noise": 75.0,
            "base_health": 85.0,
        },
    )

    # Periodic oscillatory component plus Gaussian noise
    osc = math.sin(step_index * 0.15) * 0.8
    noise = lambda scale: random.gauss(0.0, scale)

    temp = max(20.0, round(profile["base_temp"] + osc * 1.5 + noise(0.8), 2))
    vib = max(0.1, round(profile["base_vib"] + abs(osc * 0.4) + noise(0.2), 3))
    press = max(0.5, round(profile["base_press"] + osc * 0.2 + noise(0.1), 2))
    rpm = max(100.0, round(profile["base_rpm"] + osc * 12.0 + noise(15.0), 1))
    current = max(1.0, round(profile["base_current"] + osc * 0.5 + noise(0.3), 2))
    noise_db = max(40.0, round(profile["base_noise"] + osc * 1.0 + noise(1.2), 2))

    # Calculate realistic dynamic health score
    health_penalty = (
        max(0.0, (temp - 75.0) * 0.8)
        + max(0.0, (vib - 2.5) * 6.0)
        + max(0.0, abs(press - 5.0) * 4.0)
    )
    health = max(5.0, min(100.0, round(profile["base_health"] - health_penalty + noise(0.5), 1)))

    return SensorData(
        machine_id=machine_id,
        timestamp=timestamp.isoformat(),
        temperature=temp,
        vibration=vib,
        pressure=press,
        rpm=rpm,
        current=current,
        noise_db=noise_db,
        health_score=health,
    )


def initialize_sensor_store() -> None:
    """Pre-populate in-memory telemetry store with historical readings."""
    if _SENSOR_STORE:
        return

    logger.info("Initializing in-memory telemetry store with baseline readings...")
    now = datetime.now(timezone.utc)
    machine_ids = list(MACHINE_PROFILES.keys())[: max(1, settings.num_machines)]

    for machine_id in machine_ids:
        series: List[SensorData] = []
        for i in range(120, -1, -1):
            ts = now - timedelta(seconds=i * 10)
            series.append(_generate_point(machine_id, ts, step_index=120 - i))
        _SENSOR_STORE[machine_id] = series

    logger.info(f"Initialized telemetry store for {len(_SENSOR_STORE)} machines.")


# Initialize on import
initialize_sensor_store()


@router.get("/latest", response_model=List[SensorData])
async def get_latest_sensors() -> List[SensorData]:
    """Retrieve the latest telemetry snapshot for all monitored machines."""
    latest_points: List[SensorData] = []
    now = datetime.now(timezone.utc)

    for machine_id in list(_SENSOR_STORE.keys()):
        history = _SENSOR_STORE[machine_id]
        step_idx = len(history) + 1
        new_point = _generate_point(machine_id, now, step_index=step_idx)
        history.append(new_point)
        # Retain last 1000 items to cap memory
        if len(history) > 1000:
            _SENSOR_STORE[machine_id] = history[-1000:]
        latest_points.append(new_point)

    return latest_points


@router.get("/latest/{machine_id}", response_model=SensorData)
async def get_latest_sensor_by_id(machine_id: str) -> SensorData:
    """Retrieve the latest telemetry reading for a specific machine."""
    machine_id = machine_id.upper()
    if machine_id not in _SENSOR_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_id}' not found. Available machines: {list(_SENSOR_STORE.keys())}",
        )

    now = datetime.now(timezone.utc)
    history = _SENSOR_STORE[machine_id]
    step_idx = len(history) + 1
    new_point = _generate_point(machine_id, now, step_index=step_idx)
    history.append(new_point)
    if len(history) > 1000:
        _SENSOR_STORE[machine_id] = history[-1000:]

    return new_point


@router.get("/history/{machine_id}", response_model=List[SensorData])
async def get_sensor_history(
    machine_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Max historical points to return"),
) -> List[SensorData]:
    """Retrieve time-series telemetry history for a specific machine."""
    machine_id = machine_id.upper()
    if machine_id not in _SENSOR_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_id}' not found. Available machines: {list(_SENSOR_STORE.keys())}",
        )

    return _SENSOR_STORE[machine_id][-limit:]


@router.post("/simulate")
async def simulate_sensor_stream(
    num_points: int = Query(10, ge=1, le=100, description="Number of readings to generate"),
    machine_id: Optional[str] = Query(None, description="Target machine (or None for all)"),
) -> Dict[str, object]:
    """Generate and inject a simulated batch of telemetry points."""
    targets = [machine_id.upper()] if machine_id else list(_SENSOR_STORE.keys())
    generated_count = 0
    now = datetime.now(timezone.utc)

    for mid in targets:
        if mid not in _SENSOR_STORE:
            _SENSOR_STORE[mid] = []
        history = _SENSOR_STORE[mid]
        for i in range(num_points):
            ts = now - timedelta(seconds=(num_points - i) * 5)
            pt = _generate_point(mid, ts, step_index=len(history) + i)
            history.append(pt)
            generated_count += 1
        _SENSOR_STORE[mid] = history[-1000:]

    return {
        "status": "success",
        "machines_updated": targets,
        "points_generated": generated_count,
    }


def get_current_reading(machine_id: str) -> SensorData:
    """Internal helper to get latest reading for internal predictive modules."""
    machine_id = machine_id.upper()
    if machine_id in _SENSOR_STORE and _SENSOR_STORE[machine_id]:
        return _SENSOR_STORE[machine_id][-1]
    return _generate_point(machine_id, datetime.now(timezone.utc))
