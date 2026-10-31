"""Database repository providing CRUD operations for sensors, machines, and chatbot."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .models import ChatHistory, Machine, MaintenanceWorkOrder, SensorReading


def save_telemetry_point(
    db: Session,
    machine_id: str,
    temp: float,
    vib: float,
    press: float,
    rpm: float,
    current: float,
    noise: float,
    is_anomaly: bool = False,
    anomaly_score: float = 0.0,
    fault_label: str = "Normal Operation",
    timestamp: Optional[datetime] = None,
) -> SensorReading:
    """Save an incoming telemetry point to the database."""
    reading = SensorReading(
        machine_id=machine_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        temperature=round(temp, 2),
        vibration=round(vib, 3),
        pressure=round(press, 2),
        rpm=round(rpm, 1),
        current=round(current, 2),
        noise_db=round(noise, 1),
        is_anomaly=is_anomaly,
        anomaly_score=round(anomaly_score, 4),
        fault_label=fault_label,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_telemetry_history(db: Session, machine_id: str, limit: int = 60) -> List[SensorReading]:
    """Retrieve the most recent sensor telemetry points for a machine."""
    return (
        db.query(SensorReading)
        .filter(SensorReading.machine_id == machine_id)
        .order_by(desc(SensorReading.timestamp))
        .limit(limit)
        .all()
    )[::-1]  # Return in chronological order


def get_all_machines(db: Session) -> List[Machine]:
    """Retrieve all monitored machine assets."""
    return db.query(Machine).all()


def get_machine(db: Session, machine_id: str) -> Optional[Machine]:
    """Retrieve a single machine by ID."""
    return db.query(Machine).filter(Machine.machine_id == machine_id).first()


def update_machine_health(db: Session, machine_id: str, health_score: float, status: str) -> Optional[Machine]:
    """Update live health score and operational status for an asset."""
    m = get_machine(db, machine_id)
    if m:
        m.current_health_score = round(health_score, 1)
        m.status = status
        m.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(m)
    return m


def get_maintenance_history(db: Session, machine_id: Optional[str] = None) -> List[MaintenanceWorkOrder]:
    """Retrieve maintenance work orders, optionally filtered by asset."""
    query = db.query(MaintenanceWorkOrder)
    if machine_id:
        query = query.filter(MaintenanceWorkOrder.machine_id == machine_id)
    return query.order_by(desc(MaintenanceWorkOrder.service_date)).all()


def save_chat_turn(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    machine_id: Optional[str] = None,
    sources: Optional[List[str]] = None,
    safety_warnings: Optional[List[str]] = None,
) -> ChatHistory:
    """Store a conversational exchange in the database."""
    record = ChatHistory(
        session_id=session_id,
        machine_id=machine_id,
        role=role,
        content=content,
        sources_json=json.dumps(sources) if sources else None,
        safety_warnings_json=json.dumps(safety_warnings) if safety_warnings else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_chat_session_history(db: Session, session_id: str) -> List[ChatHistory]:
    """Retrieve all conversation turns for a given session."""
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )
