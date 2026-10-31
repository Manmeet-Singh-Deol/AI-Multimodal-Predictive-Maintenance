"""SQLAlchemy ORM models for the predictive maintenance database.

Supports both SQLite (local development) and PostgreSQL (Supabase / Neon cloud).
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base declarative class for all database tables."""
    pass


class Machine(Base):
    """Monitored industrial asset record."""
    __tablename__ = "machines"

    machine_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    category = Column(String(80), nullable=False, default="Rotating Equipment")
    base_temp = Column(Float, default=70.0)
    base_vib = Column(Float, default=1.5)
    base_press = Column(Float, default=5.0)
    base_rpm = Column(Float, default=3000.0)
    base_current = Column(Float, default=18.0)
    base_noise = Column(Float, default=72.0)
    current_health_score = Column(Float, default=95.0)
    status = Column(String(30), default="OPERATIONAL")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    telemetry = relationship("SensorReading", back_populates="machine", cascade="all, delete-orphan")
    work_orders = relationship("MaintenanceWorkOrder", back_populates="machine", cascade="all, delete-orphan")


class SensorReading(Base):
    """High-frequency industrial sensor reading telemetry point."""
    __tablename__ = "sensor_telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String(50), ForeignKey("machines.machine_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    temperature = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    rpm = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    noise_db = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)
    fault_label = Column(String(60), default="Normal Operation")

    machine = relationship("Machine", back_populates="telemetry")


class MaintenanceWorkOrder(Base):
    """Historical maintenance event or work order ticket."""
    __tablename__ = "maintenance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    work_order_id = Column(String(50), unique=True, nullable=False, index=True)
    machine_id = Column(String(50), ForeignKey("machines.machine_id", ondelete="CASCADE"), nullable=False, index=True)
    maintenance_type = Column(String(60), nullable=False)
    component = Column(String(100), nullable=False)
    technician = Column(String(100), nullable=False)
    service_date = Column(String(20), nullable=False)
    actions_taken = Column(Text, nullable=False)
    replaced_parts = Column(Text, nullable=True)
    downtime_hours = Column(Float, default=0.0)

    machine = relationship("Machine", back_populates="work_orders")


class ChatHistory(Base):
    """AI Assistant conversation history record."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    machine_id = Column(String(50), nullable=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)
    safety_warnings_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
