"""Database connection engine and session factory.

Seamlessly supports local persistent SQLite or remote cloud PostgreSQL (Supabase / Neon).
"""

from pathlib import Path
from typing import Generator
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from .models import Base, Machine, MaintenanceWorkOrder

settings = get_settings()

def get_engine_url() -> str:
    """Resolve database URL from settings, falling back to local SQLite file."""
    if settings.database_url:
        url = settings.database_url.strip()
        # Ensure standard driver prefix for sync SQLAlchemy
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        return url

    # Default to persistent local SQLite
    db_file = Path(settings.data_raw_dir).parent / "predictive_maintenance.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_file.resolve()}"


ENGINE_URL = get_engine_url()
is_sqlite = ENGINE_URL.startswith("sqlite")

# Create thread-safe SQLAlchemy engine
engine = create_engine(
    ENGINE_URL,
    connect_args={"check_same_thread": False} if is_sqlite else {},
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all database tables and seed initial machine assets if empty."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully at {}", ENGINE_URL.split("@")[-1])

    # Seed baseline machines if empty
    with SessionLocal() as db:
        if db.query(Machine).count() == 0:
            logger.info("Seeding initial machine assets and baseline maintenance logs...")
            seed_machines = [
                Machine(
                    machine_id="MCH-001",
                    name="Centrifugal Gas Compressor #1",
                    category="Compressors",
                    base_temp=68.0, base_vib=1.4, base_press=5.2, base_rpm=2980.0,
                    base_current=18.2, base_noise=72.0, current_health_score=95.0, status="OPERATIONAL"
                ),
                Machine(
                    machine_id="MCH-002",
                    name="Slurry Transport Pump #2",
                    category="Pumping Equipment",
                    base_temp=79.5, base_vib=4.8, base_press=4.9, base_rpm=2940.0,
                    base_current=22.4, base_noise=86.5, current_health_score=68.0, status="WARNING"
                ),
                Machine(
                    machine_id="MCH-003",
                    name="Boiler Feedwater Pump #3",
                    category="Pumping Equipment",
                    base_temp=73.0, base_vib=3.9, base_press=3.1, base_rpm=2890.0,
                    base_current=20.8, base_noise=89.0, current_health_score=54.0, status="WARNING"
                ),
                Machine(
                    machine_id="MCH-004",
                    name="Glass Lehr Thermal Annealing Furnace",
                    category="Thermal Processing",
                    base_temp=98.2, base_vib=2.6, base_press=6.8, base_rpm=3050.0,
                    base_current=26.5, base_noise=81.0, current_health_score=42.0, status="ALERT"
                ),
                Machine(
                    machine_id="MCH-005",
                    name="Glass Bottle Forming IS Machine",
                    category="Molding Equipment",
                    base_temp=64.5, base_vib=1.1, base_press=5.0, base_rpm=3010.0,
                    base_current=17.5, base_noise=69.5, current_health_score=98.0, status="OPERATIONAL"
                ),
            ]
            db.add_all(seed_machines)

            # Seed sample work orders
            seed_orders = [
                MaintenanceWorkOrder(
                    work_order_id="WO-2026-081",
                    machine_id="MCH-002",
                    maintenance_type="Corrective Repair",
                    component="Drive End Bearing",
                    technician="M. Deol",
                    service_date="2026-10-14",
                    actions_taken="Replaced SKF 6314 deep groove ball bearing due to inner race flaking. Flushed reservoir and applied synthetic polyurea grease.",
                    replaced_parts="SKF 6314 Deep Groove Bearing, Nitrile Lip Seal",
                    downtime_hours=3.5,
                ),
                MaintenanceWorkOrder(
                    work_order_id="WO-2026-064",
                    machine_id="MCH-004",
                    maintenance_type="Thermal Calibration",
                    component="Zone 2 Heating Elements",
                    technician="J. Vance",
                    service_date="2026-10-21",
                    actions_taken="Recalibrated Type-K thermocouple sensors in annealing soak zone. Re-torqued terminal connections on SCR power controller.",
                    replaced_parts="Thermocouple Probe 24-inch",
                    downtime_hours=2.0,
                ),
            ]
            db.add_all(seed_orders)
            db.commit()
            logger.info("Successfully seeded 5 machine assets and historical work orders.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
