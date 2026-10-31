"""FastAPI application entry point for AI Predictive Maintenance Assistant.

Provides RESTful endpoints for real-time sensor streams, predictive diagnostics,
multimodal troubleshooting agent, and knowledge base document processing.
"""

from contextlib import asynccontextmanager
import sys
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from config.settings import get_settings
from .routes import chat, predict, sensors, upload
from .schemas import HealthCheckResponse

# Configure loguru logging format
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    settings = get_settings()
    logger.info("Starting AI Predictive Maintenance API Service...")
    logger.info(f"Target Environment: Host={settings.api_host}:{settings.api_port}")

    # Ensure all required project data directories exist
    settings.ensure_directories()
    logger.info(f"Verified data directories: {settings.data_raw_dir}, {settings.models_dir}")

    # Initialize relational database
    try:
        from src.database import init_db
        init_db()
        logger.info("Relational database initialized.")
    except Exception as exc:
        logger.warning(f"Database initialization warning: {exc}")

    # Initialize sensor data stream
    sensors.initialize_sensor_store()
    logger.info("Sensor telemetry simulator store initialized.")

    yield

    logger.info("Shutting down AI Predictive Maintenance API Service...")


# Create FastAPI application
app = FastAPI(
    title="AI Predictive Maintenance API",
    description="Intelligent API combining predictive maintenance ML models with multimodal troubleshooting assistance.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware to support Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(chat.router)
app.include_router(predict.router)
app.include_router(sensors.router)
app.include_router(upload.router)


@app.get("/", response_model=HealthCheckResponse, tags=["Health"])
async def root_health_check() -> HealthCheckResponse:
    """Root health check endpoint returning API service status."""
    return HealthCheckResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def api_health_check() -> HealthCheckResponse:
    """Explicit health check endpoint for monitoring probes."""
    return HealthCheckResponse(status="healthy", version="1.0.0")


# --- Global Exception Handlers ---


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard FastAPI HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code} error at {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_type": "HTTPException",
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors with clean structured response."""
    logger.error(f"Validation error at {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "ValidationError",
            "status_code": 422,
            "message": "Input validation failed. Please check your request parameters.",
            "details": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled exception handler to ensure JSON responses."""
    logger.exception(f"Unhandled exception at {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "status_code": 500,
            "message": "An internal server error occurred while processing your request.",
            "details": str(exc),
            "path": str(request.url.path),
        },
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
