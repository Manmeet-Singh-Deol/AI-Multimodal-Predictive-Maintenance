"""Pydantic schemas for API request and response data models.

Defines type-safe data validation models for chat, predictions, sensor telemetry,
and document indexing in the predictive maintenance assistant.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for chat interactions."""

    message: str = Field(..., description="User query or troubleshooting prompt")
    machine_id: Optional[str] = Field(default=None, description="Optional target machine identifier")
    session_id: str = Field(default="default", description="Conversation session identifier")


class ChatResponse(BaseModel):
    """Response payload from chatbot agent."""

    response: str = Field(..., description="Agent text response")
    sources: List[str] = Field(default_factory=list, description="Referenced manual sections or documents")
    safety_warnings: List[str] = Field(default_factory=list, description="Safety precautions and warning notices")
    suggestions: List[str] = Field(default_factory=list, description="Recommended follow-up questions or actions")


class PredictionRequest(BaseModel):
    """Request payload for model inference."""

    machine_id: str = Field(..., description="Identifier of the target machine")
    sensor_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional current sensor telemetry dictionary. If omitted, latest readings are fetched.",
    )


class AnomalyResponse(BaseModel):
    """Response payload for anomaly detection."""

    is_anomaly: bool = Field(..., description="Whether an anomaly is detected")
    anomaly_score: float = Field(..., description="Normalized anomaly score between 0.0 and 1.0")
    details: str = Field(..., description="Explanation and flagged threshold details")


class FaultResponse(BaseModel):
    """Response payload for fault classification."""

    predicted_fault: str = Field(..., description="Primary identified fault mode")
    probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Class probabilities across fault categories"
    )
    confidence: float = Field(..., description="Model confidence in primary prediction (0.0 to 1.0)")


class RULResponse(BaseModel):
    """Response payload for Remaining Useful Life (RUL) estimation."""

    estimated_rul: float = Field(..., description="Estimated remaining useful life")
    unit: str = Field(default="hours", description="Time unit of RUL estimation (e.g., hours or cycles)")
    confidence_interval: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence bounds (e.g., {'lower': float, 'upper': float})",
    )


class SensorData(BaseModel):
    """Data model representing a single snapshot of machine telemetry."""

    machine_id: str = Field(..., description="Unique machine identifier (e.g., 'MCH-001')")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp string")
    temperature: float = Field(..., description="Operating temperature in degrees Celsius")
    vibration: float = Field(..., description="Vibration amplitude in mm/s RMS")
    pressure: float = Field(..., description="Operational pressure in bar / PSI")
    rpm: float = Field(..., description="Rotational speed in RPM")
    current: float = Field(..., description="Motor current draw in Amperes")
    noise_db: float = Field(..., description="Acoustic noise level in decibels")
    health_score: float = Field(..., description="Composite health indicator score (0.0 - 100.0%)")


class DashboardData(BaseModel):
    """Consolidated telemetry snapshot for fleet overview."""

    machines: List[SensorData] = Field(default_factory=list, description="Current readings for all tracked machines")
    alerts: List[str] = Field(default_factory=list, description="Active system alerts and warnings")
    overall_health: float = Field(..., description="Average fleet health index score (0.0 - 100.0)")


class UploadResponse(BaseModel):
    """Response payload for file uploads and knowledge base indexing."""

    status: str = Field(..., description="Upload status ('success', 'failed', 'processing')")
    message: str = Field(..., description="User-facing detail or summary message")
    documents_indexed: int = Field(default=0, description="Total number of chunks or documents processed")


class AudioAnalysisResponse(BaseModel):
    """Response payload for acoustic telemetry and audio inspection."""

    status: str = Field(..., description="Analysis execution status")
    rms_energy: float = Field(..., description="Root Mean Square acoustic energy")
    spectral_centroid: float = Field(..., description="Dominant frequency center in Hz")
    is_acoustic_anomaly: bool = Field(..., description="Acoustic anomaly classification")
    description: str = Field(..., description="Diagnosis based on acoustic signature")


class HealthCheckResponse(BaseModel):
    """Health check response for the service."""

    status: str = Field(default="healthy", description="Current health status of the API")
    version: str = Field(default="1.0.0", description="API version")
