"""
Centralized configuration for the AI Predictive Maintenance Assistant.
Uses pydantic-settings for type-safe environment variable loading.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API Keys ---
    gemini_api_key: str = ""

    # --- Model Configuration ---
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- ChromaDB ---
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")

    # --- Database Settings (PostgreSQL / Supabase / SQLite) ---
    database_url: str = ""

    # --- API Settings ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # --- Frontend ---
    streamlit_port: int = 8501

    # --- Logging ---
    log_level: str = "INFO"

    # --- Data Paths ---
    data_raw_dir: str = str(PROJECT_ROOT / "data" / "raw")
    data_processed_dir: str = str(PROJECT_ROOT / "data" / "processed")
    data_manuals_dir: str = str(PROJECT_ROOT / "data" / "manuals")
    data_synthetic_dir: str = str(PROJECT_ROOT / "data" / "synthetic")
    models_dir: str = str(PROJECT_ROOT / "models")

    # --- ML Model Hyperparameters ---
    # Anomaly Detection
    anomaly_contamination: float = 0.05
    autoencoder_latent_dim: int = 16
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 64
    autoencoder_lr: float = 0.001

    # Fault Classifier
    fault_classifier_epochs: int = 100
    fault_classifier_batch_size: int = 32
    fault_classifier_lr: float = 0.001
    fault_num_classes: int = 5

    # RUL Estimator
    rul_hidden_size: int = 128
    rul_num_layers: int = 2
    rul_epochs: int = 100
    rul_batch_size: int = 32
    rul_lr: float = 0.001
    rul_sequence_length: int = 30
    rul_max_rul: int = 125

    # --- RAG Settings ---
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    hybrid_search_alpha: float = 0.7  # Weight for semantic vs keyword search

    # --- Chatbot ---
    max_conversation_history: int = 20
    temperature: float = 0.3

    # --- Sensor Simulation ---
    num_machines: int = 5
    sensor_update_interval: float = 1.0  # seconds

    @property
    def raw_path(self) -> Path:
        return Path(self.data_raw_dir)

    @property
    def processed_path(self) -> Path:
        return Path(self.data_processed_dir)

    @property
    def manuals_path(self) -> Path:
        return Path(self.data_manuals_dir)

    @property
    def synthetic_path(self) -> Path:
        return Path(self.data_synthetic_dir)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for path in [
            self.raw_path,
            self.processed_path,
            self.manuals_path,
            self.synthetic_path,
            self.models_path,
            self.chroma_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
