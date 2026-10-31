"""Data pipeline module for sensor data ingestion, generation, and preprocessing."""

from src.data_pipeline.data_loader import (
    CMAPSSDataLoader,
    add_piecewise_rul,
    download_cmapss,
    generate_synthetic_cmapss,
)
from src.data_pipeline.feature_engineering import (
    FeatureEngineer,
    compute_frequency_domain_metrics,
    compute_time_domain_metrics,
    extract_all_features,
)
from src.data_pipeline.preprocessor import (
    DataPreprocessor,
    DataScaler,
    add_gaussian_noise,
    create_sliding_windows,
    temporal_train_val_test_split,
    time_warp,
)
from src.data_pipeline.synthetic_generator import (
    FAULT_SIGNATURES,
    MACHINE_STATES,
    NOMINAL_BASELINES,
    SyntheticIndustrialGenerator,
)

__all__ = [
    "CMAPSSDataLoader",
    "download_cmapss",
    "generate_synthetic_cmapss",
    "add_piecewise_rul",
    "SyntheticIndustrialGenerator",
    "MACHINE_STATES",
    "NOMINAL_BASELINES",
    "FAULT_SIGNATURES",
    "FeatureEngineer",
    "extract_all_features",
    "compute_time_domain_metrics",
    "compute_frequency_domain_metrics",
    "DataScaler",
    "DataPreprocessor",
    "temporal_train_val_test_split",
    "create_sliding_windows",
    "add_gaussian_noise",
    "time_warp",
]
