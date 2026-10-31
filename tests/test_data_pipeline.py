"""Unit tests for the data pipeline module.

Tests data loader, synthetic generator, feature engineering, and preprocessor.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.data_loader import CMAPSSDataLoader, add_piecewise_rul
from src.data_pipeline.feature_engineering import (
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
    MACHINE_STATES,
    SyntheticIndustrialGenerator,
)


def test_synthetic_generator():
    """Verify synthetic sensor data generator produces expected schema and states."""
    gen = SyntheticIndustrialGenerator(num_machines=2, num_cycles=50, fault_ratio=0.5, random_seed=42)
    df = gen.generate_dataset()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    expected_cols = [
        "timestamp",
        "machine_id",
        "temp",
        "vibration",
        "pressure",
        "rpm",
        "current",
        "noise_db",
        "fault_label",
        "health_score",
        "rul",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify states
    unique_states = set(df["fault_label"].unique())
    assert unique_states.issubset(set(MACHINE_STATES))


def test_time_domain_metrics():
    """Verify time-domain statistical indicators."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_time_domain_metrics(signal)

    assert "mean" in metrics
    assert "std" in metrics
    assert "rms" in metrics
    assert "crest_factor" in metrics
    assert np.isclose(metrics["mean"], 3.0)
    assert metrics["rms"] > 0


def test_frequency_domain_metrics():
    """Verify FFT frequency-domain indicators."""
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    metrics = compute_frequency_domain_metrics(signal, sampling_rate=100.0)

    assert "dominant_frequency" in metrics
    assert "spectral_energy" in metrics
    assert "spectral_centroid" in metrics
    assert np.isclose(metrics["dominant_frequency"], 5.0, atol=1.0)


def test_feature_engineering_pipeline():
    """Verify extract_all_features runs end-to-end on a DataFrame."""
    df = pd.DataFrame(
        {
            "machine_id": ["M1"] * 30,
            "temp": np.linspace(50, 70, 30),
            "vibration": np.random.normal(1.5, 0.2, 30),
        }
    )
    engineered = extract_all_features(df, group_col="machine_id")

    assert engineered.shape[0] == 30
    assert engineered.shape[1] > 3  # Expanded feature set
    assert not engineered.isna().any().any()  # No NaNs


def test_preprocessor_and_split():
    """Verify scaling and temporal train/val/test split."""
    df = pd.DataFrame(
        {
            "machine_id": ["M1"] * 50 + ["M2"] * 50,
            "cycle": list(range(50)) + list(range(50)),
            "temp": np.random.randn(100),
            "vibration": np.random.randn(100),
            "rul": np.linspace(100, 0, 100),
        }
    )

    train_df, val_df, test_df = temporal_train_val_test_split(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, group_col="machine_id"
    )

    assert len(train_df) == 70
    assert len(val_df) + len(test_df) == 30

    # Test scaler
    scaler = DataScaler(scaler_type="standard")
    scaler.fit(train_df[["temp", "vibration"]])
    scaled = scaler.transform(train_df[["temp", "vibration"]])
    assert np.isclose(np.mean(scaled[:, 0]), 0.0, atol=1e-5)


def test_sliding_windows():
    """Verify 3D sliding window generation for LSTM."""
    df = pd.DataFrame(
        {
            "machine_id": ["M1"] * 40,
            "f1": np.random.randn(40),
            "f2": np.random.randn(40),
            "rul": np.arange(40),
        }
    )

    seq_len = 10
    X, y = create_sliding_windows(
        df, feature_cols=["f1", "f2"], target_col="rul", sequence_length=seq_len, group_col="machine_id"
    )

    expected_samples = 40 - seq_len + 1
    assert X.shape == (expected_samples, seq_len, 2)
    assert y.shape == (expected_samples,)


def test_data_augmentation():
    """Verify Gaussian jitter and time-warping augmentation."""
    X = np.random.randn(10, 15, 4)
    X_noisy = add_gaussian_noise(X, noise_factor=0.05)
    assert X_noisy.shape == X.shape
    assert not np.array_equal(X_noisy, X)

    X_warped = time_warp(X, sigma=0.1)
    assert X_warped.shape == X.shape
