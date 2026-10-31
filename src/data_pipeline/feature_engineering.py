"""Feature Engineering Module for Industrial Sensor Telemetry.

Extracts comprehensive time-domain, frequency-domain (FFT), and rolling statistics
from multi-sensor time-series to capture degradation signatures, vibration harmonics,
and dynamic operational drift.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def compute_time_domain_metrics(signal: np.ndarray) -> Dict[str, float]:
    """Computes standard time-domain statistical indicators for a 1D signal.

    Calculates: Mean, Std, Variance, RMS, Peak-to-Peak, Crest Factor, Skewness, Kurtosis.

    Args:
        signal: 1D array of numerical sensor measurements.

    Returns:
        Dictionary mapping metric names to scalar values.
    """
    clean_sig = np.asarray(signal, dtype=float)
    clean_sig = clean_sig[~np.isnan(clean_sig)]

    if len(clean_sig) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "rms": 0.0,
            "peak_to_peak": 0.0,
            "crest_factor": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    mean_val = float(np.mean(clean_sig))
    std_val = float(np.std(clean_sig))
    var_val = float(np.var(clean_sig))
    rms_val = float(np.sqrt(np.mean(clean_sig ** 2)))
    p2p_val = float(np.ptp(clean_sig)) if len(clean_sig) > 1 else 0.0
    peak_val = float(np.max(np.abs(clean_sig)))
    crest_factor = float(peak_val / (rms_val + 1e-9))

    if len(clean_sig) >= 3 and std_val > 1e-9:
        skew_val = float(stats.skew(clean_sig))
    else:
        skew_val = 0.0

    if len(clean_sig) >= 4 and std_val > 1e-9:
        kurt_val = float(stats.kurtosis(clean_sig))
    else:
        kurt_val = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "rms": rms_val,
        "peak_to_peak": p2p_val,
        "crest_factor": crest_factor,
        "skewness": skew_val,
        "kurtosis": kurt_val,
    }


def compute_frequency_domain_metrics(
    signal: np.ndarray,
    sampling_rate: float = 1.0,
) -> Dict[str, float]:
    """Computes frequency-domain features via Fast Fourier Transform (FFT).

    Calculates: Dominant Frequency, Spectral Centroid, Spectral Bandwidth, Spectral Energy.

    Args:
        signal: 1D array of numerical sensor measurements.
        sampling_rate: Sampling frequency in Hertz.

    Returns:
        Dictionary mapping spectral metric names to scalar values.
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig[~np.isnan(sig)]
    n = len(sig)

    if n < 4:
        return {
            "dominant_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_energy": 0.0,
        }

    # Remove DC component for AC spectrum analysis
    centered = sig - np.mean(sig)
    fft_vals = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    # Exclude DC zero-frequency bin
    magnitudes = fft_vals[1:]
    nonzero_freqs = freqs[1:]

    if len(magnitudes) == 0 or np.sum(magnitudes) <= 1e-12:
        return {
            "dominant_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_energy": float(np.sum(fft_vals ** 2)),
        }

    dominant_idx = int(np.argmax(magnitudes))
    dominant_freq = float(nonzero_freqs[dominant_idx])

    sum_mag = float(np.sum(magnitudes))
    spectral_energy = float(np.sum(fft_vals ** 2))
    spectral_centroid = float(np.sum(nonzero_freqs * magnitudes) / (sum_mag + 1e-9))

    spectral_bandwidth = float(
        np.sqrt(np.sum(((nonzero_freqs - spectral_centroid) ** 2) * magnitudes) / (sum_mag + 1e-9))
    )

    return {
        "dominant_frequency": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_energy": spectral_energy,
    }


class FeatureEngineer:
    """Transforms raw sensor series into comprehensive multi-domain feature vectors."""

    def __init__(
        self,
        rolling_windows: Sequence[int] = (5, 10, 20),
        fft_window_size: int = 32,
        sampling_rate: float = 1.0,
    ) -> None:
        """Initializes FeatureEngineer.

        Args:
            rolling_windows: Sequence of integer window sizes for rolling statistics.
            fft_window_size: Window size for sliding FFT spectral feature extraction.
            sampling_rate: Sampling frequency in Hz for spectral analysis.
        """
        self.rolling_windows = list(rolling_windows)
        self.fft_window_size = fft_window_size
        self.sampling_rate = sampling_rate

    def extract_rolling_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extracts rolling statistical indicators across designated window spans.

        Indicators include rolling mean, rolling std, EWMA, and rate of change.
        Calculations respect group boundaries (e.g. per machine/unit).

        Args:
            df: Input DataFrame.
            sensor_cols: List of sensor columns to process.
            group_col: Column indicating unit/machine grouping to prevent window bleed.

        Returns:
            DataFrame containing engineered rolling feature columns.
        """
        new_cols: Dict[str, pd.Series] = {}

        for col in sensor_cols:
            if col not in df.columns:
                continue

            for w in self.rolling_windows:
                if group_col and group_col in df.columns:
                    grouped = df.groupby(group_col)[col]
                    roll_mean = grouped.transform(lambda s: s.rolling(w, min_periods=1).mean())
                    roll_std = grouped.transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0.0)
                    ewma_series = grouped.transform(lambda s: s.ewm(span=w, adjust=False).mean())
                    # Rate of change: delta relative to prior value
                    roc_series = grouped.transform(
                        lambda s: (s - s.shift(1)) / (s.shift(1).abs() + 1e-6)
                    ).fillna(0.0)
                    # Rolling RMS
                    grouped_sq = df.groupby(group_col)[col].apply(lambda s: (s ** 2))
                    roll_rms = df.groupby(group_col)[col].transform(
                        lambda s: np.sqrt((s ** 2).rolling(w, min_periods=1).mean())
                    )
                    # Rolling Peak-to-Peak
                    roll_p2p = grouped.transform(
                        lambda s: s.rolling(w, min_periods=1).max() - s.rolling(w, min_periods=1).min()
                    ).fillna(0.0)
                else:
                    s = df[col]
                    roll_mean = s.rolling(w, min_periods=1).mean()
                    roll_std = s.rolling(w, min_periods=1).std().fillna(0.0)
                    ewma_series = s.ewm(span=w, adjust=False).mean()
                    roc_series = ((s - s.shift(1)) / (s.shift(1).abs() + 1e-6)).fillna(0.0)
                    roll_rms = np.sqrt((s ** 2).rolling(w, min_periods=1).mean())
                    roll_p2p = (s.rolling(w, min_periods=1).max() - s.rolling(w, min_periods=1).min()).fillna(0.0)

                new_cols[f"{col}_rolling_mean_w{w}"] = roll_mean
                new_cols[f"{col}_rolling_std_w{w}"] = roll_std
                new_cols[f"{col}_ewma_w{w}"] = ewma_series
                new_cols[f"{col}_roc_w{w}"] = roc_series
                new_cols[f"{col}_rms_w{w}"] = roll_rms
                new_cols[f"{col}_p2p_w{w}"] = roll_p2p

        return pd.DataFrame(new_cols, index=df.index)

    def extract_spectral_sliding_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extracts sliding-window frequency-domain FFT features.

        Computes dominant frequency, spectral centroid, spectral bandwidth, and energy.

        Args:
            df: Input DataFrame.
            sensor_cols: Sensors to analyze in frequency domain (e.g. vibration, current).
            group_col: Grouping column (e.g. machine_id) for sequence boundaries.

        Returns:
            DataFrame containing spectral features.
        """
        w = self.fft_window_size
        results: Dict[str, np.ndarray] = {}

        for col in sensor_cols:
            if col not in df.columns:
                continue

            dom_freqs = np.zeros(len(df), dtype=float)
            centroids = np.zeros(len(df), dtype=float)
            bandwidths = np.zeros(len(df), dtype=float)
            energies = np.zeros(len(df), dtype=float)

            groups = [("all", df.index)] if not group_col or group_col not in df.columns else df.groupby(group_col).groups.items()

            for _, indices in groups:
                series = df.loc[indices, col].to_numpy(dtype=float)
                m = len(series)

                for i in range(m):
                    start = max(0, i - w + 1)
                    sub_sig = series[start : i + 1]
                    metrics = compute_frequency_domain_metrics(sub_sig, sampling_rate=self.sampling_rate)

                    global_idx = indices[i]
                    dom_freqs[global_idx] = metrics["dominant_frequency"]
                    centroids[global_idx] = metrics["spectral_centroid"]
                    bandwidths[global_idx] = metrics["spectral_bandwidth"]
                    energies[global_idx] = metrics["spectral_energy"]

            results[f"{col}_dominant_freq"] = dom_freqs
            results[f"{col}_spectral_centroid"] = centroids
            results[f"{col}_spectral_bandwidth"] = bandwidths
            results[f"{col}_spectral_energy"] = energies

        return pd.DataFrame(results, index=df.index)


def extract_all_features(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    rolling_windows: Sequence[int] = (5, 10, 20),
    include_spectral: bool = True,
    spectral_sensors: Optional[List[str]] = None,
    fft_window_size: int = 32,
) -> pd.DataFrame:
    """Convenience pipeline applying all feature engineering steps to a DataFrame.

    Handles NaN propagation, group boundary protection, time-domain metrics,
    and sliding FFT frequency transformations.

    Args:
        df: Input DataFrame.
        sensor_cols: List of numerical columns to extract features from.
        group_col: Group column (e.g. 'machine_id' or 'unit_id').
        rolling_windows: Rolling statistics window sizes.
        include_spectral: Whether to calculate sliding FFT features.
        spectral_sensors: Specific subset of sensors for FFT (defaults to vibration/current/temp).
        fft_window_size: Window span for FFT spectral calculations.

    Returns:
        Expanded DataFrame with all extracted features and zero NaNs.
    """
    if df.empty:
        return df.copy()

    # Determine sensor columns automatically if omitted
    if sensor_cols is None:
        exclude_cols = {
            "timestamp", "machine_id", "unit_id", "cycle",
            "fault_label", "health_score", "rul", "label", "state"
        }
        sensor_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    engineer = FeatureEngineer(
        rolling_windows=rolling_windows,
        fft_window_size=fft_window_size,
    )

    logger.debug(f"Extracting rolling features for {len(sensor_cols)} sensors...")
    rolling_df = engineer.extract_rolling_features(df, sensor_cols, group_col=group_col)

    spectral_df = pd.DataFrame(index=df.index)
    if include_spectral:
        spec_targets = spectral_sensors or [s for s in sensor_cols if any(k in s.lower() for k in ["vib", "current", "rpm", "sensor_"])]
        if spec_targets:
            logger.debug(f"Extracting spectral features for {len(spec_targets)} sensors...")
            spectral_df = engineer.extract_spectral_sliding_features(df, spec_targets, group_col=group_col)

    combined_df = pd.concat([df, rolling_df, spectral_df], axis=1)

    # Impute NaNs created by rolling windows: forward fill, backward fill, then 0
    if group_col and group_col in combined_df.columns:
        combined_df = combined_df.groupby(group_col).apply(
            lambda g: g.ffill().bfill().fillna(0.0)
        ).reset_index(drop=True)
    else:
        combined_df = combined_df.ffill().bfill().fillna(0.0)

    # Clean up duplicate columns if any were introduced
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    logger.info(
        f"Feature engineering complete: {len(df.columns)} initial columns -> "
        f"{len(combined_df.columns)} total columns"
    )
    return combined_df
