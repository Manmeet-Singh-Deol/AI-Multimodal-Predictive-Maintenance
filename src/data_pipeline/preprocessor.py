"""Data Preprocessing, Normalization, Sliding Window Generator, and Augmentation.

Provides robust scaling (StandardScaler, MinMaxScaler), temporal train/val/test splitting
respecting unit-level operational history, 3D sequential window generation for deep learning,
and time-series data augmentation (Gaussian noise, time warping).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataScaler:
    """Normalization wrapper supporting StandardScaler and MinMaxScaler with persistence."""

    def __init__(self, scaler_type: str = "standard") -> None:
        """Initializes DataScaler.

        Args:
            scaler_type: Type of scaling ('standard' for zero-mean unit-variance,
                         'minmax' for [0, 1] range).
        """
        self.scaler_type = scaler_type.lower()
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        else:
            raise ValueError(f"Unknown scaler_type: '{scaler_type}'. Use 'standard' or 'minmax'.")

        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> DataScaler:
        """Fits scaler to training data.

        Args:
            X: Input numerical array or DataFrame.
            feature_names: Optional list of column names if X is an ndarray.

        Returns:
            Fitted self.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            arr = X.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            self.feature_names = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]

        self.scaler.fit(arr)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transforms input array or DataFrame using fitted statistics.

        Args:
            X: Input data matching the fitted feature dimension.

        Returns:
            Scaled numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("DataScaler must be fitted before transform can be called.")

        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)

        return self.scaler.transform(arr)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fits and transforms input data in a single step.

        Args:
            X: Input data.
            feature_names: Optional feature labels.

        Returns:
            Scaled numpy array.
        """
        return self.fit(X, feature_names=feature_names).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverts scaling transformation back to original physical scale.

        Args:
            X_scaled: Scaled numpy array.

        Returns:
            Array transformed back to original coordinates.
        """
        if not self.is_fitted:
            raise RuntimeError("DataScaler must be fitted before inverse_transform can be called.")
        return self.scaler.inverse_transform(X_scaled)

    def save(self, filepath: Union[str, Path]) -> Path:
        """Persists fitted scaler and feature names to disk via joblib.

        Args:
            filepath: Target file path (.joblib or .pkl).

        Returns:
            Path of saved artifact.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "scaler_type": self.scaler_type,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"Saved DataScaler to {path}")
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> DataScaler:
        """Loads a saved scaler from disk.

        Args:
            filepath: Path to saved joblib file.

        Returns:
            Loaded and fitted DataScaler instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Scaler checkpoint not found: {path}")

        data = joblib.load(path)
        inst = cls(scaler_type=data.get("scaler_type", "standard"))
        inst.scaler = data["scaler"]
        inst.feature_names = data.get("feature_names", [])
        inst.is_fitted = data.get("is_fitted", True)
        logger.info(f"Loaded DataScaler from {path} (type={inst.scaler_type}, features={len(inst.feature_names)})")
        return inst


def temporal_train_val_test_split(
    df: pd.DataFrame,
    group_col: Optional[str] = "machine_id",
    time_col: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits time-series data temporally (70/15/15) respecting unit sequences.

    If group_col is provided, the split is applied sequentially within each unit/machine,
    preserving chronological order and avoiding future information leakage.

    Args:
        df: Input DataFrame.
        group_col: Column distinguishing units/machines (e.g. 'machine_id', 'unit_id').
        time_col: Optional timestamp or cycle column for strict chronological sorting.
        train_ratio: Proportion allocated for training (default: 0.70).
        val_ratio: Proportion allocated for validation (default: 0.15).
        test_ratio: Proportion allocated for testing (default: 0.15).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    total_ratio = train_ratio + val_ratio + test_ratio
    train_pct = train_ratio / total_ratio
    val_pct = val_ratio / total_ratio

    df_sorted = df.copy()
    if group_col and group_col in df_sorted.columns:
        sort_keys = [group_col]
        if time_col and time_col in df_sorted.columns:
            sort_keys.append(time_col)
        df_sorted = df_sorted.sort_values(by=sort_keys).reset_index(drop=True)

        train_chunks = []
        val_chunks = []
        test_chunks = []

        for _, group in df_sorted.groupby(group_col):
            n = len(group)
            n_train = max(1, int(n * train_pct))
            n_val = max(1, int(n * val_pct))

            train_chunks.append(group.iloc[:n_train])
            val_chunks.append(group.iloc[n_train : n_train + n_val])
            test_chunks.append(group.iloc[n_train + n_val :])

        train_df = pd.concat(train_chunks, ignore_index=True)
        val_df = pd.concat(val_chunks, ignore_index=True)
        test_df = pd.concat(test_chunks, ignore_index=True)
    else:
        if time_col and time_col in df_sorted.columns:
            df_sorted = df_sorted.sort_values(by=time_col).reset_index(drop=True)

        n = len(df_sorted)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        train_df = df_sorted.iloc[:n_train].reset_index(drop=True)
        val_df = df_sorted.iloc[n_train : n_train + n_val].reset_index(drop=True)
        test_df = df_sorted.iloc[n_train + n_val :].reset_index(drop=True)

    logger.info(
        f"Temporal split complete: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )
    return train_df, val_df, test_df


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    sequence_length: int = 30,
    stride: int = 1,
    group_col: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generates 3D sliding window arrays (samples, sequence_length, features) for sequential models.

    Windows are strictly generated within unit/machine boundaries so that transitions
    across different machines are never merged. The target is assigned from the final
    timestep of each window.

    Args:
        df: Input DataFrame.
        feature_cols: List of column names used as sequential features.
        target_col: Optional target column (classification label or RUL value).
        sequence_length: Number of time steps per window.
        stride: Step size between consecutive sliding windows.
        group_col: Column indicating unit/machine grouping (e.g. 'machine_id', 'unit_id').

    Returns:
        Tuple of (X_windows, y_targets), where:
        X_windows has shape (samples, sequence_length, len(feature_cols))
        y_targets has shape (samples,) or None if target_col is omitted.
    """
    if len(df) < sequence_length:
        logger.warning(f"DataFrame size ({len(df)}) is smaller than sequence_length ({sequence_length}).")
        empty_x = np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32) if target_col else None
        return empty_x, empty_y

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    groups = [("all", df)] if not group_col or group_col not in df.columns else df.groupby(group_col)

    for _, group in groups:
        if len(group) < sequence_length:
            continue

        feat_arr = group[feature_cols].to_numpy(dtype=np.float32)
        n_rows = len(feat_arr)

        target_arr = group[target_col].to_numpy() if target_col and target_col in group.columns else None

        for start_idx in range(0, n_rows - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            all_x.append(feat_arr[start_idx:end_idx])
            if target_arr is not None:
                all_y.append(target_arr[end_idx - 1])

    if not all_x:
        empty_x = np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32) if target_col else None
        return empty_x, empty_y

    X_windows = np.stack(all_x, axis=0)
    y_targets = np.array(all_y) if target_col is not None else None

    logger.debug(
        f"Sliding windows created: X shape={X_windows.shape}, "
        f"y shape={y_targets.shape if y_targets is not None else None}"
    )
    return X_windows, y_targets


def add_gaussian_noise(
    X: np.ndarray,
    noise_factor: float = 0.02,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Augments 3D sequence array by adding zero-mean Gaussian jitter.

    Args:
        X: 3D array of shape (samples, sequence_length, features).
        noise_factor: Noise standard deviation relative to per-feature standard deviation.
        random_seed: Seed for random generator.

    Returns:
        Augmented array with added Gaussian noise.
    """
    if noise_factor <= 0.0:
        return X.copy()

    rng = np.random.RandomState(random_seed)
    std_per_feature = np.std(X, axis=(0, 1), keepdims=True)
    std_per_feature[std_per_feature == 0] = 1.0  # Guard against constant features

    noise = rng.normal(0.0, noise_factor * std_per_feature, size=X.shape)
    return (X + noise).astype(X.dtype)


def time_warp(
    X: np.ndarray,
    sigma: float = 0.2,
    num_knots: int = 4,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Augments 3D sequence array by smoothly warping the time axis via cubic splines.

    Args:
        X: 3D array of shape (samples, sequence_length, features).
        sigma: Standard deviation of knot perturbations along the time axis.
        num_knots: Number of internal spline control points.
        random_seed: Seed for reproducible augmentation.

    Returns:
        Time-warped sequence array matching original shape.
    """
    if sigma <= 0.0 or len(X) == 0:
        return X.copy()

    rng = np.random.RandomState(random_seed)
    samples, seq_len, features = X.shape
    warped_X = np.empty_like(X)

    orig_steps = np.linspace(0, 1, seq_len)

    for i in range(samples):
        # Sample random anchor shifts
        knot_positions = np.linspace(0, 1, num_knots + 2)
        knot_values = knot_positions + rng.normal(0.0, sigma, size=num_knots + 2)
        knot_values[0] = 0.0
        knot_values[-1] = 1.0
        # Ensure monotonicity
        knot_values = np.sort(knot_values)

        spline = CubicSpline(knot_positions, knot_values)
        warped_steps = np.clip(spline(orig_steps), 0.0, 1.0)
        # Rescale back to index space [0, seq_len - 1]
        warped_indices = warped_steps * (seq_len - 1)

        for f in range(features):
            warped_X[i, :, f] = np.interp(warped_indices, np.arange(seq_len), X[i, :, f])

    return warped_X.astype(X.dtype)


class DataPreprocessor:
    """Full-cycle preprocessor orchestrating scaling, windowing, and augmentation."""

    def __init__(
        self,
        scaler_type: str = "standard",
        sequence_length: int = 30,
        stride: int = 1,
    ) -> None:
        """Initializes DataPreprocessor.

        Args:
            scaler_type: 'standard' or 'minmax'.
            sequence_length: Number of time steps per window.
            stride: Window stride.
        """
        self.scaler = DataScaler(scaler_type=scaler_type)
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> DataPreprocessor:
        """Fits scaler to specified feature columns in DataFrame."""
        self.feature_cols = list(feature_cols)
        self.scaler.fit(df[self.feature_cols])
        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales feature columns within a DataFrame while leaving metadata columns intact."""
        df_out = df.copy()
        scaled_vals = self.scaler.transform(df_out[self.feature_cols])
        df_out[self.feature_cols] = scaled_vals
        return df_out

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        group_col: Optional[str] = None,
        fit_scaler: bool = False,
        augment_noise: float = 0.0,
        augment_warp: float = 0.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Scales DataFrame and extracts 3D sequence arrays with optional augmentation.

        Args:
            df: Input data.
            target_col: Label or target column.
            group_col: Unit/machine grouping column.
            fit_scaler: If True, fits scaler on df before transforming.
            augment_noise: Factor for Gaussian jitter augmentation (0.0 to disable).
            augment_warp: Sigma for time warping augmentation (0.0 to disable).

        Returns:
            Tuple of (X_windows, y_targets).
        """
        if fit_scaler:
            self.fit(df, self.feature_cols)

        scaled_df = self.transform_df(df)
        X_seq, y = create_sliding_windows(
            df=scaled_df,
            feature_cols=self.feature_cols,
            target_col=target_col,
            sequence_length=self.sequence_length,
            stride=self.stride,
            group_col=group_col,
        )

        # Apply augmentation if requested
        if augment_noise > 0.0 and len(X_seq) > 0:
            noisy_X = add_gaussian_noise(X_seq, noise_factor=augment_noise)
            X_seq = np.concatenate([X_seq, noisy_X], axis=0)
            if y is not None:
                y = np.concatenate([y, y], axis=0)

        if augment_warp > 0.0 and len(X_seq) > 0:
            warped_X = time_warp(X_seq[:len(X_seq)//(2 if augment_noise > 0 else 1)], sigma=augment_warp)
            X_seq = np.concatenate([X_seq, warped_X], axis=0)
            if y is not None:
                y = np.concatenate([y, y[:len(warped_X)]], axis=0)

        return X_seq, y

    def save(self, filepath: Union[str, Path]) -> Path:
        """Saves scaler state and preprocessor configurations."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "sequence_length": self.sequence_length,
                "stride": self.stride,
                "feature_cols": self.feature_cols,
            },
            path,
        )
        logger.info(f"Saved DataPreprocessor to {path}")
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> DataPreprocessor:
        """Loads saved preprocessor."""
        path = Path(filepath)
        data = joblib.load(path)
        inst = cls(
            scaler_type=data["scaler"].scaler_type,
            sequence_length=data["sequence_length"],
            stride=data["stride"],
        )
        inst.scaler = data["scaler"]
        inst.feature_cols = data["feature_cols"]
        logger.info(f"Loaded DataPreprocessor from {path}")
        return inst
