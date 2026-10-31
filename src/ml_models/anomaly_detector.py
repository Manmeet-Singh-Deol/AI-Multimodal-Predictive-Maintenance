"""Anomaly Detection Models for Industrial Equipment Health Monitoring.

Implements IsolationForest, PyTorch Deep Autoencoder with statistical 3-sigma
reconstruction thresholding, and an Ensemble Anomaly Detector combining both modalities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset

from config.settings import get_settings


class IsolationForestDetector:
    """Unsupervised anomaly detector wrapping scikit-learn's Isolation Forest."""

    def __init__(
        self,
        contamination: Optional[float] = None,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        """Initializes IsolationForestDetector.

        Args:
            contamination: Expected proportion of outliers in the data.
            n_estimators: Number of decision trees in the ensemble.
            random_state: Random state for reproducibility.
        """
        settings = get_settings()
        self.contamination = contamination if contamination is not None else settings.anomaly_contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted: bool = False
        self.score_min: float = 0.0
        self.score_max: float = 1.0

    def train(self, X: Union[pd.DataFrame, np.ndarray]) -> IsolationForestDetector:
        """Trains the Isolation Forest detector on normal operational telemetry.

        Args:
            X: Training feature matrix (samples, features).

        Returns:
            Fitted detector instance.
        """
        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        logger.info(f"Training IsolationForest on {arr.shape[0]} normal samples with {arr.shape[1]} features...")
        self.model.fit(arr)
        self.is_fitted = True

        # Calibrate score bounds for normalization
        raw_scores = -self.model.decision_function(arr)
        self.score_min = float(np.min(raw_scores))
        self.score_max = float(np.max(raw_scores)) if np.max(raw_scores) > self.score_min else self.score_min + 1.0

        logger.info(f"IsolationForest fitted. Score range: [{self.score_min:.4f}, {self.score_max:.4f}]")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts anomaly status and anomaly scores.

        Args:
            X: Input telemetry matrix.

        Returns:
            Tuple of (anomaly_scores, binary_labels), where binary_labels is 1 for anomaly, 0 for normal.
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector must be trained prior to inference.")

        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        # raw decision function: higher means more normal, so negate
        raw_scores = -self.model.decision_function(arr)
        norm_scores = np.clip((raw_scores - self.score_min) / (self.score_max - self.score_min + 1e-9), 0.0, 1.0)

        # Sklearn returns -1 for outlier, 1 for inlier
        raw_preds = self.model.predict(arr)
        binary_labels = np.where(raw_preds == -1, 1, 0)
        return norm_scores, binary_labels

    def save(self, filepath: Union[str, Path]) -> Path:
        """Saves trained IsolationForest model to disk via joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "is_fitted": self.is_fitted,
                "score_min": self.score_min,
                "score_max": self.score_max,
            },
            path,
        )
        logger.info(f"Saved IsolationForestDetector to {path}")
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> IsolationForestDetector:
        """Loads trained IsolationForest model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"IsolationForest model file not found: {path}")

        data = joblib.load(path)
        inst = cls(
            contamination=data["contamination"],
            n_estimators=data["n_estimators"],
            random_state=data["random_state"],
        )
        inst.model = data["model"]
        inst.is_fitted = data["is_fitted"]
        inst.score_min = data.get("score_min", 0.0)
        inst.score_max = data.get("score_max", 1.0)
        logger.info(f"Loaded IsolationForestDetector from {path}")
        return inst


class AutoencoderNet(nn.Module):
    """PyTorch Deep Autoencoder architecture: Input -> 64 -> 32 -> latent -> 32 -> 64 -> Input."""

    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        """Initializes Autoencoder neural network.

        Args:
            input_dim: Number of input features.
            latent_dim: Dimension of latent bottleneck representation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes and reconstructs input tensor."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector:
    """Deep Autoencoder Anomaly Detector with dynamic 3-sigma reconstruction thresholding."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        threshold_sigma: float = 3.0,
        device: Optional[str] = None,
    ) -> None:
        """Initializes AutoencoderDetector.

        Args:
            input_dim: Feature dimension.
            latent_dim: Bottleneck dimension (default from settings).
            threshold_sigma: Number of standard deviations above mean error for anomaly threshold.
            device: 'cuda' or 'cpu'. Defaults to auto-detect.
        """
        settings = get_settings()
        self.input_dim = input_dim
        self.latent_dim = latent_dim if latent_dim is not None else settings.autoencoder_latent_dim
        self.threshold_sigma = threshold_sigma

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[AutoencoderNet] = None
        if self.input_dim is not None:
            self.model = AutoencoderNet(input_dim=self.input_dim, latent_dim=self.latent_dim).to(self.device)

        self.threshold: float = 0.0
        self.train_mean_error: float = 0.0
        self.train_std_error: float = 0.0
        self.is_fitted: bool = False

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        val_split: float = 0.15,
    ) -> AutoencoderDetector:
        """Trains Autoencoder on normal baseline data and computes the 3-sigma threshold.

        Args:
            X: Input telemetry matrix of healthy/normal operating conditions.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            val_split: Validation fraction for monitoring generalization.

        Returns:
            Fitted detector instance.
        """
        settings = get_settings()
        num_epochs = epochs or settings.autoencoder_epochs
        b_size = batch_size or settings.autoencoder_batch_size
        learning_rate = lr or settings.autoencoder_lr

        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        num_samples, num_features = arr.shape

        if self.model is None or self.input_dim != num_features:
            self.input_dim = num_features
            self.model = AutoencoderNet(input_dim=self.input_dim, latent_dim=self.latent_dim).to(self.device)

        # Train/validation split for threshold calibration
        val_size = int(num_samples * val_split)
        train_size = num_samples - val_size

        indices = np.random.permutation(num_samples)
        train_idx, val_idx = indices[:train_size], indices[train_size:]

        train_tensor = torch.tensor(arr[train_idx], dtype=torch.float32)
        val_tensor = torch.tensor(arr[val_idx], dtype=torch.float32) if val_size > 0 else train_tensor

        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=b_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        logger.info(
            f"Training Autoencoder on {train_size} samples (features={num_features}, "
            f"latent={self.latent_dim}, epochs={num_epochs})..."
        )

        self.model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            epoch_loss = total_loss / train_size
            if epoch % 10 == 0 or epoch == num_epochs:
                logger.debug(f"Autoencoder Epoch [{epoch}/{num_epochs}] - Train Loss: {epoch_loss:.6f}")

        # Compute reconstruction error on validation set (or training set)
        self.model.eval()
        with torch.no_grad():
            val_in = val_tensor.to(self.device)
            val_recon = self.model(val_in)
            # Sample-wise Mean Squared Error
            sample_errors = torch.mean((val_in - val_recon) ** 2, dim=1).cpu().numpy()

        self.train_mean_error = float(np.mean(sample_errors))
        self.train_std_error = float(np.std(sample_errors))
        self.threshold = self.train_mean_error + self.threshold_sigma * self.train_std_error
        self.is_fitted = True

        logger.info(
            f"Autoencoder training complete. Reconstruction error: mean={self.train_mean_error:.6f}, "
            f"std={self.train_std_error:.6f}, 3-sigma Threshold={self.threshold:.6f}"
        )
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates reconstruction errors and predicts binary anomaly status.

        Args:
            X: Input telemetry matrix.

        Returns:
            Tuple of (reconstruction_errors, binary_labels), where 1 indicates anomaly.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("AutoencoderDetector must be trained before predicting.")

        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        tensor_x = torch.tensor(arr, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            sample_errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1).cpu().numpy()

        binary_labels = (sample_errors > self.threshold).astype(int)
        return sample_errors, binary_labels

    def save(self, filepath: Union[str, Path]) -> Path:
        """Saves model weights, architecture parameters, and statistical threshold."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted AutoencoderDetector.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "threshold_sigma": self.threshold_sigma,
                "threshold": self.threshold,
                "train_mean_error": self.train_mean_error,
                "train_std_error": self.train_std_error,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"Saved AutoencoderDetector to {path}")
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[str] = None) -> AutoencoderDetector:
        """Loads trained Autoencoder detector checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Autoencoder model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        inst = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            threshold_sigma=checkpoint.get("threshold_sigma", 3.0),
            device=device,
        )
        inst.model.load_state_dict(checkpoint["state_dict"])
        inst.model.to(inst.device)
        inst.threshold = checkpoint["threshold"]
        inst.train_mean_error = checkpoint["train_mean_error"]
        inst.train_std_error = checkpoint["train_std_error"]
        inst.is_fitted = checkpoint["is_fitted"]
        logger.info(f"Loaded AutoencoderDetector from {path} (threshold={inst.threshold:.6f})")
        return inst


class EnsembleDetector:
    """Ensemble combining IsolationForest and PyTorch Deep Autoencoder."""

    def __init__(
        self,
        weight_if: float = 0.5,
        weight_ae: float = 0.5,
        contamination: Optional[float] = None,
        latent_dim: Optional[int] = None,
    ) -> None:
        """Initializes EnsembleDetector.

        Args:
            weight_if: Relative voting weight for Isolation Forest.
            weight_ae: Relative voting weight for Autoencoder.
            contamination: Isolation Forest contamination parameter.
            latent_dim: Autoencoder bottleneck dimension.
        """
        self.weight_if = weight_if
        self.weight_ae = weight_ae
        self.if_detector = IsolationForestDetector(contamination=contamination)
        self.ae_detector = AutoencoderDetector(latent_dim=latent_dim)
        self.ensemble_threshold: float = 0.5
        self.is_fitted: bool = False

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> EnsembleDetector:
        """Fits both underlying detectors on healthy baseline telemetry.

        Args:
            X: Input training data.
            epochs: Autoencoder training epochs.
            batch_size: Autoencoder batch size.

        Returns:
            Fitted ensemble instance.
        """
        logger.info("Training Ensemble Anomaly Detector...")
        self.if_detector.train(X)
        self.ae_detector.train(X, epochs=epochs, batch_size=batch_size)

        # Calibrate ensemble threshold on training baseline
        if_scores, _ = self.if_detector.predict(X)
        ae_errors, _ = self.ae_detector.predict(X)
        ae_scores = np.clip(ae_errors / (self.ae_detector.threshold + 1e-9), 0.0, 2.0) / 2.0

        total_weight = self.weight_if + self.weight_ae
        combined = (self.weight_if * if_scores + self.weight_ae * ae_scores) / total_weight

        # Set ensemble threshold at 95th percentile of normal baseline
        self.ensemble_threshold = float(np.percentile(combined, 95))
        self.is_fitted = True
        logger.info(f"EnsembleDetector fitted. Calibrated threshold: {self.ensemble_threshold:.4f}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates blended anomaly score and binary decision.

        Args:
            X: Telemetry matrix.

        Returns:
            Tuple of (combined_anomaly_scores, binary_labels).
        """
        if not self.is_fitted:
            raise RuntimeError("EnsembleDetector must be trained prior to inference.")

        if_scores, if_labels = self.if_detector.predict(X)
        ae_errors, ae_labels = self.ae_detector.predict(X)
        ae_scores = np.clip(ae_errors / (self.ae_detector.threshold + 1e-9), 0.0, 2.0) / 2.0

        total_weight = self.weight_if + self.weight_ae
        combined_scores = (self.weight_if * if_scores + self.weight_ae * ae_scores) / total_weight

        # Outlier if combined score exceeds threshold or if both models agree
        binary_labels = np.where(
            (combined_scores > self.ensemble_threshold) | ((if_labels == 1) & (ae_labels == 1)),
            1,
            0,
        )
        return combined_scores, binary_labels

    def save(self, dir_path: Union[str, Path]) -> Path:
        """Saves both component models and ensemble configuration to a directory."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self.if_detector.save(path / "isolation_forest.joblib")
        self.ae_detector.save(path / "autoencoder.pt")
        joblib.dump(
            {
                "weight_if": self.weight_if,
                "weight_ae": self.weight_ae,
                "ensemble_threshold": self.ensemble_threshold,
                "is_fitted": self.is_fitted,
            },
            path / "ensemble_meta.joblib",
        )
        logger.info(f"Saved EnsembleDetector components to {path}")
        return path

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> EnsembleDetector:
        """Loads ensemble detector from artifact directory."""
        path = Path(dir_path)
        meta = joblib.load(path / "ensemble_meta.joblib")
        inst = cls(weight_if=meta["weight_if"], weight_ae=meta["weight_ae"])
        inst.if_detector = IsolationForestDetector.load(path / "isolation_forest.joblib")
        inst.ae_detector = AutoencoderDetector.load(path / "autoencoder.pt")
        inst.ensemble_threshold = meta["ensemble_threshold"]
        inst.is_fitted = meta["is_fitted"]
        logger.info(f"Loaded EnsembleDetector from {path}")
        return inst
