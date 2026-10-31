"""Remaining Useful Life (RUL) Estimation Models.

Implements a Bidirectional LSTM (Bi-LSTM) neural network for predictive degradation modeling.
Includes Monte Carlo Dropout uncertainty estimation for confidence intervals,
piece-wise linear RUL thresholding (capped at max_rul=125), and NASA C-MAPSS
asymmetric scoring function evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from config.settings import get_settings


def compute_cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the official NASA C-MAPSS asymmetric evaluation score.

    The asymmetric score heavily penalizes late predictions (where predicted RUL > true RUL)
    more severely than early predictions (where predicted RUL < true RUL) to reflect
    catastrophic industrial failure risk:
        d_i = y_pred_i - y_true_i
        score_i = exp(-d_i / 13) - 1   if d_i < 0  (early prediction)
        score_i = exp(d_i / 10) - 1    if d_i >= 0 (late prediction)

    Args:
        y_true: 1D array of ground truth Remaining Useful Life cycles.
        y_pred: 1D array of predicted Remaining Useful Life cycles.

    Returns:
        Cumulative asymmetric penalty score (lower is superior).
    """
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))


class RULBiLSTM(nn.Module):
    """Bidirectional LSTM model for continuous RUL trajectory estimation.

    Architecture:
        BiLSTM(input_size, hidden_size=128, num_layers=2, dropout=0.3)
        -> FC(256 -> 128) -> ReLU -> Dropout(0.3) -> FC(128 -> 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initializes RULBiLSTM model.

        Args:
            input_size: Number of input features per time step.
            hidden_size: Hidden state dimension of LSTM units.
            num_layers: Number of stacked bidirectional LSTM layers.
            dropout: Dropout probability applied between LSTM layers and FC.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dense regression head taking both forward and backward final hidden representations (128 * 2 = 256)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Bi-LSTM.

        Args:
            x: Sequence tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Predicted RUL scalar tensor of shape (batch_size, 1).
        """
        # lstm_out: (batch_size, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)

        # Extract last time-step hidden states from forward and backward passes
        last_step_features = lstm_out[:, -1, :]  # Shape: (batch_size, 256)

        out = self.fc1(last_step_features)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # Shape: (batch_size, 1)
        return out


class RULEstimatorWrapper:
    """Production wrapper for training, predicting with uncertainty, and evaluating Bi-LSTM RUL."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        max_rul: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes RULEstimatorWrapper.

        Args:
            input_size: Number of features per time step.
            hidden_size: Number of hidden units per direction.
            num_layers: Number of stacked BiLSTM layers.
            max_rul: Maximum RUL ceiling for piece-wise saturation. Defaults to settings.rul_max_rul.
            device: 'cuda' or 'cpu'. Defaults to auto-detect.
        """
        settings = get_settings()
        self.input_size = input_size
        self.hidden_size = hidden_size or settings.rul_hidden_size
        self.num_layers = num_layers or settings.rul_num_layers
        self.max_rul = max_rul if max_rul is not None else settings.rul_max_rul

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[RULBiLSTM] = None
        if self.input_size is not None:
            self.model = RULBiLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.3,
            ).to(self.device)

        self.is_fitted: bool = False

    def train(
        self,
        X_sequences: np.ndarray,
        y_rul: Union[np.ndarray, pd.Series, List[float]],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        val_data: Optional[Tuple[np.ndarray, Any]] = None,
        val_split: float = 0.15,
        patience: int = 10,
        loss_fn: str = "huber",
    ) -> Dict[str, List[float]]:
        """Trains Bi-LSTM with Adam, LR scheduling, and early stopping.

        Applies piece-wise linear capping at max_rul to ground truth targets.

        Args:
            X_sequences: 3D sequence array of shape (samples, sequence_length, features).
            y_rul: Target RUL cycles.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            val_data: Optional tuple of (X_val, y_val).
            val_split: Validation fraction if val_data is omitted.
            patience: Early stopping patience.
            loss_fn: Loss objective ('huber' for outlier robustness, or 'mse').

        Returns:
            Dictionary with training history ('train_loss', 'val_loss', 'val_rmse').
        """
        settings = get_settings()
        num_epochs = epochs or settings.rul_epochs
        b_size = batch_size or settings.rul_batch_size
        learning_rate = lr or settings.rul_lr

        X_arr = np.asarray(X_sequences, dtype=np.float32)
        # Apply piece-wise linear saturation
        y_capped = np.minimum(np.asarray(y_rul, dtype=np.float32), float(self.max_rul))

        # Setup model dimensions if needed
        features_dim = X_arr.shape[2]
        if self.model is None or self.input_size != features_dim:
            self.input_size = features_dim
            self.model = RULBiLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.3,
            ).to(self.device)

        # Train/validation split
        if val_data is not None:
            X_tr, y_tr = X_arr, y_capped
            X_v = np.asarray(val_data[0], dtype=np.float32)
            y_v = np.minimum(np.asarray(val_data[1], dtype=np.float32), float(self.max_rul))
        else:
            num_samples = len(X_arr)
            val_count = int(num_samples * val_split)
            indices = np.random.permutation(num_samples)
            train_idx, val_idx = indices[val_count:], indices[:val_count]
            X_tr, y_tr = X_arr[train_idx], y_capped[train_idx]
            X_v, y_v = X_arr[val_idx], y_capped[val_idx]

        train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1))
        val_dataset = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v).unsqueeze(1))

        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)

        criterion = nn.HuberLoss(delta=1.0) if loss_fn.lower() == "huber" else nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        logger.info(
            f"Training RULBiLSTM on {len(X_tr)} sequences (val={len(X_v)}, "
            f"features={self.input_size}, hidden={self.hidden_size}, epochs={num_epochs})..."
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_rmse": []}
        best_val_loss = float("inf")
        best_weights: Optional[Dict[str, Any]] = None
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                total_train_loss += loss.item() * len(bx)

            avg_train_loss = total_train_loss / max(1, len(X_tr))

            # Validation evaluation
            self.model.eval()
            total_val_loss = 0.0
            val_preds_list = []
            val_true_list = []

            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    pred = self.model(bx)
                    loss = criterion(pred, by)
                    total_val_loss += loss.item() * len(bx)
                    val_preds_list.extend(pred.squeeze(1).cpu().numpy())
                    val_true_list.extend(by.squeeze(1).cpu().numpy())

            avg_val_loss = total_val_loss / max(1, len(X_v))
            val_rmse = float(np.sqrt(mean_squared_error(val_true_list, val_preds_list)))
            scheduler.step(avg_val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_rmse"].append(val_rmse)

            if epoch % 10 == 0 or epoch == num_epochs:
                logger.info(
                    f"RUL Estimator Epoch [{epoch:03d}/{num_epochs:03d}] - "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val RMSE: {val_rmse:.2f} cycles"
                )

            # Checkpoint & early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.is_fitted = True
        return history

    def predict(
        self,
        X_sequence: np.ndarray,
        num_mc_samples: int = 20,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predicts Remaining Useful Life cycles with Monte Carlo Dropout confidence intervals.

        Args:
            X_sequence: 3D sequence array (samples, sequence_length, features).
            num_mc_samples: Number of forward stochastic passes for uncertainty estimation.

        Returns:
            Tuple of:
            - predicted_rul_cycles: 1D array of expected RUL values in cycles (clipped >= 0)
            - confidence_interval: Tuple of (lower_bound_95, upper_bound_95)
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("RULEstimatorWrapper must be trained prior to inference.")

        X_arr = np.asarray(X_sequence, dtype=np.float32)
        tensor_x = torch.from_numpy(X_arr).to(self.device)

        # Enable dropout during inference for Monte Carlo uncertainty sampling
        self.model.train()
        mc_predictions = []

        with torch.no_grad():
            for _ in range(num_mc_samples):
                preds = self.model(tensor_x).squeeze(1).cpu().numpy()
                mc_predictions.append(preds)

        mc_array = np.array(mc_predictions)  # Shape: (num_mc_samples, num_samples)
        mean_rul = np.mean(mc_array, axis=0)
        std_rul = np.std(mc_array, axis=0)

        # 95% Confidence Interval: mean +/- 1.96 * std
        lower_bound = np.clip(mean_rul - 1.96 * std_rul, 0.0, float(self.max_rul))
        upper_bound = np.clip(mean_rul + 1.96 * std_rul, 0.0, float(self.max_rul))
        mean_clipped = np.clip(mean_rul, 0.0, float(self.max_rul))

        return mean_clipped, (lower_bound, upper_bound)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Union[np.ndarray, pd.Series, List[float]],
    ) -> Dict[str, float]:
        """Evaluates model performance against test labels using RMSE, MAE, R2, and NASA asymmetric score.

        Args:
            X_test: 3D sequence array of test trajectories.
            y_test: Ground truth Remaining Useful Life values.

        Returns:
            Dictionary with 'rmse', 'mae', 'r2', and 'scoring_function'.
        """
        y_true_capped = np.minimum(np.asarray(y_test, dtype=float), float(self.max_rul))
        y_pred, _ = self.predict(X_test, num_mc_samples=10)

        rmse = float(np.sqrt(mean_squared_error(y_true_capped, y_pred)))
        mae = float(mean_absolute_error(y_true_capped, y_pred))
        r2 = float(r2_score(y_true_capped, y_pred))
        asym_score = compute_cmapss_score(y_true_capped, y_pred)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "scoring_function": asym_score,
        }
        logger.info(
            f"RUL Evaluation: RMSE={rmse:.2f} cycles, MAE={mae:.2f} cycles, "
            f"R2={r2:.4f}, NASA Score={asym_score:.2f}"
        )
        return metrics

    def save(self, filepath: Union[str, Path]) -> Path:
        """Persists trained model weights and hyperparameter specifications."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted RULEstimatorWrapper.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "max_rul": self.max_rul,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"Saved RULEstimatorWrapper checkpoint to {path}")
        return path

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        device: Optional[str] = None,
    ) -> RULEstimatorWrapper:
        """Loads trained Bi-LSTM RUL model from checkpoint."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"RUL model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        inst = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            max_rul=checkpoint.get("max_rul", 125),
            device=device,
        )
        inst.model.load_state_dict(checkpoint["state_dict"])
        inst.model.to(inst.device)
        inst.is_fitted = checkpoint["is_fitted"]
        logger.info(f"Loaded RULEstimatorWrapper from {path} (hidden={inst.hidden_size}, layers={inst.num_layers})")
        return inst
