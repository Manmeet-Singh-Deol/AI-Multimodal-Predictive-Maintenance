"""Industrial Equipment Fault Classification Model.

Implements a 1D Convolutional Neural Network (1D-CNN) for multi-class fault classification
across 5 machine states: Normal, Bearing_Fault, Overheating, Misalignment, and Electrical_Fault.
Includes evaluation metrics, confidence scoring, probability outputs, and full persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from config.settings import get_settings

DEFAULT_CLASSES: List[str] = [
    "Normal",
    "Bearing_Fault",
    "Overheating",
    "Misalignment",
    "Electrical_Fault",
]


class FaultClassifier1DCNN(nn.Module):
    """1D-CNN architecture for multi-sensor time-series fault classification.

    Architecture:
        Conv1d(in_channels, 64, k=5) -> BN -> ReLU -> MaxPool(2)
        -> Conv1d(64, 128, k=3) -> BN -> ReLU -> MaxPool(2)
        -> Conv1d(128, 64, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)
        -> Flatten -> FC(64) -> ReLU -> Dropout(0.3) -> FC(num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int = 5) -> None:
        """Initializes 1D-CNN fault classification network.

        Args:
            in_channels: Number of sensor telemetry feature channels.
            num_classes: Number of target fault classification categories.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 1D-CNN.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, in_channels)
               or (batch_size, in_channels, sequence_length).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # Ensure tensor is in channels-first format (B, C, L)
        if x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.permute(0, 2, 1)

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        logits = self.classifier(out)
        return logits


class FaultClassifierWrapper:
    """Production wrapper for training, predicting, and evaluating FaultClassifier1DCNN."""

    def __init__(
        self,
        in_channels: Optional[int] = None,
        classes: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes FaultClassifierWrapper.

        Args:
            in_channels: Number of input features per time step.
            classes: Ordered list of class names. Defaults to 5 standard machine states.
            device: 'cuda' or 'cpu'. Defaults to auto-detect.
        """
        self.classes = classes or DEFAULT_CLASSES
        self.num_classes = len(self.classes)
        self.label_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_label: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}

        self.in_channels = in_channels
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[FaultClassifier1DCNN] = None
        if self.in_channels is not None:
            self.model = FaultClassifier1DCNN(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            ).to(self.device)

        self.is_fitted: bool = False

    def _encode_labels(self, y: Union[np.ndarray, pd.Series, List[Any]]) -> np.ndarray:
        """Encodes string or categorical labels to integer class IDs."""
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.integer):
            return y_arr.astype(np.int64)

        encoded = []
        for val in y_arr:
            str_val = str(val).strip()
            if str_val in self.label_to_idx:
                encoded.append(self.label_to_idx[str_val])
            else:
                # Fallback to closest match or class 0
                logger.warning(f"Unrecognized class '{str_val}', mapping to 0 ({self.classes[0]})")
                encoded.append(0)
        return np.array(encoded, dtype=np.int64)

    def train(
        self,
        X_windows: np.ndarray,
        y_labels: Union[np.ndarray, pd.Series, List[Any]],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        val_data: Optional[Tuple[np.ndarray, Any]] = None,
        val_split: float = 0.15,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """Trains 1D-CNN fault classifier using Adam optimizer and CrossEntropyLoss.

        Args:
            X_windows: 3D sequence array (samples, sequence_length, features).
            y_labels: 1D array of target class strings or integers.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            val_data: Optional pre-split validation tuple (X_val, y_val).
            val_split: Validation fraction if val_data is omitted.
            patience: Early stopping patience.

        Returns:
            Dictionary with training history ('train_loss', 'val_loss', 'val_acc').
        """
        settings = get_settings()
        num_epochs = epochs or settings.fault_classifier_epochs
        b_size = batch_size or settings.fault_classifier_batch_size
        learning_rate = lr or settings.fault_classifier_lr

        X_arr = np.asarray(X_windows, dtype=np.float32)
        y_encoded = self._encode_labels(y_labels)

        # Initialize model if not already set
        features_dim = X_arr.shape[2] if X_arr.ndim == 3 else X_arr.shape[1]
        if self.model is None or self.in_channels != features_dim:
            self.in_channels = features_dim
            self.model = FaultClassifier1DCNN(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            ).to(self.device)

        # Handle train/val split
        if val_data is not None:
            X_tr, y_tr = X_arr, y_encoded
            X_v, y_v = np.asarray(val_data[0], dtype=np.float32), self._encode_labels(val_data[1])
        else:
            num_samples = len(X_arr)
            val_count = int(num_samples * val_split)
            indices = np.random.permutation(num_samples)
            train_idx, val_idx = indices[val_count:], indices[:val_count]
            X_tr, y_tr = X_arr[train_idx], y_encoded[train_idx]
            X_v, y_v = X_arr[val_idx], y_encoded[val_idx]

        train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_dataset = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v))

        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        logger.info(
            f"Training FaultClassifier1DCNN on {len(X_tr)} windows (val={len(X_v)}, "
            f"features={self.in_channels}, classes={self.num_classes}, epochs={num_epochs})..."
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_weights: Optional[Dict[str, Any]] = None
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(bx)

            avg_train_loss = total_train_loss / max(1, len(X_tr))

            # Validation pass
            self.model.eval()
            total_val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    logits = self.model(bx)
                    loss = criterion(logits, by)
                    total_val_loss += loss.item() * len(bx)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == by).sum().item()

            avg_val_loss = total_val_loss / max(1, len(X_v))
            val_acc = correct / max(1, len(X_v))
            scheduler.step(avg_val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

            if epoch % 10 == 0 or epoch == num_epochs:
                logger.info(
                    f"Fault Classifier Epoch [{epoch:03d}/{num_epochs:03d}] - "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}"
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

    def predict(self, X_windows: np.ndarray) -> Dict[str, Any]:
        """Performs inference and returns predicted class names, probabilities, and confidence.

        Args:
            X_windows: 3D sequence array (samples, sequence_length, features).

        Returns:
            Dictionary with keys:
            - 'predicted_class': List of predicted class strings
            - 'predicted_idx': 1D numpy array of class indices
            - 'probabilities': 2D numpy array of class probabilities (samples, num_classes)
            - 'confidence': 1D numpy array of maximum probability scores
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("FaultClassifierWrapper must be trained prior to inference.")

        X_arr = np.asarray(X_windows, dtype=np.float32)
        tensor_x = torch.from_numpy(X_arr).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        predicted_idx = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        predicted_classes = [self.idx_to_label.get(int(idx), "Unknown") for idx in predicted_idx]

        return {
            "predicted_class": np.array(predicted_classes),
            "predicted_idx": predicted_idx,
            "probabilities": probs,
            "confidence": confidence,
        }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Union[np.ndarray, pd.Series, List[Any]],
    ) -> Dict[str, Any]:
        """Evaluates model performance on test partition.

        Args:
            X_test: Test 3D sequence array.
            y_test: True class labels.

        Returns:
            Dictionary with accuracy, precision, recall, f1, confusion_matrix,
            and classification_report.
        """
        y_true = self._encode_labels(y_test)
        preds = self.predict(X_test)
        y_pred = preds["predicted_idx"]

        acc = float(accuracy_score(y_true, y_pred))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            target_names=self.classes,
            zero_division=0,
            output_dict=True,
        )

        metrics = {
            "accuracy": acc,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": cm,
            "classification_report": report,
        }
        logger.info(
            f"Fault Classifier Evaluation: Accuracy={acc:.2%}, "
            f"F1 (Macro)={f1_macro:.4f}, F1 (Weighted)={f1_weighted:.4f}"
        )
        return metrics

    def save(self, filepath: Union[str, Path]) -> Path:
        """Persists trained model checkpoint and metadata to disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted FaultClassifierWrapper.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "in_channels": self.in_channels,
                "classes": self.classes,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"Saved FaultClassifierWrapper checkpoint to {path}")
        return path

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        device: Optional[str] = None,
    ) -> FaultClassifierWrapper:
        """Loads trained fault classifier checkpoint."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Fault classifier checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        inst = cls(
            in_channels=checkpoint["in_channels"],
            classes=checkpoint["classes"],
            device=device,
        )
        inst.model.load_state_dict(checkpoint["state_dict"])
        inst.model.to(inst.device)
        inst.is_fitted = checkpoint["is_fitted"]
        logger.info(f"Loaded FaultClassifierWrapper from {path} ({inst.num_classes} classes)")
        return inst
