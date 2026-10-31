"""Unit tests for ML models module.

Tests anomaly detection (Isolation Forest + Autoencoder), 1D-CNN fault classification,
and BiLSTM Remaining Useful Life (RUL) estimation.
"""

import numpy as np
import pytest
import torch

from src.ml_models.anomaly_detector import (
    AutoencoderDetector,
    EnsembleDetector,
    IsolationForestDetector,
)
from src.ml_models.evaluator import ModelEvaluator
from src.ml_models.fault_classifier import FaultClassifier1DCNN, FaultClassifierWrapper
from src.ml_models.rul_estimator import (
    RULBiLSTM,
    RULEstimatorWrapper,
    compute_cmapss_score,
)


def test_isolation_forest_detector():
    """Verify Isolation Forest anomaly detector trains and predicts scores."""
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 5))
    detector = IsolationForestDetector(contamination=0.05)
    detector.train(X)

    scores, labels = detector.predict(X)
    assert len(scores) == 100
    assert len(labels) == 100
    assert set(np.unique(labels)).issubset({0, 1})
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_autoencoder_detector():
    """Verify deep Autoencoder anomaly detector with reconstruction error threshold."""
    torch.manual_seed(42)
    X = np.random.normal(0, 1, size=(80, 8)).astype(np.float32)
    detector = AutoencoderDetector(latent_dim=4)
    detector.train(X, epochs=5, batch_size=16)

    errors, labels = detector.predict(X)
    assert len(errors) == 80
    assert len(labels) == 80
    assert detector.threshold > 0.0


def test_ensemble_detector():
    """Verify ensemble anomaly detector combines both scores."""
    X = np.random.normal(0, 1, size=(60, 6)).astype(np.float32)
    detector = EnsembleDetector(weight_if=0.5, weight_ae=0.5)
    detector.train(X, epochs=5)

    scores, labels = detector.predict(X)
    assert len(scores) == 60
    assert len(labels) == 60


def test_fault_classifier_1d_cnn():
    """Verify 1D-CNN network and wrapper classify multi-class fault states."""
    torch.manual_seed(42)
    num_samples = 40
    seq_len = 15
    in_channels = 6

    X = np.random.randn(num_samples, seq_len, in_channels).astype(np.float32)
    classes = ["Normal", "Bearing_Fault", "Overheating", "Misalignment", "Electrical_Fault"]
    y = np.random.choice(classes, size=num_samples)

    wrapper = FaultClassifierWrapper(in_channels=in_channels, classes=classes)
    wrapper.train(X, y, epochs=5, batch_size=8)

    preds = wrapper.predict(X)
    assert "predicted_class" in preds
    assert "confidence" in preds
    assert len(preds["predicted_class"]) == num_samples

    eval_results = wrapper.evaluate(X, y)
    assert "accuracy" in eval_results
    assert 0.0 <= eval_results["accuracy"] <= 1.0


def test_rul_bilstm_estimator():
    """Verify BiLSTM model estimates continuous RUL with uncertainty bounds."""
    torch.manual_seed(42)
    num_samples = 30
    seq_len = 12
    in_channels = 5

    X = np.random.randn(num_samples, seq_len, in_channels).astype(np.float32)
    y = np.random.uniform(5, 120, size=num_samples).astype(np.float32)

    estimator = RULEstimatorWrapper(input_size=in_channels, hidden_size=64, num_layers=1)
    estimator.train(X, y, epochs=5, batch_size=8)

    rul_preds, (lower, upper) = estimator.predict(X)
    assert len(rul_preds) == num_samples
    assert len(lower) == num_samples
    assert len(upper) == num_samples
    # Lower bound should generally be <= upper bound
    assert np.all(lower <= upper + 1e-3)

    metrics = estimator.evaluate(X, y)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "scoring_function" in metrics


def test_cmapss_scoring_function():
    """Verify asymmetric NASA scoring function penalizes late predictions more."""
    y_true = np.array([50.0, 50.0])
    y_early = np.array([40.0, 50.0])  # Early by 10 cycles
    y_late = np.array([60.0, 50.0])   # Late by 10 cycles

    score_early = compute_cmapss_score(y_true, y_early)
    score_late = compute_cmapss_score(y_true, y_late)

    assert score_late > score_early


def test_model_evaluator():
    """Verify ModelEvaluator produces reports and metrics."""
    evaluator = ModelEvaluator()
    y_true = np.array([0, 1, 0, 1, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.85])

    class DummyModel:
        def predict(self, X):
            return y_scores, (y_scores > 0.5).astype(int)

    metrics = evaluator.evaluate_anomaly(DummyModel(), np.zeros((5, 2)), y_true)
    assert "f1" in metrics
    assert "roc_auc" in metrics
