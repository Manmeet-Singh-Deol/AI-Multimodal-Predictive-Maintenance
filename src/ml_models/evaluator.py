"""Model Evaluation and Diagnostics Suite.

Computes comprehensive diagnostic metrics, generates publication-quality visualization
plots (Confusion Matrix, ROC Curve, RUL Predictions with Uncertainty Bounds),
and generates markdown evaluation reports for the predictive maintenance assistant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless rendering for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from config.settings import get_settings
from src.ml_models.rul_estimator import compute_cmapss_score


class ModelEvaluator:
    """Comprehensive performance evaluator and diagnostic report generator."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Initializes ModelEvaluator.

        Args:
            output_dir: Target directory for diagnostic figures and markdown reports.
                        Defaults to settings.processed_path / 'evaluation'.
        """
        settings = get_settings()
        self.output_dir = Path(output_dir) if output_dir else settings.processed_path / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_anomaly(
        self,
        model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series, List[Any]],
    ) -> Dict[str, float]:
        """Evaluates anomaly detection performance against binary ground truth.

        Args:
            model: Anomaly detector (IsolationForestDetector, AutoencoderDetector, or EnsembleDetector).
            X_test: Test telemetry feature matrix.
            y_test: Ground truth labels (1 for anomaly, 0 for normal, or state strings).

        Returns:
            Dictionary with precision, recall, f1, roc_auc, and pr_auc.
        """
        # Map string states to binary: Normal -> 0, anything else -> 1
        y_arr = np.asarray(y_test)
        if not np.issubdtype(y_arr.dtype, np.number):
            y_true = np.where(y_arr == "Normal", 0, 1)
        else:
            y_true = (y_arr > 0).astype(int)

        scores, preds = model.predict(X_test)
        scores = np.asarray(scores, dtype=float)
        preds = np.asarray(preds, dtype=int)

        # Handle NaNs if present
        scores = np.nan_to_num(scores, nan=0.0)
        preds = np.nan_to_num(preds, nan=0)

        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)

        # ROC-AUC & PR-AUC
        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, scores))
                pr_auc = float(average_precision_score(y_true, scores))
            else:
                roc_auc = 0.5
                pr_auc = 0.0
        except Exception as e:
            logger.warning(f"Error calculating AUC metrics: {e}")
            roc_auc = 0.5
            pr_auc = 0.0

        metrics = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
        logger.info(
            f"Anomaly Evaluation: Precision={p:.4f}, Recall={r:.4f}, "
            f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}"
        )
        return metrics

    def evaluate_classifier(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: Union[np.ndarray, pd.Series, List[Any]],
    ) -> Dict[str, Any]:
        """Evaluates multi-class fault classification performance.

        Args:
            model: FaultClassifierWrapper instance.
            X_test: Test 3D sequence array.
            y_test: True class labels.

        Returns:
            Dictionary with accuracy, precision, recall, f1, confusion_matrix,
            and classification report.
        """
        metrics = model.evaluate(X_test, y_test)
        return metrics

    def evaluate_rul(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: Union[np.ndarray, pd.Series, List[float]],
    ) -> Dict[str, float]:
        """Evaluates Remaining Useful Life regression predictions.

        Args:
            model: RULEstimatorWrapper instance.
            X_test: Test 3D sequence array.
            y_test: True Remaining Useful Life cycles.

        Returns:
            Dictionary with rmse, mae, r2, and scoring_function.
        """
        metrics = model.evaluate(X_test, y_test)
        return metrics

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Fault Classification Confusion Matrix",
    ) -> Path:
        """Plots and saves an annotated confusion matrix heatmap.

        Args:
            cm: 2D confusion matrix array.
            class_names: Ordered category names.
            save_path: File destination. Defaults to output_dir / 'confusion_matrix.png'.
            title: Plot title.

        Returns:
            Path of saved figure.
        """
        dest = Path(save_path) if save_path else self.output_dir / "confusion_matrix.png"
        dest.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True Fault Label",
            xlabel="Predicted Fault Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

        # Annotate text values in matrix cells
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                )

        fig.tight_layout()
        plt.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot to {dest}")
        return dest

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Anomaly Detection ROC Curve",
    ) -> Path:
        """Generates and saves Receiver Operating Characteristic (ROC) curve plot.

        Args:
            y_true: Binary ground truth labels (0 or 1).
            y_scores: Anomaly continuous severity scores.
            save_path: File destination.
            title: Plot title.

        Returns:
            Path of saved figure.
        """
        dest = Path(save_path) if save_path else self.output_dir / "roc_curve.png"
        dest.parent.mkdir(parents=True, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True)

        fig.tight_layout()
        plt.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ROC curve plot to {dest}")
        return dest

    def plot_rul_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        lower_ci: Optional[np.ndarray] = None,
        upper_ci: Optional[np.ndarray] = None,
        sample_limit: int = 100,
        title: str = "Remaining Useful Life (RUL): Predicted vs Actual",
    ) -> Path:
        """Generates dual-panel RUL diagnostic visualization: parity scatter and trajectory time-series.

        Args:
            y_true: True Remaining Useful Life cycles.
            y_pred: Predicted Remaining Useful Life cycles.
            save_path: File destination.
            lower_ci: Optional 95% lower confidence interval.
            upper_ci: Optional 95% upper confidence interval.
            sample_limit: Number of trajectory steps to display in sequence view.
            title: Overall figure title.

        Returns:
            Path of saved figure.
        """
        dest = Path(save_path) if save_path else self.output_dir / "rul_predictions.png"
        dest.parent.mkdir(parents=True, exist_ok=True)

        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        fig, (ax_scatter, ax_seq) = plt.subplots(1, 2, figsize=(15, 6))

        # Panel 1: Parity scatter plot
        max_val = max(float(np.max(yt)), float(np.max(yp)), 10.0)
        ax_scatter.scatter(yt, yp, alpha=0.5, color="#1f77b4", edgecolors="k", s=30, label="Predictions")
        ax_scatter.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal Line (y = x)")
        ax_scatter.set_title("RUL Prediction Parity", fontsize=12, fontweight="bold")
        ax_scatter.set_xlabel("True RUL (Cycles)", fontsize=11)
        ax_scatter.set_ylabel("Predicted RUL (Cycles)", fontsize=11)
        ax_scatter.grid(True, linestyle=":", alpha=0.6)
        ax_scatter.legend()

        # Panel 2: Sequence trajectory slice
        n_slice = min(len(yt), sample_limit)
        steps = np.arange(n_slice)
        ax_seq.plot(steps, yt[:n_slice], color="#111111", lw=2, label="Actual Ground Truth")
        ax_seq.plot(steps, yp[:n_slice], color="#e74c3c", lw=2, linestyle="-.", label="Bi-LSTM Prediction")

        if lower_ci is not None and upper_ci is not None:
            ax_seq.fill_between(
                steps,
                lower_ci[:n_slice],
                upper_ci[:n_slice],
                color="#e74c3c",
                alpha=0.25,
                label="95% Confidence Interval (MC Dropout)",
            )

        ax_seq.set_title(f"Degradation Trajectory Sample (First {n_slice} Steps)", fontsize=12, fontweight="bold")
        ax_seq.set_xlabel("Time Step / Test Sample", fontsize=11)
        ax_seq.set_ylabel("RUL (Cycles)", fontsize=11)
        ax_seq.grid(True, linestyle=":", alpha=0.6)
        ax_seq.legend()

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        plt.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved RUL diagnostic plot to {dest}")
        return dest

    def generate_report(
        self,
        all_results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Compiles evaluation metrics across all models into a structured markdown report.

        Args:
            all_results: Nested dictionary containing metrics for 'anomaly', 'classifier', and 'rul'.
            output_path: File destination. Defaults to output_dir / 'evaluation_report.md'.

        Returns:
            Formatted Markdown report string.
        """
        dest = Path(output_path) if output_path else self.output_dir / "evaluation_report.md"
        dest.parent.mkdir(parents=True, exist_ok=True)

        anomaly_m = all_results.get("anomaly", {})
        clf_m = all_results.get("classifier", {})
        rul_m = all_results.get("rul", {})

        report_lines = [
            "# Predictive Maintenance Model Evaluation Report",
            "",
            "## Executive Summary",
            "This report summarizes the empirical performance metrics for the multi-sensor predictive maintenance suite:",
            "- **Ensemble Anomaly Detector**: Unsupervised isolation forest and deep autoencoder 3-sigma statistical thresholding.",
            "- **Fault Classifier**: 1D Convolutional Neural Network across 5 industrial machine operational states.",
            "- **RUL Estimator**: Bidirectional LSTM network with Monte Carlo Dropout uncertainty estimation.",
            "",
            "---",
            "",
            "## 1. Anomaly Detection Performance",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Precision** | {anomaly_m.get('precision', 0.0):.4f} |",
            f"| **Recall** | {anomaly_m.get('recall', 0.0):.4f} |",
            f"| **F1 Score** | {anomaly_m.get('f1', 0.0):.4f} |",
            f"| **ROC-AUC** | {anomaly_m.get('roc_auc', 0.0):.4f} |",
            f"| **PR-AUC** | {anomaly_m.get('pr_auc', 0.0):.4f} |",
            "",
            "---",
            "",
            "## 2. Multi-Class Fault Classification Performance",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Overall Accuracy** | {clf_m.get('accuracy', 0.0):.2%} |",
            f"| **Precision (Macro)** | {clf_m.get('precision_macro', 0.0):.4f} |",
            f"| **Recall (Macro)** | {clf_m.get('recall_macro', 0.0):.4f} |",
            f"| **F1 Score (Macro)** | {clf_m.get('f1_macro', 0.0):.4f} |",
            f"| **F1 Score (Weighted)** | {clf_m.get('f1_weighted', 0.0):.4f} |",
            "",
        ]

        # Per-class breakdown table
        if "classification_report" in clf_m and isinstance(clf_m["classification_report"], dict):
            report_lines.extend([
                "### Per-Class Detailed Breakdown",
                "",
                "| State | Precision | Recall | F1-Score | Support |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for cls_name, vals in clf_m["classification_report"].items():
                if isinstance(vals, dict):
                    report_lines.append(
                        f"| **{cls_name}** | {vals.get('precision', 0.0):.3f} | "
                        f"{vals.get('recall', 0.0):.3f} | {vals.get('f1-score', 0.0):.3f} | "
                        f"{int(vals.get('support', 0))} |"
                    )
            report_lines.append("")

        report_lines.extend([
            "---",
            "",
            "## 3. Remaining Useful Life (RUL) Estimation Performance",
            "",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
            f"| **RMSE** | {rul_m.get('rmse', 0.0):.2f} cycles | Root Mean Squared Error |",
            f"| **MAE** | {rul_m.get('mae', 0.0):.2f} cycles | Mean Absolute Error |",
            f"| **R² Score** | {rul_m.get('r2', 0.0):.4f} | Coefficient of Determination |",
            f"| **NASA Asymmetric Score** | {rul_m.get('scoring_function', 0.0):.2f} | Exponential Late-Prediction Penalty |",
            "",
            "---",
            "",
            "## Diagnostic Visualizations",
            "- Confusion Matrix: `confusion_matrix.png`",
            "- ROC Curve: `roc_curve.png`",
            "- RUL Predicted vs Actual: `rul_predictions.png`",
            "",
        ])

        report_content = "\n".join(report_lines)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Generated comprehensive evaluation report at {dest}")
        return report_content
