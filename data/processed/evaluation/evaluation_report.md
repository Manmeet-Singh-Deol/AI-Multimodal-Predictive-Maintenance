# Predictive Maintenance Model Evaluation Report

## Executive Summary
This report summarizes the empirical performance metrics for the multi-sensor predictive maintenance suite:
- **Ensemble Anomaly Detector**: Unsupervised isolation forest and deep autoencoder 3-sigma statistical thresholding.
- **Fault Classifier**: 1D Convolutional Neural Network across 5 industrial machine operational states.
- **RUL Estimator**: Bidirectional LSTM network with Monte Carlo Dropout uncertainty estimation.

---

## 1. Anomaly Detection Performance

| Metric | Value |
| :--- | :--- |
| **Precision** | 0.0000 |
| **Recall** | 0.0000 |
| **F1 Score** | 0.0000 |
| **ROC-AUC** | 0.5000 |
| **PR-AUC** | 0.0000 |

---

## 2. Multi-Class Fault Classification Performance

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | 100.00% |
| **Precision (Macro)** | 1.0000 |
| **Recall (Macro)** | 1.0000 |
| **F1 Score (Macro)** | 1.0000 |
| **F1 Score (Weighted)** | 1.0000 |

### Per-Class Detailed Breakdown

| State | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 1.000 | 1.000 | 1.000 | 346 |
| **Bearing_Fault** | 0.000 | 0.000 | 0.000 | 0 |
| **Overheating** | 0.000 | 0.000 | 0.000 | 0 |
| **Misalignment** | 0.000 | 0.000 | 0.000 | 0 |
| **Electrical_Fault** | 0.000 | 0.000 | 0.000 | 0 |
| **macro avg** | 0.200 | 0.200 | 0.200 | 346 |
| **weighted avg** | 1.000 | 1.000 | 1.000 | 346 |

---

## 3. Remaining Useful Life (RUL) Estimation Performance

| Metric | Value | Description |
| :--- | :--- | :--- |
| **RMSE** | 3.22 cycles | Root Mean Squared Error |
| **MAE** | 2.09 cycles | Mean Absolute Error |
| **R² Score** | 0.0000 | Coefficient of Determination |
| **NASA Asymmetric Score** | 67.97 | Exponential Late-Prediction Penalty |

---

## Diagnostic Visualizations
- Confusion Matrix: `confusion_matrix.png`
- ROC Curve: `roc_curve.png`
- RUL Predicted vs Actual: `rul_predictions.png`
