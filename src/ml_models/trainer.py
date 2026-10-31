"""Unified Model Training Orchestrator.

Orchestrates end-to-end data loading, feature preprocessing, model training, validation,
and persistence across all three core predictive maintenance model suites:
1. Ensemble Anomaly Detector (Isolation Forest + Autoencoder)
2. 1D-CNN Fault Classifier (5 Machine States)
3. Bi-LSTM Remaining Useful Life (RUL) Estimator
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from config.settings import get_settings
from src.data_pipeline.feature_engineering import extract_all_features
from src.data_pipeline.preprocessor import DataPreprocessor, temporal_train_val_test_split
from src.data_pipeline.synthetic_generator import SyntheticIndustrialGenerator
from src.ml_models.anomaly_detector import EnsembleDetector
from src.ml_models.fault_classifier import FaultClassifierWrapper
from src.ml_models.rul_estimator import RULEstimatorWrapper


class UnifiedTrainer:
    """Unified orchestrator for training, validating, and saving predictive maintenance models."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        sequence_length: Optional[int] = None,
    ) -> None:
        """Initializes UnifiedTrainer.

        Args:
            output_dir: Directory where trained model checkpoints are stored.
                        Defaults to settings.models_path.
            sequence_length: Sliding window sequence length. Defaults to settings.rul_sequence_length.
        """
        settings = get_settings()
        self.output_dir = Path(output_dir) if output_dir else settings.models_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length or settings.rul_sequence_length

        self.preprocessor = DataPreprocessor(
            scaler_type="standard",
            sequence_length=self.sequence_length,
            stride=1,
        )

    def _load_or_generate_dataset(self, data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Loads dataset from CSV or synthesizes an industrial multi-sensor dataset."""
        settings = get_settings()
        target_path = Path(data_path) if data_path else settings.synthetic_path / "synthetic_sensor_data.csv"

        if target_path.exists():
            logger.info(f"Loading sensor dataset from {target_path}...")
            df = pd.read_csv(target_path)
        else:
            logger.info(f"Dataset not found at {target_path}. Generating synthetic industrial data...")
            generator = SyntheticIndustrialGenerator(
                num_machines=settings.num_machines,
                num_cycles=500,
                fault_ratio=0.4,
                random_seed=42,
            )
            df = generator.generate_dataset()
            generator.save_dataset(df, target_path)

        return df

    def _extract_features_and_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Applies feature extraction and splits chronologically into train/val/test partitions."""
        group_col = "machine_id" if "machine_id" in df.columns else ("unit_id" if "unit_id" in df.columns else None)
        time_col = "timestamp" if "timestamp" in df.columns else ("cycle" if "cycle" in df.columns else None)

        # Determine numerical base sensor columns
        exclude_cols = {"timestamp", "machine_id", "unit_id", "cycle", "fault_label", "health_score", "rul"}
        sensor_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

        logger.info(f"Extracting multi-domain features for sensors: {sensor_cols}...")
        df_feat = extract_all_features(
            df=df,
            sensor_cols=sensor_cols,
            group_col=group_col,
            rolling_windows=(5, 10),
            include_spectral=True,
            fft_window_size=16,
        )

        train_df, val_df, test_df = temporal_train_val_test_split(
            df_feat,
            group_col=group_col,
            time_col=time_col,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        feature_cols = [c for c in df_feat.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_feat[c])]
        logger.info(f"Partitioned dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Features={len(feature_cols)}")

        # Fit and save preprocessor
        self.preprocessor.fit(train_df, feature_cols)
        preproc_path = self.output_dir / "preprocessor.joblib"
        self.preprocessor.save(preproc_path)

        return train_df, val_df, test_df, feature_cols

    def train_anomaly_detector(
        self,
        data_path: Optional[Union[str, Path]] = None,
        train_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        epochs: Optional[int] = None,
    ) -> EnsembleDetector:
        """Trains and saves the Ensemble Anomaly Detector on normal operating telemetry.

        Args:
            data_path: Path to dataset CSV.
            train_df: Optional preprocessed training DataFrame.
            feature_cols: List of features to train on.
            epochs: Training epochs override.

        Returns:
            Trained EnsembleDetector instance.
        """
        logger.info("========== Training Ensemble Anomaly Detector ==========")
        if train_df is None or feature_cols is None:
            raw_df = self._load_or_generate_dataset(data_path)
            train_df, _, _, feature_cols = self._extract_features_and_split(raw_df)

        # Filter for normal baseline conditions
        if "fault_label" in train_df.columns:
            normal_df = train_df[train_df["fault_label"] == "Normal"]
        else:
            normal_df = train_df

        if len(normal_df) < 20:
            logger.warning("Normal slice is very small, using full train slice for anomaly detector.")
            normal_df = train_df

        # Scale features
        X_normal_scaled = self.preprocessor.scaler.transform(normal_df[feature_cols])

        settings = get_settings()
        detector = EnsembleDetector(
            weight_if=0.5,
            weight_ae=0.5,
            contamination=settings.anomaly_contamination,
            latent_dim=settings.autoencoder_latent_dim,
        )
        ae_epochs = epochs or settings.autoencoder_epochs
        detector.train(
            X_normal_scaled,
            epochs=ae_epochs,
            batch_size=settings.autoencoder_batch_size,
        )

        save_path = self.output_dir / "anomaly_detector"
        detector.save(save_path)
        logger.info(f"Ensemble Anomaly Detector trained and saved to {save_path}")
        return detector

    def train_fault_classifier(
        self,
        data_path: Optional[Union[str, Path]] = None,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        epochs: Optional[int] = None,
    ) -> FaultClassifierWrapper:
        """Trains and saves the 1D-CNN Fault Classifier.

        Args:
            data_path: Path to dataset CSV.
            train_df: Optional preprocessed training partition.
            val_df: Optional preprocessed validation partition.
            feature_cols: Feature columns list.
            epochs: Training epochs override.

        Returns:
            Trained FaultClassifierWrapper instance.
        """
        logger.info("========== Training 1D-CNN Fault Classifier ==========")
        if train_df is None or val_df is None or feature_cols is None:
            raw_df = self._load_or_generate_dataset(data_path)
            train_df, val_df, _, feature_cols = self._extract_features_and_split(raw_df)

        group_col = "machine_id" if "machine_id" in train_df.columns else ("unit_id" if "unit_id" in train_df.columns else None)
        target_col = "fault_label" if "fault_label" in train_df.columns else None

        if target_col is None:
            raise ValueError("Training Fault Classifier requires a 'fault_label' column in the dataset.")

        X_train, y_train = self.preprocessor.prepare_sequences(
            df=train_df,
            target_col=target_col,
            group_col=group_col,
            augment_noise=0.01,
        )
        X_val, y_val = self.preprocessor.prepare_sequences(
            df=val_df,
            target_col=target_col,
            group_col=group_col,
        )

        settings = get_settings()
        classifier = FaultClassifierWrapper(in_channels=len(feature_cols))
        clf_epochs = epochs or settings.fault_classifier_epochs
        classifier.train(
            X_windows=X_train,
            y_labels=y_train,
            epochs=clf_epochs,
            batch_size=settings.fault_classifier_batch_size,
            lr=settings.fault_classifier_lr,
            val_data=(X_val, y_val),
            patience=10,
        )

        save_path = self.output_dir / "fault_classifier.pt"
        classifier.save(save_path)
        logger.info(f"Fault Classifier trained and saved to {save_path}")
        return classifier

    def train_rul_estimator(
        self,
        data_path: Optional[Union[str, Path]] = None,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        epochs: Optional[int] = None,
    ) -> RULEstimatorWrapper:
        """Trains and saves the Bi-LSTM Remaining Useful Life (RUL) Estimator.

        Args:
            data_path: Path to dataset CSV.
            train_df: Optional preprocessed training partition.
            val_df: Optional preprocessed validation partition.
            feature_cols: Feature columns list.
            epochs: Training epochs override.

        Returns:
            Trained RULEstimatorWrapper instance.
        """
        logger.info("========== Training Bi-LSTM RUL Estimator ==========")
        if train_df is None or val_df is None or feature_cols is None:
            raw_df = self._load_or_generate_dataset(data_path)
            train_df, val_df, _, feature_cols = self._extract_features_and_split(raw_df)

        group_col = "machine_id" if "machine_id" in train_df.columns else ("unit_id" if "unit_id" in train_df.columns else None)
        target_col = "rul" if "rul" in train_df.columns else None

        if target_col is None:
            raise ValueError("Training RUL Estimator requires a 'rul' column in the dataset.")

        X_train, y_train = self.preprocessor.prepare_sequences(
            df=train_df,
            target_col=target_col,
            group_col=group_col,
            augment_noise=0.01,
        )
        X_val, y_val = self.preprocessor.prepare_sequences(
            df=val_df,
            target_col=target_col,
            group_col=group_col,
        )

        settings = get_settings()
        estimator = RULEstimatorWrapper(
            input_size=len(feature_cols),
            hidden_size=settings.rul_hidden_size,
            num_layers=settings.rul_num_layers,
            max_rul=settings.rul_max_rul,
        )
        r_epochs = epochs or settings.rul_epochs
        estimator.train(
            X_sequences=X_train,
            y_rul=y_train,
            epochs=r_epochs,
            batch_size=settings.rul_batch_size,
            lr=settings.rul_lr,
            val_data=(X_val, y_val),
            patience=10,
            loss_fn="huber",
        )

        save_path = self.output_dir / "rul_estimator.pt"
        estimator.save(save_path)
        logger.info(f"RUL Estimator trained and saved to {save_path}")
        return estimator

    def train_all(
        self,
        data_path: Optional[Union[str, Path]] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Trains all three model families sequentially using a shared preprocessed partition.

        Args:
            data_path: Path to dataset CSV.
            epochs: Optional epoch override across models.

        Returns:
            Dictionary containing trained model instances and test data partition for evaluation.
        """
        logger.info("========== Starting Unified Predictive Maintenance Training Pipeline ==========")
        raw_df = self._load_or_generate_dataset(data_path)
        train_df, val_df, test_df, feature_cols = self._extract_features_and_split(raw_df)

        anomaly_model = self.train_anomaly_detector(train_df=train_df, feature_cols=feature_cols, epochs=epochs)
        classifier_model = self.train_fault_classifier(train_df=train_df, val_df=val_df, feature_cols=feature_cols, epochs=epochs)
        rul_model = self.train_rul_estimator(train_df=train_df, val_df=val_df, feature_cols=feature_cols, epochs=epochs)

        logger.info("All predictive maintenance models trained and successfully persisted!")
        return {
            "anomaly_model": anomaly_model,
            "classifier_model": classifier_model,
            "rul_model": rul_model,
            "preprocessor": self.preprocessor,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "feature_cols": feature_cols,
        }


def main() -> None:
    """CLI interface for training predictive maintenance models."""
    parser = argparse.ArgumentParser(description="Predictive Maintenance Unified Model Trainer")
    parser.add_argument("--train-all", action="store_true", help="Train all models sequentially")
    parser.add_argument("--anomaly", action="store_true", help="Train only the anomaly detector")
    parser.add_argument("--classifier", action="store_true", help="Train only the fault classifier")
    parser.add_argument("--rul", action="store_true", help="Train only the RUL estimator")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--data-path", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--output-dir", type=str, default=None, help="Model checkpoint directory")
    args = parser.parse_args()

    trainer = UnifiedTrainer(output_dir=args.output_dir)

    if args.train_all or (not args.anomaly and not args.classifier and not args.rul):
        trainer.train_all(data_path=args.data_path, epochs=args.epochs)
    else:
        if args.anomaly:
            trainer.train_anomaly_detector(data_path=args.data_path, epochs=args.epochs)
        if args.classifier:
            trainer.train_fault_classifier(data_path=args.data_path, epochs=args.epochs)
        if args.rul:
            trainer.train_rul_estimator(data_path=args.data_path, epochs=args.epochs)


if __name__ == "__main__":
    main()
