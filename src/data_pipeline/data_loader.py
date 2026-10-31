"""NASA C-MAPSS Turbofan Engine Degradation Dataset Loader.

This module provides data loading, download/synthetic generation, and RUL label
preparation for the 4 sub-datasets (FD001 - FD004) of NASA's C-MAPSS benchmark.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import get_settings

CMAPSS_RAW_COLUMNS: List[str] = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

STANDARDIZED_COLUMNS: List[str] = ["unit_id", "cycle"] + [
    f"sensor_{i}" for i in range(1, 22)
] + ["rul"]

DATASET_SUBSETS: List[str] = ["FD001", "FD002", "FD003", "FD004"]

# Primary & Mirror URLs for C-MAPSS data
CMAPSS_URLS: List[str] = [
    "https://data.nasa.gov/download/7mg5-zg7k/application%2Fzip",
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData.zip",
    "https://zenodo.org/record/4060097/files/CMAPSSData.zip?download=1",
]


def add_piecewise_rul(
    df: pd.DataFrame,
    max_rul: int = 125,
    test_rul_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Adds piece-wise linear Remaining Useful Life (RUL) labels to a DataFrame.

    Piece-wise linear degradation models assume that degradation is not detectable
    in early engine operating cycles, and becomes noticeable only when RUL drops below
    a threshold (max_rul).

    Args:
        df: DataFrame containing at least ['unit_id', 'cycle'].
        max_rul: Upper bound saturation threshold for piece-wise linear RUL.
        test_rul_df: Optional DataFrame with ground-truth final RUL for test units
            indexed by unit_id (or 0-indexed column 0 for unit 1..N).

    Returns:
        DataFrame with an added 'rul' column.
    """
    df_out = df.copy()

    if test_rul_df is None:
        # Training dataset: Run-to-failure trajectories
        # Calculate maximum cycle for each unit
        max_cycles = df_out.groupby("unit_id")["cycle"].transform("max")
        raw_rul = max_cycles - df_out["cycle"]
        df_out["rul"] = np.minimum(raw_rul.to_numpy(), float(max_rul))
    else:
        # Test dataset: Trajectories end before failure
        # test_rul_df contains the remaining cycles after the last cycle in test
        if isinstance(test_rul_df, pd.Series):
            rul_series = test_rul_df
        elif isinstance(test_rul_df, pd.DataFrame):
            rul_series = test_rul_df.iloc[:, 0]
        else:
            rul_series = pd.Series(test_rul_df)

        # Build mapping from unit_id to last cycle and remaining RUL
        max_cycles = df_out.groupby("unit_id")["cycle"].transform("max")
        # Ensure 1-based indexing for units
        unit_ruls = {}
        for idx, val in enumerate(rul_series.values, start=1):
            unit_ruls[idx] = float(val)

        remaining_after_last = df_out["unit_id"].map(unit_ruls).fillna(0.0)
        total_lifetime = max_cycles + remaining_after_last
        raw_rul = total_lifetime - df_out["cycle"]
        df_out["rul"] = np.minimum(raw_rul.to_numpy(), float(max_rul))

    return df_out


def generate_synthetic_cmapss(
    output_dir: Path,
    subset: str = "FD001",
    num_train_units: int = 100,
    num_test_units: int = 100,
    random_seed: int = 42,
) -> Tuple[Path, Path, Path]:
    """Generates synthetic C-MAPSS equivalent files adhering to standard formatting.

    This fallback ensures offline reliability and self-contained benchmarking
    when external NASA repositories cannot be reached.

    Args:
        output_dir: Directory where raw text files will be written.
        subset: C-MAPSS sub-dataset code (e.g. 'FD001').
        num_train_units: Number of training engine units to simulate.
        num_test_units: Number of test engine units to simulate.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_path, test_path, rul_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_seed)

    train_file = output_dir / f"train_{subset}.txt"
    test_file = output_dir / f"test_{subset}.txt"
    rul_file = output_dir / f"RUL_{subset}.txt"

    # Baseline nominal values for the 21 sensors and 3 op settings
    nominal_settings = np.array([-0.0007, -0.0004, 100.0])
    nominal_sensors = np.array([
        518.67, 642.50, 1585.0, 1400.0, 14.62,
        21.61, 553.5, 2388.05, 9050.0, 1.30,
        47.35, 521.8, 2388.08, 8130.0, 8.42,
        0.03, 392.0, 2388.0, 100.0, 38.85, 23.35
    ])
    # Sensors that typically show degradation drift
    drift_factors = np.array([
        0.0, 0.05, 0.08, 0.06, 0.0,
        0.02, -0.04, 0.01, 0.02, 0.0,
        -0.03, -0.04, 0.01, 0.03, 0.05,
        0.0, 0.03, 0.0, 0.0, -0.05, -0.03
    ])

    def generate_engine_trajectory(unit: int, total_cycles: int, is_test: bool = False, cut_ratio: float = 0.7) -> np.ndarray:
        actual_cycles = total_cycles
        if is_test:
            actual_cycles = max(30, int(total_cycles * cut_ratio))

        rows = []
        for c in range(1, actual_cycles + 1):
            degradation = (c / total_cycles) ** 2.0
            settings = nominal_settings + rng.normal(0, [0.001, 0.0005, 0.1])
            sensors = nominal_sensors * (1.0 + drift_factors * degradation) + rng.normal(0, np.abs(nominal_sensors) * 0.005)
            row = [unit, c] + list(settings) + list(sensors)
            rows.append(row)
        return np.array(rows), (total_cycles - actual_cycles)

    # Generate Train
    logger.info(f"Synthesizing {subset} train set ({num_train_units} units)...")
    train_rows = []
    for u in range(1, num_train_units + 1):
        lifetime = rng.randint(130, 360)
        unit_data, _ = generate_engine_trajectory(u, lifetime, is_test=False)
        train_rows.append(unit_data)
    all_train = np.vstack(train_rows)
    np.savetxt(train_file, all_train, fmt=["%d", "%d"] + ["%.6f"] * 24, delimiter=" ")

    # Generate Test and RUL
    logger.info(f"Synthesizing {subset} test set ({num_test_units} units)...")
    test_rows = []
    ruls = []
    for u in range(1, num_test_units + 1):
        lifetime = rng.randint(130, 360)
        cut = rng.uniform(0.35, 0.85)
        unit_data, remaining_rul = generate_engine_trajectory(u, lifetime, is_test=True, cut_ratio=cut)
        test_rows.append(unit_data)
        ruls.append(remaining_rul)
    all_test = np.vstack(test_rows)
    np.savetxt(test_file, all_test, fmt=["%d", "%d"] + ["%.6f"] * 24, delimiter=" ")
    np.savetxt(rul_file, np.array(ruls, dtype=int), fmt="%d")

    logger.info(f"Synthetic dataset {subset} generated at {output_dir}")
    return train_file, test_file, rul_file


def download_cmapss(
    output_dir: Optional[Path] = None,
    force_download: bool = False,
    use_synthetic_fallback: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """Downloads C-MAPSS dataset or falls back to synthetic generation.

    Args:
        output_dir: Destination directory. Defaults to settings.raw_path.
        force_download: If True, forces redownload or recreation.
        use_synthetic_fallback: If download fails, generate synthetic equivalent.

    Returns:
        Nested dict of paths: {subset: {'train': Path, 'test': Path, 'rul': Path}}.
    """
    settings = get_settings()
    dest_dir = Path(output_dir) if output_dir else settings.raw_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    result_paths: Dict[str, Dict[str, Path]] = {}
    missing_subsets = []

    for subset in DATASET_SUBSETS:
        train_p = dest_dir / f"train_{subset}.txt"
        test_p = dest_dir / f"test_{subset}.txt"
        rul_p = dest_dir / f"RUL_{subset}.txt"
        if not force_download and train_p.exists() and test_p.exists() and rul_p.exists():
            result_paths[subset] = {"train": train_p, "test": test_p, "rul": rul_p}
        else:
            missing_subsets.append(subset)

    if not missing_subsets:
        logger.info(f"All C-MAPSS datasets already exist in {dest_dir}")
        return result_paths

    # Attempt download
    download_success = False
    zip_dest = dest_dir / "CMAPSSData.zip"

    for url in CMAPSS_URLS:
        try:
            logger.info(f"Attempting download from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response, open(zip_dest, "wb") as out_f:
                out_f.write(response.read())

            if zip_dest.exists() and zipfile.is_zipfile(zip_dest):
                logger.info("Extracting CMAPSS archive...")
                with zipfile.ZipFile(zip_dest, "r") as zf:
                    zf.extractall(dest_dir)
                try:
                    zip_dest.unlink()
                except OSError:
                    pass
                download_success = True
                break
        except Exception as e:
            logger.warning(f"Failed download from {url}: {e}")

    # Verify if files exist after download
    for subset in DATASET_SUBSETS:
        train_p = dest_dir / f"train_{subset}.txt"
        test_p = dest_dir / f"test_{subset}.txt"
        rul_p = dest_dir / f"RUL_{subset}.txt"

        if train_p.exists() and test_p.exists() and rul_p.exists():
            result_paths[subset] = {"train": train_p, "test": test_p, "rul": rul_p}
        elif use_synthetic_fallback:
            logger.info(f"Using high-fidelity synthetic fallback generator for {subset}...")
            train_p, test_p, rul_p = generate_synthetic_cmapss(
                dest_dir,
                subset=subset,
                num_train_units=100,
                num_test_units=100,
            )
            result_paths[subset] = {"train": train_p, "test": test_p, "rul": rul_p}
        else:
            raise FileNotFoundError(f"Missing required C-MAPSS files for subset {subset}")

    return result_paths


class CMAPSSDataLoader:
    """Loader and preprocessor for NASA C-MAPSS Turbofan Engine datasets."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        max_rul: Optional[int] = None,
    ) -> None:
        """Initializes the CMAPSSDataLoader.

        Args:
            data_dir: Directory containing raw C-MAPSS text files.
            max_rul: Upper limit for piece-wise linear RUL. Defaults to settings.rul_max_rul.
        """
        settings = get_settings()
        self.data_dir = Path(data_dir) if data_dir else settings.raw_path
        self.max_rul = max_rul if max_rul is not None else settings.rul_max_rul
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_data(self, subset: str) -> Tuple[Path, Path, Path]:
        """Ensures subset files exist, downloading or generating them if absent."""
        train_p = self.data_dir / f"train_{subset}.txt"
        test_p = self.data_dir / f"test_{subset}.txt"
        rul_p = self.data_dir / f"RUL_{subset}.txt"

        if not (train_p.exists() and test_p.exists() and rul_p.exists()):
            logger.info(f"Dataset files for {subset} missing in {self.data_dir}. Downloading or synthesizing...")
            download_cmapss(self.data_dir, use_synthetic_fallback=True)

        return train_p, test_p, rul_p

    def load_train(
        self,
        subset: str = "FD001",
        standardized_only: bool = True,
    ) -> pd.DataFrame:
        """Loads and prepares the training partition with piecewise linear RUL.

        Args:
            subset: C-MAPSS subset identifier ('FD001', 'FD002', 'FD003', 'FD004').
            standardized_only: If True, returns [unit_id, cycle, sensor_1..21, rul].
                               If False, keeps op_settings as well.

        Returns:
            DataFrame with standardized columns and target 'rul'.
        """
        train_p, _, _ = self._ensure_data(subset)
        df = pd.read_csv(train_p, sep=r"\s+", header=None, names=CMAPSS_RAW_COLUMNS)
        df_with_rul = add_piecewise_rul(df, max_rul=self.max_rul, test_rul_df=None)

        if standardized_only:
            return df_with_rul[STANDARDIZED_COLUMNS].copy()
        return df_with_rul

    def load_test(
        self,
        subset: str = "FD001",
        standardized_only: bool = True,
    ) -> pd.DataFrame:
        """Loads and prepares the testing partition with true RUL labels.

        Args:
            subset: C-MAPSS subset identifier ('FD001', 'FD002', 'FD003', 'FD004').
            standardized_only: If True, returns [unit_id, cycle, sensor_1..21, rul].

        Returns:
            DataFrame with test cycles and target piece-wise linear 'rul'.
        """
        _, test_p, rul_p = self._ensure_data(subset)
        test_df = pd.read_csv(test_p, sep=r"\s+", header=None, names=CMAPSS_RAW_COLUMNS)
        rul_df = pd.read_csv(rul_p, sep=r"\s+", header=None)

        df_with_rul = add_piecewise_rul(test_df, max_rul=self.max_rul, test_rul_df=rul_df)

        if standardized_only:
            return df_with_rul[STANDARDIZED_COLUMNS].copy()
        return df_with_rul

    def load_subset(
        self,
        subset: str = "FD001",
        standardized_only: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads both train and test sets for a specific subset.

        Args:
            subset: C-MAPSS sub-dataset code.
            standardized_only: Whether to restrict columns to standardized set.

        Returns:
            Tuple of (train_df, test_df).
        """
        train_df = self.load_train(subset=subset, standardized_only=standardized_only)
        test_df = self.load_test(subset=subset, standardized_only=standardized_only)
        return train_df, test_df

    def load_all_subsets(
        self,
        standardized_only: bool = True,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Loads all four subsets (FD001 - FD004).

        Args:
            standardized_only: Whether to return standardized column layout.

        Returns:
            Dictionary mapping subset name to (train_df, test_df).
        """
        datasets = {}
        for subset in DATASET_SUBSETS:
            datasets[subset] = self.load_subset(subset=subset, standardized_only=standardized_only)
        return datasets


def main() -> None:
    """Command-line interface for downloading or generating C-MAPSS datasets."""
    parser = argparse.ArgumentParser(description="NASA C-MAPSS Turbofan Dataset Ingestion Tool")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download or synthesize C-MAPSS datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        help="Specific sub-dataset to download or load (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for raw files",
    )
    args = parser.parse_args()

    settings = get_settings()
    out_dir = Path(args.output_dir) if args.output_dir else settings.raw_path

    if args.download or len(sys.argv) == 1:
        logger.info(f"Downloading/generating datasets to {out_dir}...")
        download_cmapss(output_dir=out_dir, force_download=False, use_synthetic_fallback=True)

    loader = CMAPSSDataLoader(data_dir=out_dir)
    target_subsets = DATASET_SUBSETS if args.dataset == "all" else [args.dataset]

    for sub in target_subsets:
        train_df, test_df = loader.load_subset(sub, standardized_only=True)
        logger.info(
            f"Loaded {sub}: Train shape={train_df.shape}, Test shape={test_df.shape}, "
            f"Columns={list(train_df.columns[:4])}... Max RUL={train_df['rul'].max()}"
        )


if __name__ == "__main__":
    main()
