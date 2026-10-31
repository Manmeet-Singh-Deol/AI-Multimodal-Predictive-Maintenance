"""Realistic Multi-Sensor Industrial Machine Synthetic Data Generator.

Simulates multi-sensor time-series measurements across five operational states:
Normal, Bearing_Fault, Overheating, Misalignment, and Electrical_Fault.
Degradation trajectories are synthesized using Weibull-distributed failure curves
to provide realistic ground-truth Remaining Useful Life (RUL) and Health Scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import get_settings

# Machine states
MACHINE_STATES: List[str] = [
    "Normal",
    "Bearing_Fault",
    "Overheating",
    "Misalignment",
    "Electrical_Fault",
]

# Baseline nominal operating parameters (Mean and standard deviation)
NOMINAL_BASELINES: Dict[str, Tuple[float, float]] = {
    "temp": (65.0, 1.5),        # °C
    "vibration": (1.8, 0.2),    # mm/s RMS
    "pressure": (4.2, 0.15),    # bar
    "rpm": (1780.0, 15.0),      # RPM
    "current": (12.5, 0.5),     # Amperes
    "noise_db": (68.0, 1.5),    # Decibels (dB)
}

# Fault signature multipliers / target offsets
FAULT_SIGNATURES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Bearing_Fault": {
        "temp": (76.0, 3.0),
        "vibration": (9.5, 2.5),       # Substantial vibration spike
        "pressure": (4.1, 0.2),
        "rpm": (1760.0, 25.0),
        "current": (13.8, 0.8),
        "noise_db": (88.0, 4.0),       # High noise
    },
    "Overheating": {
        "temp": (115.0, 8.0),          # Severe thermal runaway
        "vibration": (3.2, 0.5),
        "pressure": (2.8, 0.3),        # Hydraulic pressure loss
        "rpm": (1750.0, 20.0),
        "current": (15.5, 1.2),
        "noise_db": (74.0, 2.5),
    },
    "Misalignment": {
        "temp": (74.0, 2.5),
        "vibration": (7.5, 1.8),       # Strong 1X / 2X harmonic vibration
        "pressure": (4.2, 0.15),
        "rpm": (1765.0, 18.0),
        "current": (14.2, 0.7),
        "noise_db": (83.0, 3.0),
    },
    "Electrical_Fault": {
        "temp": (80.0, 4.0),
        "vibration": (4.5, 1.0),
        "pressure": (4.2, 0.15),
        "rpm": (1680.0, 80.0),         # Torque ripple / speed hunting
        "current": (24.0, 4.5),        # Erratic current spikes
        "noise_db": (82.0, 3.5),       # Electrical hum
    },
}


class SyntheticIndustrialGenerator:
    """Industrial multi-sensor simulator with Weibull degradation dynamics."""

    def __init__(
        self,
        num_machines: int = 5,
        num_cycles: int = 500,
        noise_level: float = 0.05,
        fault_ratio: float = 0.4,
        random_seed: Optional[int] = 42,
    ) -> None:
        """Initializes the synthetic generator.

        Args:
            num_machines: Number of distinct machines to simulate.
            num_cycles: Number of temporal cycles/measurements per machine.
            noise_level: Multiplier for background Gaussian sensor noise.
            fault_ratio: Proportion of machines that encounter a degradation fault.
            random_seed: Seed for reproducible data generation.
        """
        self.num_machines = max(1, num_machines)
        self.num_cycles = max(50, num_cycles)
        self.noise_level = max(0.0, noise_level)
        self.fault_ratio = min(max(0.0, fault_ratio), 1.0)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def _weibull_degradation(
        self,
        current_step: int,
        fault_start_step: int,
        failure_step: int,
        shape: float = 3.0,
    ) -> float:
        """Calculates Weibull cumulative degradation index in [0.0, 1.0].

        Args:
            current_step: Current operating cycle.
            fault_start_step: Cycle at which anomalous degradation begins.
            failure_step: Cycle where total failure occurs.
            shape: Weibull shape parameter (k > 1 signifies wear-out phase).

        Returns:
            Degradation severity factor from 0.0 (healthy) to 1.0 (failure).
        """
        if current_step <= fault_start_step:
            # Baseline microscopic aging
            baseline_wear = 0.05 * (current_step / fault_start_step)
            return float(np.clip(baseline_wear, 0.0, 0.08))

        if current_step >= failure_step:
            return 1.0

        elapsed = float(current_step - fault_start_step)
        span = float(failure_step - fault_start_step)
        # Normalized Weibull CDF: 1 - exp(-(t/scale)^k)
        scale = span * 0.75
        cdf = 1.0 - np.exp(-((elapsed / scale) ** shape))
        return float(np.clip(cdf, 0.0, 1.0))

    def generate_machine_trajectory(
        self,
        machine_id: str,
        start_time: datetime,
        cycle_interval_seconds: int = 60,
    ) -> pd.DataFrame:
        """Generates continuous multi-sensor trajectory for an individual machine.

        Args:
            machine_id: Identifier for the machine (e.g. 'MACH_001').
            start_time: Timestamp for cycle 1.
            cycle_interval_seconds: Temporal step between successive cycles.

        Returns:
            DataFrame containing the machine's time-series measurements.
        """
        # Determine if this machine experiences an eventual fault
        experiences_fault = self.rng.uniform(0.0, 1.0) < self.fault_ratio
        if experiences_fault:
            fault_type = self.rng.choice(
                ["Bearing_Fault", "Overheating", "Misalignment", "Electrical_Fault"]
            )
            fault_start = self.rng.randint(int(self.num_cycles * 0.45), int(self.num_cycles * 0.75))
            failure_step = self.rng.randint(int(self.num_cycles * 0.90), self.num_cycles + 1)
        else:
            fault_type = "Normal"
            fault_start = self.num_cycles + 100
            failure_step = self.num_cycles + 200

        weibull_shape = self.rng.uniform(2.5, 3.8)

        records = []
        for step in range(1, self.num_cycles + 1):
            ts = start_time + timedelta(seconds=(step - 1) * cycle_interval_seconds)

            # Compute degradation index and health score
            deg_factor = self._weibull_degradation(
                current_step=step,
                fault_start_step=fault_start,
                failure_step=failure_step,
                shape=weibull_shape,
            )
            health_score = float(np.clip(1.0 - deg_factor, 0.0, 1.0))

            # Ground truth Remaining Useful Life (RUL)
            if step >= failure_step:
                rul = 0
            else:
                rul = int(failure_step - step)

            # Labeling logic
            if step < fault_start:
                current_label = "Normal"
            else:
                # Progressive transition: above 20% degradation becomes the fault class
                current_label = fault_type if deg_factor >= 0.15 else "Normal"

            # Sensor synthesis blending nominal and fault baselines
            row_sensors = {}
            for sensor, (nom_mean, nom_std) in NOMINAL_BASELINES.items():
                if fault_type != "Normal" and step >= fault_start:
                    target_mean, target_std = FAULT_SIGNATURES[fault_type][sensor]
                    # Smooth sigmoidal or linear blend based on degradation
                    eff_mean = (1.0 - deg_factor) * nom_mean + deg_factor * target_mean
                    eff_std = (1.0 - deg_factor) * nom_std + deg_factor * target_std
                else:
                    eff_mean = nom_mean
                    eff_std = nom_std

                # Gaussian noise & dynamic disturbance
                noise = self.rng.normal(0.0, eff_std * (1.0 + self.noise_level))
                val = eff_mean + noise

                # Add specific physical harmonic dynamics
                if sensor == "vibration" and current_label == "Bearing_Fault":
                    # Periodic shock wave / impact spikes
                    if self.rng.uniform(0.0, 1.0) < 0.25:
                        val += self.rng.uniform(3.0, 8.0) * deg_factor
                elif sensor == "current" and current_label == "Electrical_Fault":
                    # Current surge transient
                    if self.rng.uniform(0.0, 1.0) < 0.20:
                        val += self.rng.uniform(5.0, 15.0) * deg_factor
                elif sensor == "vibration" and current_label == "Misalignment":
                    # 1X & 2X rotational harmonic modulation
                    harmonic = 1.2 * np.sin(2.0 * np.pi * step / 10.0) + 0.8 * np.sin(4.0 * np.pi * step / 10.0)
                    val += harmonic * deg_factor

                row_sensors[sensor] = round(float(val), 3)

            record = {
                "timestamp": ts.isoformat(),
                "machine_id": machine_id,
                "temp": row_sensors["temp"],
                "vibration": max(0.01, row_sensors["vibration"]),
                "pressure": max(0.01, row_sensors["pressure"]),
                "rpm": max(0.0, row_sensors["rpm"]),
                "current": max(0.0, row_sensors["current"]),
                "noise_db": max(10.0, row_sensors["noise_db"]),
                "fault_label": current_label,
                "health_score": round(health_score, 4),
                "rul": rul,
            }
            records.append(record)

        return pd.DataFrame(records)

    def generate_dataset(
        self,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generates unified dataset across all simulated machines.

        Args:
            start_time: Initial timestamp. Defaults to current UTC time.

        Returns:
            DataFrame with [timestamp, machine_id, temp, vibration, pressure,
                            rpm, current, noise_db, fault_label, health_score, rul]
        """
        base_time = start_time or datetime(2026, 1, 1, 8, 0, 0)
        dfs = []

        logger.info(
            f"Generating synthetic dataset: {self.num_machines} machines, "
            f"{self.num_cycles} cycles each, fault_ratio={self.fault_ratio}"
        )

        for i in range(1, self.num_machines + 1):
            machine_id = f"MACH_{i:03d}"
            df_m = self.generate_machine_trajectory(
                machine_id=machine_id,
                start_time=base_time,
            )
            dfs.append(df_m)

        combined_df = pd.concat(dfs, ignore_index=True)
        # Ensure timestamp column is sorted and formatted properly
        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
        combined_df = combined_df.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
        return combined_df

    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Saves generated dataset to CSV or parquet.

        Args:
            df: Generated synthetic DataFrame.
            output_path: Target file path. Defaults to settings.synthetic_path / 'synthetic_sensor_data.csv'.

        Returns:
            Path of written file.
        """
        settings = get_settings()
        dest = Path(output_path) if output_path else settings.synthetic_path / "synthetic_sensor_data.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        logger.info(f"Saved {len(df)} synthetic records to {dest}")
        return dest


def main() -> None:
    """CLI tool for generating and inspecting industrial synthetic sensor data."""
    parser = argparse.ArgumentParser(description="Synthetic Industrial Sensor Data Generator")
    parser.add_argument("--machines", type=int, default=5, help="Number of machines to simulate")
    parser.add_argument("--cycles", type=int, default=500, help="Cycles per machine")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level multiplier")
    parser.add_argument("--fault-ratio", type=float, default=0.4, help="Proportion of failing machines")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = SyntheticIndustrialGenerator(
        num_machines=args.machines,
        num_cycles=args.cycles,
        noise_level=args.noise,
        fault_ratio=args.fault_ratio,
        random_seed=args.seed,
    )
    df = generator.generate_dataset()

    settings = get_settings()
    out_p = Path(args.output) if args.output else settings.synthetic_path / "synthetic_sensor_data.csv"
    generator.save_dataset(df, out_p)

    logger.info(f"Dataset summary:\n{df['fault_label'].value_counts()}")
    logger.info(f"Preview:\n{df.head(3)}")


if __name__ == "__main__":
    main()
