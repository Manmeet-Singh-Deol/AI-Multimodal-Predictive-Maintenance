"""Industrial Machine Audio Processing and Acoustic Diagnostic Module.

Analyzes acoustic emissions and sound recordings of machinery to extract
spectral features, detect anomalies using reference baselines, classify
abnormal sound signatures (grinding, rattling, humming, clicking, squealing),
and generate visual spectrograms and waveforms.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore[assignment]
    HAS_SOUNDFILE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

from config.settings import get_settings


@dataclass
class AudioFeatures:
    """Acoustic feature representation of an industrial sound signal.

    Attributes:
        mel_spectrogram: Mel-scaled power spectrogram matrix.
        mfccs: 13 Mel-Frequency Cepstral Coefficients over time.
        spectral_centroid: Center of mass of the spectrum over time (Hz).
        spectral_bandwidth: Spectral spread/width around the centroid (Hz).
        spectral_rolloff: Frequency below which 85% of spectral energy lies (Hz).
        zero_crossing_rate: Rate of sign-changes along the signal frames.
        rms_energy: Root-mean-square energy per frame.
        chroma_features: 12-dimensional chroma energy distribution over time.
    """

    mel_spectrogram: np.ndarray
    mfccs: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray
    chroma_features: np.ndarray

    def to_summary_dict(self) -> dict[str, float]:
        """Compute scalar statistical summaries of all acoustic features."""
        return {
            "spectral_centroid_mean": float(np.mean(self.spectral_centroid)),
            "spectral_centroid_std": float(np.std(self.spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(self.spectral_bandwidth)),
            "spectral_rolloff_mean": float(np.mean(self.spectral_rolloff)),
            "zero_crossing_rate_mean": float(np.mean(self.zero_crossing_rate)),
            "zero_crossing_rate_std": float(np.std(self.zero_crossing_rate)),
            "rms_energy_mean": float(np.mean(self.rms_energy)),
            "rms_energy_std": float(np.std(self.rms_energy)),
            "rms_energy_max": float(np.max(self.rms_energy)) if self.rms_energy.size > 0 else 0.0,
            "mfcc_1_mean": float(np.mean(self.mfccs[0])) if self.mfccs.shape[0] > 0 else 0.0,
            "mfcc_2_mean": float(np.mean(self.mfccs[1])) if self.mfccs.shape[0] > 1 else 0.0,
            "mfcc_3_mean": float(np.mean(self.mfccs[2])) if self.mfccs.shape[0] > 2 else 0.0,
        }

    def to_feature_vector(self) -> np.ndarray:
        """Create a 1D feature vector for distance-based anomaly detection."""
        summaries = [
            np.mean(self.spectral_centroid),
            np.std(self.spectral_centroid),
            np.mean(self.spectral_bandwidth),
            np.std(self.spectral_bandwidth),
            np.mean(self.spectral_rolloff),
            np.mean(self.zero_crossing_rate),
            np.std(self.zero_crossing_rate),
            np.mean(self.rms_energy),
            np.std(self.rms_energy),
        ]
        # Append mean of first 6 MFCC coefficients
        for i in range(min(6, self.mfccs.shape[0])):
            summaries.append(float(np.mean(self.mfccs[i])))

        return np.array(summaries, dtype=np.float64)


@dataclass
class AnomalyResult:
    """Result of acoustic anomaly detection against normal operating baselines.

    Attributes:
        is_anomaly: Boolean flag indicating if acoustic anomaly is detected.
        confidence: Confidence score of the anomaly decision (0.0 to 1.0).
        anomaly_type: Type of acoustic fault identified or 'normal'.
        details: Technical description of deviations from nominal behavior.
        distance_score: Quantified statistical distance from reference baseline.
        threshold: Decision boundary used for anomaly detection.
    """

    is_anomaly: bool
    confidence: float
    anomaly_type: str
    details: str
    distance_score: float = 0.0
    threshold: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        """Convert anomaly result to serializable dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "details": self.details,
            "distance_score": round(self.distance_score, 4),
            "threshold": self.threshold,
        }


@dataclass
class SoundClassification:
    """Result of acoustic signature pattern classification.

    Attributes:
        predicted_class: Predicted sound category (normal, grinding, rattling, humming, clicking, squealing).
        confidence: Probability of the predicted sound class (0.0 to 1.0).
        class_probabilities: Probability distribution across all 6 supported classes.
        features_summary: Summary metrics extracted from the acoustic signal.
        explanation: Physical interpretation of acoustic indicators.
    """

    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    features_summary: dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert sound classification to serializable dictionary."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "class_probabilities": {k: round(v, 4) for k, v in self.class_probabilities.items()},
            "features_summary": {k: round(v, 4) for k, v in self.features_summary.items()},
            "explanation": self.explanation,
        }


class AudioProcessor:
    """Acoustic feature extractor, anomaly detector, and sound classifier.

    Designed for factory floor condition monitoring of pumps, compressors,
    bearings, motors, and transmissions.
    """

    SOUND_CLASSES = ["normal", "grinding", "rattling", "humming", "clicking", "squealing"]

    # Nominal healthy machinery baseline feature vector (mean, std)
    NOMINAL_BASELINE_MEAN = np.array([
        1450.0,   # Centroid mean (Hz)
        320.0,    # Centroid std (Hz)
        1800.0,   # Bandwidth mean (Hz)
        280.0,    # Bandwidth std (Hz)
        2900.0,   # Rolloff mean (Hz)
        0.055,    # ZCR mean
        0.018,    # ZCR std
        0.065,    # RMS energy mean
        0.015,    # RMS energy std
        -180.0,   # MFCC 0 mean
        45.0,     # MFCC 1 mean
        -12.0,    # MFCC 2 mean
        10.0,     # MFCC 3 mean
        -8.0,     # MFCC 4 mean
        4.0,      # MFCC 5 mean
    ], dtype=np.float64)

    NOMINAL_BASELINE_STD = np.array([
        250.0,    # Centroid mean std
        80.0,     # Centroid std std
        260.0,    # Bandwidth mean std
        60.0,     # Bandwidth std std
        450.0,    # Rolloff mean std
        0.015,    # ZCR mean std
        0.006,    # ZCR std std
        0.020,    # RMS energy mean std
        0.008,    # RMS energy std std
        35.0,     # MFCC 0 std
        15.0,     # MFCC 1 std
        8.0,      # MFCC 2 std
        6.0,      # MFCC 3 std
        5.0,      # MFCC 4 std
        4.0,      # MFCC 5 std
    ], dtype=np.float64)

    def __init__(
        self,
        reference_mean: np.ndarray | None = None,
        reference_std: np.ndarray | None = None,
        anomaly_threshold: float = 3.0,
    ) -> None:
        """Initialize AudioProcessor with optional custom reference baselines.

        Args:
            reference_mean: Mean vector of normal operating baseline.
            reference_std: Standard deviation vector of normal baseline.
            anomaly_threshold: Normalized statistical distance threshold for anomaly flag.
        """
        self.settings = get_settings()
        self.reference_mean = (
            reference_mean if reference_mean is not None else self.NOMINAL_BASELINE_MEAN.copy()
        )
        self.reference_std = (
            reference_std if reference_std is not None else self.NOMINAL_BASELINE_STD.copy()
        )
        self.anomaly_threshold = anomaly_threshold
        logger.info("AudioProcessor initialized with anomaly threshold: {}", self.anomaly_threshold)

    def load_audio(
        self,
        file_path: str | Path,
        sr: int = 22050,
    ) -> tuple[np.ndarray, int]:
        """Load an audio file, convert to mono, and resample to target rate.

        Args:
            file_path: Absolute or relative path to the audio file.
            sr: Target sampling rate (default 22050 Hz).

        Returns:
            Tuple of (audio_waveform, sample_rate).

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If audio decoding fails.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.debug("Loading audio file from '{}' at sr={}", path, sr)

        if HAS_LIBROSA:
            try:
                y, sample_rate = librosa.load(str(path), sr=sr, mono=True)
                return y.astype(np.float32), sample_rate
            except Exception as exc:
                logger.warning("librosa.load failed for '{}': {}. Trying soundfile.", path, exc)

        if HAS_SOUNDFILE:
            try:
                data, sample_rate = sf.read(str(path))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)  # Convert to mono
                if HAS_LIBROSA and sample_rate != sr:
                    data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
                    sample_rate = sr
                return data.astype(np.float32), sample_rate
            except Exception as exc:
                raise RuntimeError(f"soundfile failed to read audio '{path}': {exc}") from exc

        # Fallback if libraries are missing
        raise RuntimeError("Neither librosa nor soundfile is available to load audio files.")

    def load_from_bytes(
        self,
        audio_bytes: bytes,
        sr: int = 22050,
    ) -> tuple[np.ndarray, int]:
        """Load and decode audio from an in-memory byte buffer.

        Args:
            audio_bytes: Raw audio file bytes (WAV, MP3, OGG, FLAC).
            sr: Target sampling rate.

        Returns:
            Tuple of (audio_waveform, sample_rate).
        """
        if not audio_bytes:
            raise ValueError("Empty audio byte buffer provided.")

        bio = io.BytesIO(audio_bytes)
        if HAS_SOUNDFILE:
            try:
                data, sample_rate = sf.read(bio)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if HAS_LIBROSA and sample_rate != sr:
                    data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
                    sample_rate = sr
                return data.astype(np.float32), sample_rate
            except Exception as exc:
                logger.debug("soundfile failed to decode audio bytes: {}", exc)

        if HAS_LIBROSA:
            try:
                bio.seek(0)
                y, sample_rate = librosa.load(bio, sr=sr, mono=True)
                return y.astype(np.float32), sample_rate
            except Exception as exc:
                raise RuntimeError(f"Failed to decode audio bytes with librosa: {exc}") from exc

        raise RuntimeError("Audio decoding libraries (soundfile, librosa) unavailable.")

    def extract_features(
        self,
        audio: np.ndarray,
        sr: int = 22050,
    ) -> AudioFeatures:
        """Extract spectral, temporal, and energy features from an audio signal.

        Args:
            audio: 1D float numpy array representing audio waveform.
            sr: Sampling rate in Hz.

        Returns:
            AudioFeatures dataclass containing all extracted acoustic features.
        """
        if audio.size == 0:
            raise ValueError("Audio waveform is empty.")

        # Ensure float32 and 1D
        y = audio.astype(np.float32).flatten()

        if not HAS_LIBROSA:
            logger.warning("librosa unavailable; computing fallback acoustic features.")
            return self._fallback_features(y, sr)

        try:
            # 1. Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
            )

            # 2. MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)

            # 3. Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)

            # 4. Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)

            # 5. Spectral Rolloff (85% energy)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, roll_percent=0.85, hop_length=512
            )

            # 6. Zero Crossing Rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, hop_length=512)

            # 7. RMS Energy
            rms_energy = librosa.feature.rms(y=y, hop_length=512)

            # 8. Chroma Features (12 pitch classes)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)

            return AudioFeatures(
                mel_spectrogram=mel_spec,
                mfccs=mfccs,
                spectral_centroid=spectral_centroid,
                spectral_bandwidth=spectral_bandwidth,
                spectral_rolloff=spectral_rolloff,
                zero_crossing_rate=zero_crossing_rate,
                rms_energy=rms_energy,
                chroma_features=chroma,
            )
        except Exception as exc:
            logger.error("Feature extraction encountered an error: {}. Falling back.", exc)
            return self._fallback_features(y, sr)

    def _fallback_features(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Pure numpy fallback for extracting acoustic metrics when librosa fails."""
        # Simple STFT / FFT approximation
        n_fft = min(2048, len(y))
        hop = min(512, max(1, len(y) // 20))
        n_frames = max(1, (len(y) - n_fft) // hop + 1)

        rms_frames = []
        zcr_frames = []
        centroid_frames = []

        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        for i in range(n_frames):
            frame = y[i * hop : i * hop + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))

            rms_val = np.sqrt(np.mean(frame**2))
            rms_frames.append(rms_val)

            zcr_val = np.mean(np.abs(np.diff(np.signbit(frame))))
            zcr_frames.append(zcr_val)

            fft_mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            sum_mag = np.sum(fft_mag) + 1e-9
            centroid_val = np.sum(freqs * fft_mag) / sum_mag
            centroid_frames.append(centroid_val)

        rms_arr = np.array([rms_frames], dtype=np.float32)
        zcr_arr = np.array([zcr_frames], dtype=np.float32)
        cent_arr = np.array([centroid_frames], dtype=np.float32)
        band_arr = cent_arr * 0.8
        roll_arr = cent_arr * 1.8
        mfccs_mock = np.zeros((13, n_frames), dtype=np.float32)
        mel_mock = np.zeros((128, n_frames), dtype=np.float32)
        chroma_mock = np.zeros((12, n_frames), dtype=np.float32)

        return AudioFeatures(
            mel_spectrogram=mel_mock,
            mfccs=mfccs_mock,
            spectral_centroid=cent_arr,
            spectral_bandwidth=band_arr,
            spectral_rolloff=roll_arr,
            zero_crossing_rate=zcr_arr,
            rms_energy=rms_arr,
            chroma_features=chroma_mock,
        )

    def detect_anomaly(
        self,
        audio: np.ndarray,
        sr: int = 22050,
        reference_features: AudioFeatures | None = None,
    ) -> AnomalyResult:
        """Detect acoustic anomalies by comparing against a healthy operating baseline.

        Uses standardized Mahalanobis / Euclidean distance across spectral moments.

        Args:
            audio: 1D audio waveform array.
            sr: Sampling rate.
            reference_features: Optional specific healthy reference features to compare against.

        Returns:
            AnomalyResult with anomaly flag, confidence, type, and details.
        """
        features = self.extract_features(audio, sr)
        current_vec = features.to_feature_vector()

        if reference_features is not None:
            ref_vec = reference_features.to_feature_vector()
            ref_std = self.reference_std[: len(ref_vec)]
        else:
            ref_vec = self.reference_mean[: len(current_vec)]
            ref_std = self.reference_std[: len(current_vec)]

        # Standardized Euclidean distance (normalized z-score deviation)
        eps = 1e-6
        z_scores = np.abs((current_vec - ref_vec) / (ref_std + eps))
        distance = float(np.mean(z_scores))
        max_deviation = float(np.max(z_scores))

        is_anomaly = distance > self.anomaly_threshold or max_deviation > (self.anomaly_threshold * 2.2)

        # Compute confidence based on distance relative to threshold
        if is_anomaly:
            confidence = float(min(0.99, 0.70 + 0.29 * (distance / (self.anomaly_threshold * 2.5))))
        else:
            confidence = float(min(0.98, 0.65 + 0.33 * (1.0 - (distance / self.anomaly_threshold))))

        # Identify dominant anomaly type based on feature deviations
        mean_centroid = float(np.mean(features.spectral_centroid))
        mean_rms = float(np.mean(features.rms_energy))
        std_rms = float(np.std(features.rms_energy))
        mean_zcr = float(np.mean(features.zero_crossing_rate))

        anomaly_type = "normal"
        details_list: list[str] = []

        if is_anomaly:
            if mean_centroid > 3200.0 or mean_zcr > 0.12:
                anomaly_type = "high_frequency_resonance"
                details_list.append(
                    f"Elevated spectral centroid ({mean_centroid:.1f} Hz) and high ZCR ({mean_zcr:.3f}) "
                    "indicate friction, lack of lubrication, or dry bearing race contact."
                )
            elif std_rms > 0.045:
                anomaly_type = "erratic_transients"
                details_list.append(
                    f"High RMS variance (std={std_rms:.4f}) indicates impulsive knocking, "
                    "loose mechanical fasteners, or structural rattling."
                )
            elif mean_centroid < 850.0 and mean_rms > 0.10:
                anomaly_type = "low_frequency_pulsation"
                details_list.append(
                    f"Low frequency concentration (centroid={mean_centroid:.1f} Hz) with high energy "
                    "indicates motor unbalance, misalignment, or electrical 50/60 Hz hum."
                )
            elif mean_rms > 0.15:
                anomaly_type = "excessive_vibration_energy"
                details_list.append(
                    f"High continuous RMS energy ({mean_rms:.3f}) signifies heavy mechanical grinding "
                    "or imminent component binding."
                )
            else:
                anomaly_type = "unclassified_acoustic_deviation"
                details_list.append(
                    f"Acoustic profile deviates significantly from baseline (distance={distance:.2f}, "
                    f"threshold={self.anomaly_threshold})."
                )
        else:
            details_list.append(
                f"Acoustic emissions within nominal tolerances (distance score={distance:.2f} < "
                f"threshold={self.anomaly_threshold}). Machinery running smoothly."
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=round(confidence, 4),
            anomaly_type=anomaly_type,
            details=" ".join(details_list),
            distance_score=distance,
            threshold=self.anomaly_threshold,
        )

    def classify_sound(
        self,
        audio: np.ndarray,
        sr: int = 22050,
    ) -> SoundClassification:
        """Classify industrial sound into operational and failure categories.

        Classes:
            - normal: Balanced nominal hum and steady operation.
            - grinding: Continuous abrasive friction, wide high-energy bandwidth.
            - rattling: Periodic transient bursts, loose mechanical components.
            - humming: Strong low-frequency harmonic resonance, unbalance/electrical.
            - clicking: Sharp intermittent pulses, broken gear teeth/relays.
            - squealing: High-frequency whistling, dry bearing, belt slippage.

        Args:
            audio: 1D audio waveform.
            sr: Sampling rate.

        Returns:
            SoundClassification dataclass.
        """
        features = self.extract_features(audio, sr)
        summary = features.to_summary_dict()

        centroid = summary["spectral_centroid_mean"]
        bandwidth = summary["spectral_bandwidth_mean"]
        rolloff = summary["spectral_rolloff_mean"]
        zcr = summary["zero_crossing_rate_mean"]
        zcr_std = summary["zero_crossing_rate_std"]
        rms = summary["rms_energy_mean"]
        rms_std = summary["rms_energy_std"]
        rms_max = summary["rms_energy_max"]

        # Calculate Crest Factor (Peak to RMS ratio)
        peak_amp = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        crest_factor = peak_amp / (rms + 1e-6)

        # Compute raw logits for each class based on acoustic physics
        scores: dict[str, float] = {
            "normal": 0.0,
            "grinding": 0.0,
            "rattling": 0.0,
            "humming": 0.0,
            "clicking": 0.0,
            "squealing": 0.0,
        }

        # 1. Normal: Centroid in 1000-2400 Hz, low RMS variance, moderate energy
        if 900.0 <= centroid <= 2500.0 and rms < 0.10 and rms_std < 0.03 and crest_factor < 4.0:
            scores["normal"] += 3.5
        elif centroid < 2800.0 and rms < 0.12:
            scores["normal"] += 1.5

        # 2. Squealing: High centroid (>3200 Hz), high rolloff, elevated ZCR
        if centroid > 3200.0:
            scores["squealing"] += 3.0 + min(3.0, (centroid - 3200.0) / 800.0)
        if rolloff > 5000.0:
            scores["squealing"] += 2.0
        if zcr > 0.11:
            scores["squealing"] += 1.5

        # 3. Grinding: High continuous RMS, wide bandwidth, moderate-to-high centroid
        if rms > 0.11:
            scores["grinding"] += 2.5 + min(3.0, (rms - 0.11) * 20.0)
        if bandwidth > 2200.0:
            scores["grinding"] += 2.0
        if rms_std < 0.04 and rms > 0.08:  # Continuous steady grind
            scores["grinding"] += 1.5

        # 4. Rattling: High RMS fluctuation, irregular spikes, moderate frequency
        if rms_std > 0.035:
            scores["rattling"] += 2.5 + min(3.0, (rms_std - 0.035) * 60.0)
        if zcr_std > 0.025:
            scores["rattling"] += 1.5
        if 3.0 <= crest_factor <= 6.0 and rms_std > 0.025:
            scores["rattling"] += 1.5

        # 5. Humming: Dominant low frequency, low ZCR, low rolloff
        if centroid < 900.0:
            scores["humming"] += 3.0 + min(3.0, (900.0 - centroid) / 200.0)
        if rolloff < 1800.0:
            scores["humming"] += 2.0
        if zcr < 0.035:
            scores["humming"] += 1.5

        # 6. Clicking: Sharp transient bursts, high crest factor, low baseline RMS
        if crest_factor > 4.5:
            scores["clicking"] += 3.0 + min(3.0, (crest_factor - 4.5) * 0.8)
        if rms_max > (rms * 3.5) and rms < 0.08:
            scores["clicking"] += 2.5

        # Base bias so no class is zero
        for k in scores:
            scores[k] = max(0.1, scores[k])

        # Softmax normalization
        exp_scores = {k: np.exp(v) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        probabilities = {k: float(v / total_exp) for k, v in exp_scores.items()}

        best_class = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
        best_prob = probabilities[best_class]

        explanations = {
            "normal": "Acoustic signature demonstrates steady harmonic operation within nominal bounds.",
            "grinding": "Continuous abrasive friction detected with high broadband spectral energy.",
            "rattling": "Intermittent acoustic bursts and elevated RMS variance indicate loose components.",
            "humming": "Pronounced low-frequency spectral concentration indicates unbalance or electrical hum.",
            "clicking": "Sharp transient impulse spikes detected, consistent with gear tooth wear or valve tapping.",
            "squealing": "High-frequency acoustic resonance detected, typical of belt slippage or dry bearing race.",
        }

        return SoundClassification(
            predicted_class=best_class,
            confidence=best_prob,
            class_probabilities=probabilities,
            features_summary=summary,
            explanation=explanations.get(best_class, "Industrial sound classification completed."),
        )

    def plot_spectrogram(
        self,
        audio: np.ndarray,
        sr: int = 22050,
        save_path: str | Path | None = None,
    ) -> Any:
        """Generate a Mel-Spectrogram visualization plot.

        Args:
            audio: 1D audio waveform.
            sr: Sampling rate in Hz.
            save_path: Optional file path to save the generated image.

        Returns:
            Matplotlib Figure instance, or None if matplotlib is unavailable.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib is not available; cannot plot spectrogram.")
            return None

        features = self.extract_features(audio, sr)
        fig, ax = plt.subplots(figsize=(10, 4))

        if HAS_LIBROSA:
            s_db = librosa.power_to_db(features.mel_spectrogram, ref=np.max)
            img = librosa.display.specshow(
                s_db,
                x_axis="time",
                y_axis="mel",
                sr=sr,
                fmax=sr // 2,
                ax=ax,
                cmap="magma",
            )
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        else:
            # Fallback imshow
            img = ax.imshow(
                features.mel_spectrogram,
                aspect="auto",
                origin="lower",
                cmap="magma",
            )
            fig.colorbar(img, ax=ax)
            ax.set_xlabel("Frames")
            ax.set_ylabel("Mel Frequency Bands")

        ax.set_title("Machine Acoustic Mel-Spectrogram (Condition Monitoring)")
        fig.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(p), dpi=150)
            logger.info("Saved spectrogram plot to '{}'", p)

        return fig

    def plot_waveform(
        self,
        audio: np.ndarray,
        sr: int = 22050,
        save_path: str | Path | None = None,
    ) -> Any:
        """Generate an amplitude vs time waveform visualization.

        Args:
            audio: 1D audio waveform array.
            sr: Sampling rate.
            save_path: Optional file path to save the plot.

        Returns:
            Matplotlib Figure instance, or None if matplotlib is unavailable.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib is not available; cannot plot waveform.")
            return None

        fig, ax = plt.subplots(figsize=(10, 3))
        time_axis = np.linspace(0, len(audio) / sr, num=len(audio))

        ax.plot(time_axis, audio, color="#008080", alpha=0.85, linewidth=0.8)
        ax.set_title("Acoustic Emission Waveform (Time Domain)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(p), dpi=150)
            logger.info("Saved waveform plot to '{}'", p)

        return fig

    # --- Standard ML Lifecycle Methods (train, predict, save, load) ---

    def train(self, normal_audio_samples: list[np.ndarray], sr: int = 22050) -> None:
        """Calibrate reference baseline using a cohort of healthy operational recordings.

        Args:
            normal_audio_samples: List of healthy 1D audio waveforms.
            sr: Sampling rate.
        """
        if not normal_audio_samples:
            raise ValueError("No audio samples provided for baseline training.")

        logger.info("Training acoustic baseline with {} normal recordings", len(normal_audio_samples))
        vectors = []
        for sample in normal_audio_samples:
            feat = self.extract_features(sample, sr)
            vectors.append(feat.to_feature_vector())

        matrix = np.vstack(vectors)
        self.reference_mean = np.mean(matrix, axis=0)
        self.reference_std = np.std(matrix, axis=0) + 1e-4
        logger.info("Baseline calibration complete.")

    def predict(self, audio: np.ndarray, sr: int = 22050) -> dict[str, Any]:
        """Perform combined anomaly detection and acoustic signature classification.

        Args:
            audio: 1D audio waveform.
            sr: Sampling rate.

        Returns:
            Dictionary combining AnomalyResult and SoundClassification outputs.
        """
        anomaly = self.detect_anomaly(audio, sr)
        classification = self.classify_sound(audio, sr)
        return {
            "anomaly": anomaly.to_dict(),
            "classification": classification.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        """Save calibrated acoustic reference baseline to a JSON file.

        Args:
            path: Destination file path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reference_mean": self.reference_mean.tolist(),
            "reference_std": self.reference_std.tolist(),
            "anomaly_threshold": self.anomaly_threshold,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved AudioProcessor baseline to '{}'", target)

    def load(self, path: str | Path) -> None:
        """Load calibrated acoustic reference baseline from a JSON file.

        Args:
            path: Path to serialized baseline JSON file.
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Baseline file not found: {target}")

        with open(target, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.reference_mean = np.array(payload["reference_mean"], dtype=np.float64)
        self.reference_std = np.array(payload["reference_std"], dtype=np.float64)
        self.anomaly_threshold = float(payload.get("anomaly_threshold", self.anomaly_threshold))
        logger.info("Loaded AudioProcessor baseline from '{}'", target)
