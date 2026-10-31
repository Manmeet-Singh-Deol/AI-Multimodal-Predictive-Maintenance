"""Offline Industrial Speech-to-Text Module using OpenAI Whisper.

Transcribes spoken maintenance notes, hands-free technician logs, and
operator voice commands directly on the edge without external internet dependencies.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    whisper = None  # type: ignore[assignment]
    HAS_WHISPER = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore[assignment]
    HAS_SOUNDFILE = False


@dataclass
class TranscriptionResult:
    """Encapsulates the transcription result of an audio recording.

    Attributes:
        text: Transcribed textual content.
        language: Detected or specified language code (e.g., 'en', 'es', 'de').
        confidence: Estimated confidence score between 0.0 and 1.0.
        duration: Audio duration in seconds.
        segments: List of detailed segment timestamps and sub-phrases.
    """

    text: str
    language: str = "en"
    confidence: float = 0.90
    duration: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert transcription result to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "duration": round(self.duration, 2),
            "segment_count": len(self.segments),
        }


class SpeechToText:
    """Local offline speech-to-text transcriber using OpenAI Whisper.

    Features:
        - Lazy model loading (instantiates weights on first request).
        - Edge-compatible CPU/GPU inference.
        - Audio amplitude level normalization.
        - Support for WAV, MP3, M4A, WEBM, OGG, and FLAC containers.
    """

    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}

    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
    ) -> None:
        """Initialize the SpeechToText transcriber.

        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                        Defaults to 'base' for optimal balance of speed and accuracy.
            device: Computing device ('cpu', 'cuda', etc.). Defaults to auto-detect.
        """
        self.model_name = model_name
        self._device = device or self._detect_device()
        self._model = None
        logger.info(
            "SpeechToText configured with model='{}' on device='{}' (lazy load enabled)",
            self.model_name,
            self._device,
        )

    @staticmethod
    def _detect_device() -> str:
        """Detect the optimal PyTorch compute device."""
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_model(self) -> Any:
        """Lazy loader for the Whisper model weights.

        Returns:
            Loaded Whisper model instance.

        Raises:
            RuntimeError: If openai-whisper is not installed.
        """
        if self._model is not None:
            return self._model

        if not HAS_WHISPER:
            logger.error("openai-whisper is not installed in the current environment.")
            raise RuntimeError(
                "openai-whisper package is not installed. Run 'pip install openai-whisper'."
            )

        logger.info("Loading Whisper '{}' model onto '{}'...", self.model_name, self._device)
        try:
            self._model = whisper.load_model(self.model_name, device=self._device)
            logger.info("Whisper model loaded successfully.")
            return self._model
        except Exception as exc:
            logger.error("Failed to load Whisper model '{}': {}", self.model_name, exc)
            # Attempt CPU fallback if CUDA failed
            if self._device != "cpu":
                logger.info("Retrying Whisper loading on CPU fallback...")
                self._device = "cpu"
                self._model = whisper.load_model(self.model_name, device="cpu")
                return self._model
            raise RuntimeError(f"Unable to load Whisper model: {exc}") from exc

    def normalize_audio(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Apply peak amplitude normalization to improve speech recognition fidelity.

        Args:
            audio: 1D numpy array of audio samples.
            target_peak: Target peak amplitude (0.0 to 1.0).

        Returns:
            Normalized 1D numpy array.
        """
        if audio.size == 0:
            return audio

        peak = float(np.max(np.abs(audio)))
        if peak > 1e-5:
            scaling = target_peak / peak
            return (audio * scaling).astype(np.float32)
        return audio

    def _calculate_confidence(self, segments: list[dict[str, Any]]) -> float:
        """Derive overall confidence from Whisper segment log probabilities.

        Args:
            segments: List of segment dictionaries returned by Whisper.

        Returns:
            Confidence float between 0.0 and 1.0.
        """
        if not segments:
            return 0.85

        probs = []
        for seg in segments:
            avg_logprob = seg.get("avg_logprob")
            if avg_logprob is not None:
                # avg_logprob is in [-inf, 0]. Convert to probability: e^(avg_logprob)
                try:
                    p = math.exp(avg_logprob)
                    probs.append(max(0.0, min(1.0, p)))
                except OverflowError:
                    probs.append(0.0)

        if probs:
            return float(np.mean(probs))
        return 0.85

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file from disk.

        Args:
            audio_path: Path to the audio file.
            language: Optional ISO language code (e.g., 'en') to force specific language.

        Returns:
            TranscriptionResult dataclass.
        """
        path = Path(audio_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            logger.warning("File format '{}' is not in standard list {}. Proceeding anyway.", ext, self.SUPPORTED_FORMATS)

        logger.info("Transcribing audio file: {}", path)

        try:
            model = self._get_model()

            # Preprocess and normalize audio if possible
            if HAS_WHISPER:
                # whisper.load_audio resamples to 16kHz mono float32
                raw_audio = whisper.load_audio(str(path))
                norm_audio = self.normalize_audio(raw_audio)

                # Duration from sample count at 16kHz
                duration = len(norm_audio) / 16000.0

                # Transcribe with fp16=False for CPU stability
                use_fp16 = self._device == "cuda"
                options: dict[str, Any] = {"fp16": use_fp16}
                if language:
                    options["language"] = language

                result = model.transcribe(norm_audio, **options)

                text = result.get("text", "").strip()
                detected_lang = result.get("language", language or "en")
                segments = result.get("segments", [])
                confidence = self._calculate_confidence(segments)

                logger.info(
                    "Transcription finished ({} s, lang='{}', conf={:.2f}): '{}'",
                    round(duration, 1),
                    detected_lang,
                    confidence,
                    text[:80] + "..." if len(text) > 80 else text,
                )

                return TranscriptionResult(
                    text=text,
                    language=detected_lang,
                    confidence=confidence,
                    duration=duration,
                    segments=segments,
                )
        except Exception as exc:
            logger.error("Audio transcription failed for '{}': {}", path, exc)
            return TranscriptionResult(
                text=f"[Transcription Error: {exc}]",
                language=language or "en",
                confidence=0.0,
                duration=0.0,
                segments=[],
            )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        format: str = "wav",
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio provided directly as binary bytes.

        Saves bytes to a temporary container, executes transcription, and
        guarantees cleanup of the temporary file.

        Args:
            audio_bytes: Binary payload of the audio recording.
            format: Extension or container format (e.g., 'wav', 'mp3', 'webm').
            language: Optional language code hint.

        Returns:
            TranscriptionResult dataclass.
        """
        if not audio_bytes:
            return TranscriptionResult(
                text="",
                language=language or "en",
                confidence=0.0,
                duration=0.0,
                segments=[],
            )

        clean_ext = format.lower().lstrip(".")
        suffix = f".{clean_ext}"

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(audio_bytes)
                tf.flush()
                temp_file = Path(tf.name)

            return self.transcribe(temp_file, language=language)
        finally:
            if temp_file is not None and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as exc:
                    logger.debug("Failed to delete temp audio file {}: {}", temp_file, exc)
