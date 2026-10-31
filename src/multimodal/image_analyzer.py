"""Industrial Machine Image Analysis Module using Google Gemini Vision.

Provides defect detection, wear/corrosion identification, and control panel
error code extraction from industrial component photographs.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    HAS_GENAI = True
except ImportError:
    genai = None  # type: ignore[assignment]
    GenerateContentResponse = Any  # type: ignore[misc,assignment]
    HAS_GENAI = False

from config.settings import get_settings


@dataclass
class ImageAnalysisResult:
    """Structured result of an industrial image defect analysis.

    Attributes:
        description: Comprehensive narrative of component state and visual findings.
        defects_found: List of detected faults (e.g., surface pitting, leak, crack).
        severity: Assessed severity level ('none', 'low', 'medium', 'high', 'critical').
        recommendations: Actionable maintenance steps and corrective interventions.
        raw_response: Full text output returned by Gemini.
        confidence: Estimated confidence score between 0.0 and 1.0.
    """

    description: str
    defects_found: list[str] = field(default_factory=list)
    severity: str = "none"
    recommendations: list[str] = field(default_factory=list)
    raw_response: str = ""
    confidence: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a serializable dictionary."""
        return {
            "description": self.description,
            "defects_found": self.defects_found,
            "severity": self.severity,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
        }


class ImageAnalyzer:
    """Industrial image analyzer powered by Google Gemini Vision.

    Analyzes photographs of industrial machinery, mechanical assemblies,
    bearings, pipes, and electrical panels to identify anomalies, wear patterns,
    corrosion, fluid leaks, and HMI display error codes.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    DEFAULT_PROMPT = (
        "Analyze this industrial machine/component image. Identify any visible defects, "
        "damage, wear, corrosion, leaks, misalignment, or abnormal conditions. "
        "Describe the condition and suggest possible causes."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the ImageAnalyzer with Gemini API credentials.

        Args:
            api_key: Optional Gemini API key. Defaults to Settings.gemini_api_key.
            model_name: Gemini model name. Defaults to Settings.gemini_model.
        """
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model or "gemini-1.5-flash"
        self._model = None

        if not HAS_GENAI:
            logger.warning(
                "google-generativeai package is not installed. Live vision queries will fail."
            )
        elif self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
                logger.info(
                    "ImageAnalyzer initialized with Gemini model: {}", self.model_name
                )
            except Exception as exc:
                logger.error("Failed to configure Gemini GenerativeModel: {}", exc)
                self._model = None
        else:
            logger.warning(
                "Gemini API key is not configured. Live image analysis will be disabled."
            )

    def _get_model(self) -> Any:
        """Ensure and return the configured Gemini model instance."""
        if self._model is not None:
            return self._model

        if not HAS_GENAI:
            raise RuntimeError(
                "google-generativeai is not installed. Please install requirements."
            )

        settings = get_settings()
        key = self.api_key or settings.gemini_api_key
        if not key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY in environment or .env file."
            )

        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def _validate_image_file(self, image_path: Path) -> None:
        """Validate that the image file exists and has a supported extension.

        Args:
            image_path: Path to the image file.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the file format is unsupported.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        ext = image_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{ext}'. Supported formats: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    def _build_analysis_prompt(self, user_query: str | None = None) -> str:
        """Construct the prompt instructing Gemini to return structured JSON.

        Args:
            user_query: Optional custom diagnostic question or user focus.

        Returns:
            Formatted prompt string.
        """
        base_query = user_query.strip() if user_query else self.DEFAULT_PROMPT
        return f"""You are a master industrial reliability engineer and certified machine diagnostic specialist.
Examine this industrial component photograph with high technical precision.

Task:
{base_query}

Respond strictly with a valid JSON object matching this schema:
{{
  "description": "Detailed technical description of the visible component, operating environment, and observed state.",
  "defects_found": ["Specific defect 1", "Specific defect 2"],
  "severity": "none | low | medium | high | critical",
  "recommendations": ["Immediate corrective action 1", "Maintenance recommendation 2"],
  "confidence": 0.90
}}

Guidelines for Severity:
- none: Component appears healthy, clean, properly lubricated, no visible damage.
- low: Minor surface discoloration, light cosmetic dust, acceptable superficial wear.
- medium: Noticeable seal seepage, moderate rust, minor belt fraying, requires scheduled maintenance.
- high: Active fluid leakage, heavy corrosion pitting, severe thermal scorching, structural crack.
- critical: Catastrophic structural rupture, severe mechanical misalignment, immediate safety shutdown hazard.

Do not include any text, markdown backticks, or commentary outside of the JSON object.
"""

    def _parse_gemini_json(self, raw_text: str) -> ImageAnalysisResult:
        """Parse Gemini's textual response into an ImageAnalysisResult dataclass.

        Args:
            raw_text: Text response from the model.

        Returns:
            ImageAnalysisResult object populated with parsed data.
        """
        cleaned = raw_text.strip()
        # Remove markdown code block fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            severity = str(data.get("severity", "medium")).lower()
            if severity not in {"none", "low", "medium", "high", "critical"}:
                severity = "medium"

            defects = data.get("defects_found") or []
            if isinstance(defects, str):
                defects = [defects]

            recs = data.get("recommendations") or []
            if isinstance(recs, str):
                recs = [recs]

            conf = float(data.get("confidence", 0.85))

            return ImageAnalysisResult(
                description=str(data.get("description", "Analysis completed.")),
                defects_found=[str(d) for d in defects],
                severity=severity,
                recommendations=[str(r) for r in recs],
                raw_response=raw_text,
                confidence=max(0.0, min(1.0, conf)),
            )
        except json.JSONDecodeError:
            logger.warning("Gemini output was not strict JSON; executing heuristic parsing.")
            return self._heuristic_parse(raw_text)

    def _heuristic_parse(self, raw_text: str) -> ImageAnalysisResult:
        """Fallback parser when model output is freeform markdown or plain text.

        Args:
            raw_text: Raw output string.

        Returns:
            Heuristically parsed ImageAnalysisResult.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        description = raw_text[:500]
        defects: list[str] = []
        recommendations: list[str] = []
        severity = "medium"

        lower_text = raw_text.lower()
        if any(w in lower_text for w in ["critical", "catastrophic", "immediate shutdown", "rupture"]):
            severity = "critical"
        elif any(w in lower_text for w in ["high", "severe", "heavy corrosion", "active leak"]):
            severity = "high"
        elif any(w in lower_text for w in ["low", "minor", "slight", "cosmetic"]):
            severity = "low"
        elif any(w in lower_text for w in ["normal", "no defect", "healthy", "optimal"]):
            severity = "none"

        for line in lines:
            line_clean = re.sub(r"^[-*•\d\.\)]\s*", "", line)
            if any(term in line.lower() for term in ["defect", "damage", "wear", "corrosion", "leak"]):
                if len(line_clean) > 5 and len(defects) < 5:
                    defects.append(line_clean)
            elif any(term in line.lower() for term in ["recommend", "action", "replace", "inspect", "clean"]):
                if len(line_clean) > 5 and len(recommendations) < 5:
                    recommendations.append(line_clean)

        if not defects:
            defects = ["Visual anomalies noted; see description for details."]
        if not recommendations:
            recommendations = ["Conduct physical inspection and verify operational clearance."]

        return ImageAnalysisResult(
            description=description,
            defects_found=defects,
            severity=severity,
            recommendations=recommendations,
            raw_response=raw_text,
            confidence=0.75,
        )

    def analyze_machine_image(
        self,
        image_path: str | Path,
        query: str | None = None,
    ) -> ImageAnalysisResult:
        """Analyze an industrial machine image file from disk.

        Args:
            image_path: Absolute or relative file path to the image.
            query: Optional specific user question or area of interest.

        Returns:
            ImageAnalysisResult dataclass containing defect details and recommendations.
        """
        path = Path(image_path).resolve()
        logger.info("Analyzing machine image from file: {}", path)

        try:
            self._validate_image_file(path)
            with Image.open(path) as pil_image:
                pil_image.load()
                # Create a copy in RGB/RGBA format
                rgb_image = pil_image.convert("RGB")
                return self._call_gemini_vision(rgb_image, query)
        except Exception as exc:
            logger.error("Image analysis failed for '{}': {}", path, exc)
            return ImageAnalysisResult(
                description=f"Image analysis error: {exc}",
                defects_found=[f"Error reading or processing image: {exc}"],
                severity="unknown",
                recommendations=["Verify image file exists, is uncorrupted, and GEMINI_API_KEY is valid."],
                raw_response=str(exc),
                confidence=0.0,
            )

    def analyze_from_bytes(
        self,
        image_bytes: bytes,
        query: str | None = None,
    ) -> ImageAnalysisResult:
        """Analyze an industrial machine image provided as raw bytes.

        Args:
            image_bytes: Raw binary content of the image.
            query: Optional specific user question or area of interest.

        Returns:
            ImageAnalysisResult dataclass containing defect details and recommendations.
        """
        if not image_bytes:
            return ImageAnalysisResult(
                description="Empty image payload received.",
                defects_found=["No image data provided."],
                severity="none",
                recommendations=["Please upload a valid image file."],
                confidence=0.0,
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                pil_image.load()
                rgb_image = pil_image.convert("RGB")
                return self._call_gemini_vision(rgb_image, query)
        except Exception as exc:
            logger.error("Image byte analysis failed: {}", exc)
            return ImageAnalysisResult(
                description=f"Failed to decode image bytes: {exc}",
                defects_found=[f"Image decode error: {exc}"],
                severity="unknown",
                recommendations=["Ensure image format is JPG, PNG, WEBP, or GIF."],
                raw_response=str(exc),
                confidence=0.0,
            )

    def _call_gemini_vision(
        self,
        image: Image.Image,
        query: str | None = None,
    ) -> ImageAnalysisResult:
        """Invoke Gemini Vision API with a PIL image and formatted prompt.

        Args:
            image: In-memory PIL Image object.
            query: Optional diagnostic prompt.

        Returns:
            ImageAnalysisResult instance.
        """
        try:
            model = self._get_model()
            prompt = self._build_analysis_prompt(query)
            logger.debug("Dispatching vision prompt to model '{}'", self.model_name)
            response: GenerateContentResponse = model.generate_content([prompt, image])
            raw_text = response.text or ""
            return self._parse_gemini_json(raw_text)
        except Exception as exc:
            logger.error("Gemini Vision API invocation failed: {}", exc)
            return ImageAnalysisResult(
                description=f"Gemini API invocation failure: {exc}",
                defects_found=["Vision API call failed"],
                severity="unknown",
                recommendations=[
                    "Check network connectivity and verify GEMINI_API_KEY quota/permissions."
                ],
                raw_response=str(exc),
                confidence=0.0,
            )

    def extract_error_code(self, image_path: str | Path) -> str:
        """Extract error codes, alarm text, and fault IDs from a control panel image.

        Args:
            image_path: Path to the image of the control panel, HMI, or instrument.

        Returns:
            Formatted string of extracted error codes and displayed diagnostic messages.
        """
        path = Path(image_path).resolve()
        logger.info("Extracting error code from panel image: {}", path)

        try:
            self._validate_image_file(path)
            with Image.open(path) as pil_image:
                pil_image.load()
                rgb_image = pil_image.convert("RGB")
                return self.extract_error_code_from_image(rgb_image)
        except Exception as exc:
            logger.error("Failed to extract error code from '{}': {}", path, exc)
            return f"Error extracting code: {exc}"

    def extract_error_code_from_bytes(self, image_bytes: bytes) -> str:
        """Extract error codes and messages from control panel image bytes.

        Args:
            image_bytes: Raw binary content of the control panel photograph.

        Returns:
            Formatted string of detected error codes and readings.
        """
        if not image_bytes:
            return "No image data provided."

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                pil_image.load()
                rgb_image = pil_image.convert("RGB")
                return self.extract_error_code_from_image(rgb_image)
        except Exception as exc:
            logger.error("Failed to extract error code from bytes: {}", exc)
            return f"Error decoding image: {exc}"

    def extract_error_code_from_image(self, image: Image.Image) -> str:
        """Perform OCR and error extraction on a loaded PIL image using Gemini.

        Args:
            image: PIL Image object of the control panel.

        Returns:
            String with identified error codes and status descriptions.
        """
        prompt = (
            "You are an industrial automation and SCADA technician. Examine this "
            "control panel, HMI screen, or diagnostic display. Read and extract all "
            "visible error codes, alarm banners, fault codes, indicator lights, and status messages.\n"
            "Format the output clearly as:\n"
            "Error Code: [Code, e.g., ERR-401, F03, E-STOP]\n"
            "Alarm Description: [Text shown on screen]\n"
            "Subsystem: [Drive, Inverter, Hydraulic, PLC, etc.]\n"
            "Severity: [Info | Warning | Fault | Critical Trip]\n"
            "Suggested First Step: [Primary action to acknowledge or reset]\n"
            "If no error code is visible, state 'No error codes detected on display.'"
        )

        try:
            model = self._get_model()
            response: GenerateContentResponse = model.generate_content([prompt, image])
            return (response.text or "").strip()
        except Exception as exc:
            logger.error("Gemini error code extraction failed: {}", exc)
            return f"Error code extraction unavailable: {exc}"
