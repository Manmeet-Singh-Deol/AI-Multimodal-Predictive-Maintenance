"""Industrial Machine Troubleshooting and Predictive Maintenance AI Agent.

Orchestrates multimodal inputs (images, acoustic recordings, operator voice),
LangChain tool invocation (telemetry, manuals, ML health, RUL, maintenance logs),
safety guardrails, and persistent dialogue memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    ChatGoogleGenerativeAI = Any  # type: ignore[misc,assignment]
    AgentExecutor = Any  # type: ignore[misc,assignment]

from config.settings import get_settings
from src.chatbot.memory import ConversationMemory
from src.chatbot.safety_guardrails import SafetyGuardrails, ValidatedResponse
from src.chatbot.tools import (
    CHATBOT_TOOLS,
    analyze_machine_image,
    estimate_remaining_life,
    get_maintenance_history,
    get_sensor_readings,
    predict_machine_health,
    search_maintenance_manuals,
)
from src.multimodal.image_analyzer import ImageAnalyzer
from src.multimodal.speech_to_text import SpeechToText


SYSTEM_PROMPT = """You are Jarvis, a Senior Industrial Reliability Engineer and Certified Predictive Maintenance (PdM) Specialist.
You oversee high-value production machinery across the plant, including centrifugal pumps, induction motors, screw compressors, hydraulic systems, and gearboxes.

YOUR OBJECTIVE:
Diagnose industrial machinery faults, interpret sensor telemetry, recommend corrective actions, estimate remaining operating life, and guide technicians through repairs with 100% safety compliance.

TROUBLESHOOTING METHODOLOGY (5-Step Reliability Framework):
1. **SAFETY FIRST**: Always verify Lockout/Tagout (LOTO), zero-energy isolation, and required PPE before recommending physical contact with machinery.
2. **SYMPTOM & TELEMETRY TRIAGE**: Identify machine ID, review live sensor telemetry (vibration RMS, temperature, current, pressure, acoustic levels).
3. **HISTORICAL & DOCUMENT CORRELATION**: Query maintenance manuals and past work orders to identify known failure modes and previous repairs.
4. **ROOT-CAUSE HYPOTHESIS**: Formulate physics-backed root causes (e.g., bearing raceway fatigue, cavitation, unbalance, seal degradation, lubrication breakdown).
5. **ACTIONABLE REMEDIATION**: Provide numbered, step-by-step corrective procedures with required torque, lubrication specs, and verification checks.

TOOL USAGE DIRECTIVES:
- Use `get_sensor_readings(machine_id)` when the user mentions a specific machine or asks about current conditions.
- Use `predict_machine_health(machine_id)` to evaluate AI anomaly detection and fault classification.
- Use `estimate_remaining_life(machine_id)` when asked about RUL, hours/cycles remaining, or overhaul timing.
- Use `search_maintenance_manuals(query)` to find OEM operating procedures, torque values, and lubrication guides.
- Use `get_maintenance_history(machine_id)` to inspect past repairs and replaced components.
- Use `analyze_machine_image(image_description)` to analyze visual defects and wear patterns.

RESPONSE FORMAT:
Structure your troubleshooting responses clearly using markdown headings, bullet points, and tables. Keep instructions concise, highly technical, and actionable for plant floor technicians.
"""


@dataclass
class AgentResponse:
    """Standardized response container returned by the MaintenanceAgent.

    Attributes:
        message: Final assistant message text with safety advisories.
        sources: Technical manual citations or telemetry references used.
        tool_calls_made: List of tools executed with their inputs and outputs.
        safety_warnings: List of specific hazard warnings triggered.
        suggestions: List of proactive next-step recommendations for technician.
        ppe_required: List of mandatory PPE items required for the task.
        risk_level: Risk level assessed ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    """

    message: str
    sources: list[str] = field(default_factory=list)
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    safety_warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    ppe_required: list[str] = field(default_factory=list)
    risk_level: str = "LOW"

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "message": self.message,
            "sources": self.sources,
            "tool_calls_made": self.tool_calls_made,
            "safety_warnings": self.safety_warnings,
            "suggestions": self.suggestions,
            "ppe_required": self.ppe_required,
            "risk_level": self.risk_level,
        }


class MaintenanceAgent:
    """Central AI troubleshooting orchestrator for predictive maintenance."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        session_id: str = "default_session",
    ) -> None:
        """Initialize MaintenanceAgent with LLM, tools, memory, and multimodal engines.

        Args:
            api_key: Gemini API key. Defaults to Settings.gemini_api_key.
            model_name: Gemini model identifier. Defaults to Settings.gemini_model.
            temperature: LLM sampling temperature. Defaults to Settings.temperature.
            session_id: Conversation session identifier.
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.gemini_api_key
        self.model_name = model_name or self.settings.gemini_model or "gemini-1.5-flash"
        self.temperature = (
            temperature if temperature is not None else self.settings.temperature
        )
        self.session_id = session_id

        # Core subsystems
        self.tools = CHATBOT_TOOLS
        self.memory = ConversationMemory(
            max_history=self.settings.max_conversation_history,
            session_id=self.session_id,
        )
        self.guardrails = SafetyGuardrails()
        self.image_analyzer = ImageAnalyzer(api_key=self.api_key, model_name=self.model_name)
        self.speech_to_text = SpeechToText()

        # Agent Executor setup
        self.llm = None
        self.executor = None
        self._init_agent_executor()

    def _init_agent_executor(self) -> None:
        """Construct the LangChain tool-calling agent executor."""
        if not HAS_LANGCHAIN:
            logger.warning("LangChain is not installed; agent will use deterministic fallback router.")
            return

        if not self.api_key:
            logger.warning(
                "Gemini API key is not set. MaintenanceAgent initialized in offline fallback mode."
            )
            return

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self.executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                max_iterations=8,
            )
            logger.info("MaintenanceAgent LangChain executor initialized with model: {}", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize LangChain ChatGoogleGenerativeAI agent: {}", exc)
            self.executor = None

    def reset_conversation(self) -> None:
        """Clear active conversation memory and reset dialogue context."""
        self.memory.clear()
        logger.info("Conversation reset for session '{}'", self.session_id)

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Retrieve full conversation history for the current session."""
        return self.memory.get_history()

    def _extract_machine_id(self, text: str) -> str | None:
        """Heuristically extract machine identifiers like 'machine_1' or 'machine 2'."""
        match = re.search(r"\bmachine[_\s-]?(\d+)\b", text, re.IGNORECASE)
        if match:
            return f"machine_{match.group(1)}"
        return None

    def _extract_sources(self, intermediate_steps: list[Any], final_text: str) -> list[str]:
        """Extract citations and documentation sources from tool executions."""
        sources = set()

        # Extract from manual search outputs
        for step in intermediate_steps:
            try:
                action, output = step
                if "search_maintenance_manuals" in str(action):
                    for line in str(output).splitlines():
                        if "Source:" in line or "Handbook" in line or "Manual" in line or "Guide" in line or "SOP" in line:
                            clean_src = line.replace("📄", "").replace("📖", "").replace("**", "").strip()
                            if len(clean_src) > 5:
                                sources.add(clean_src)
            except Exception:
                continue

        # Extract ISO standards and OEM references from final text
        iso_matches = re.findall(r"\b(?:ISO\s*\d+(?:-\d+)?|API\s*Plan\s*\d+|NFPA\s*70E|OSHA\s*\d+\.\d+)\b", final_text)
        for iso in iso_matches:
            sources.add(iso)

        if not sources:
            sources.add("OEM Plant Maintenance Standard Operating Procedures")

        return sorted(sources)

    def _generate_proactive_suggestions(
        self,
        message: str,
        machine_id: str | None,
        tool_names: list[str],
    ) -> list[str]:
        """Generate smart follow-up suggestions for technician next steps."""
        suggestions = []
        target = machine_id.upper() if machine_id else "equipment"

        if "get_sensor_readings" in tool_names or "predict_machine_health" in tool_names:
            suggestions.append(f"Estimate Remaining Useful Life (RUL) for {target}")
            suggestions.append(f"Inspect historical work orders and replaced parts for {target}")
            suggestions.append(f"Search OEM manual for bearing lubrication and alignment specs")
        elif "estimate_remaining_life" in tool_names:
            suggestions.append(f"Check real-time vibration and temperature telemetry for {target}")
            suggestions.append(f"View recommended overhaul parts and maintenance history for {target}")
        elif "search_maintenance_manuals" in tool_names:
            suggestions.append(f"Retrieve current sensor readings for {target}")
            suggestions.append("Review required Personal Protective Equipment (PPE) checklist")
        else:
            suggestions.append("Check real-time sensor readings for Machine 1")
            suggestions.append("Run AI health & anomaly diagnostic on Machine 1")
            suggestions.append("Estimate Remaining Useful Life for Machine 1")
            suggestions.append("Review recent work orders and maintenance history")

        return suggestions[:4]

    def _fallback_tool_dispatcher(
        self,
        query: str,
        machine_id: str | None,
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        """Deterministic offline diagnostic dispatcher when Gemini API is unavailable."""
        target_mid = machine_id or self._extract_machine_id(query) or "machine_1"
        tool_calls: list[dict[str, Any]] = []
        sources: list[str] = []
        output_blocks: list[str] = []

        q_lower = query.lower()

        def _call_tool(tool_fn: Any, arg: Any) -> Any:
            """Invoke tool whether it is a LangChain BaseTool or raw function."""
            if hasattr(tool_fn, "invoke"):
                return tool_fn.invoke(arg)
            return tool_fn(arg)

        # Route 1: Telemetry & Sensor Readings
        if any(k in q_lower for k in ["sensor", "reading", "telemetry", "vibration", "temperature", "current", "pressure"]):
            res = _call_tool(get_sensor_readings, target_mid)
            tool_calls.append({"tool": "get_sensor_readings", "input": target_mid, "output": res})
            output_blocks.append(res)

        # Route 2: Machine Health & Anomaly Detection
        if any(k in q_lower for k in ["health", "anomaly", "predict", "fault", "condition", "status", "diagnos"]):
            res = _call_tool(predict_machine_health, target_mid)
            tool_calls.append({"tool": "predict_machine_health", "input": target_mid, "output": res})
            output_blocks.append(res)

        # Route 3: Remaining Useful Life
        if any(k in q_lower for k in ["rul", "remaining life", "useful life", "hours left", "cycle", "overhaul", "fail"]):
            res = _call_tool(estimate_remaining_life, target_mid)
            tool_calls.append({"tool": "estimate_remaining_life", "input": target_mid, "output": res})
            output_blocks.append(res)

        # Route 4: Maintenance Manuals / SOPs
        if any(k in q_lower for k in ["manual", "procedure", "sop", "how to", "lubricat", "grease", "spec", "torque", "cavitation"]):
            res = _call_tool(search_maintenance_manuals, query)
            tool_calls.append({"tool": "search_maintenance_manuals", "input": query, "output": res})
            output_blocks.append(res)
            sources.append("OEM Plant Maintenance Standard Operating Procedures")

        # Route 5: Maintenance History
        if any(k in q_lower for k in ["history", "log", "work order", "previous", "past repair", "replaced parts"]):
            res = _call_tool(get_maintenance_history, target_mid)
            tool_calls.append({"tool": "get_maintenance_history", "input": target_mid, "output": res})
            output_blocks.append(res)

        # Default fallback if no specific tool matched
        if not output_blocks:
            # Provide comprehensive telemetry and health overview
            sensor_res = _call_tool(get_sensor_readings, target_mid)
            health_res = _call_tool(predict_machine_health, target_mid)
            tool_calls.append({"tool": "get_sensor_readings", "input": target_mid, "output": sensor_res})
            tool_calls.append({"tool": "predict_machine_health", "input": target_mid, "output": health_res})
            output_blocks.extend([sensor_res, health_res])

        response_text = "\n\n".join(output_blocks)
        return response_text, tool_calls, sources

    def chat(
        self,
        user_message: str = "",
        image: bytes | None = None,
        audio: bytes | None = None,
        machine_id: str | None = None,
    ) -> AgentResponse:
        """Process technician input through multimodal perception, tool reasoning, and guardrails.

        Args:
            user_message: Natural language prompt or question from technician.
            image: Optional raw bytes of equipment photograph or HMI display.
            audio: Optional raw bytes of voice recording from technician.
            machine_id: Optional explicit machine ID (e.g., 'machine_1').

        Returns:
            AgentResponse dataclass populated with answers, sources, tool telemetry,
            safety warnings, and PPE requirements.
        """
        tool_calls_made: list[dict[str, Any]] = []
        sources: list[str] = []
        input_context_parts: list[str] = []

        # 1. Multimodal Perception: Audio Transcription
        if audio:
            logger.info("Technician provided audio input; initiating Whisper transcription...")
            trans_result = self.speech_to_text.transcribe_bytes(audio)
            if trans_result.text:
                logger.info("Transcribed audio: '{}'", trans_result.text)
                if not user_message:
                    user_message = trans_result.text
                else:
                    user_message = f"{user_message}\n[Spoken Operator Audio: \"{trans_result.text}\"]"
            tool_calls_made.append(
                {
                    "tool": "speech_to_text.transcribe_bytes",
                    "input": "<audio_bytes>",
                    "output": trans_result.to_dict(),
                }
            )

        # 2. Multimodal Perception: Image Analysis
        if image:
            logger.info("Technician provided image input; initiating Gemini Vision analysis...")
            img_result = self.image_analyzer.analyze_from_bytes(image, query=user_message)
            tool_calls_made.append(
                {
                    "tool": "image_analyzer.analyze_from_bytes",
                    "input": "<image_bytes>",
                    "output": img_result.to_dict(),
                }
            )
            vision_summary = (
                f"\n[VISUAL INSPECTION ANALYSIS]:\n"
                f"- Component Findings: {img_result.description}\n"
                f"- Identified Defects: {', '.join(img_result.defects_found) if img_result.defects_found else 'None'}\n"
                f"- Severity: {img_result.severity.upper()}\n"
                f"- Visual Recommendations: {', '.join(img_result.recommendations)}"
            )
            input_context_parts.append(vision_summary)

        # Determine target machine
        effective_mid = machine_id or self._extract_machine_id(user_message)
        if effective_mid:
            input_context_parts.append(f"[Target Equipment Context: {effective_mid.upper()}]")

        full_prompt = user_message
        if input_context_parts:
            full_prompt = f"{user_message}\n" + "\n".join(input_context_parts)

        if not full_prompt.strip():
            full_prompt = "Perform comprehensive diagnostic condition check for Machine 1."

        # 3. Pre-execution Safety Check
        query_warnings = self.guardrails.check_query(full_prompt)

        # 4. Agent Reasoning Execution
        raw_response = ""
        intermediate_steps: list[Any] = []

        if self.executor is not None:
            try:
                chat_history = self.memory.to_langchain_messages()
                logger.info("Executing LangChain agent executor with prompt: '{}'", full_prompt[:120])
                result = self.executor.invoke(
                    {
                        "input": full_prompt,
                        "chat_history": chat_history,
                    }
                )
                raw_response = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])

                for action, step_output in intermediate_steps:
                    tool_calls_made.append(
                        {
                            "tool": getattr(action, "tool", str(action)),
                            "input": getattr(action, "tool_input", {}),
                            "output": str(step_output)[:400],
                        }
                    )

                sources.extend(self._extract_sources(intermediate_steps, raw_response))
            except Exception as exc:
                logger.error("LangChain agent execution encountered an error: {}. Switching to fallback.", exc)
                raw_response, fallback_tools, fallback_src = self._fallback_tool_dispatcher(
                    full_prompt, effective_mid
                )
                tool_calls_made.extend(fallback_tools)
                sources.extend(fallback_src)
        else:
            logger.info("Using deterministic expert tool dispatcher (offline mode).")
            raw_response, fallback_tools, fallback_src = self._fallback_tool_dispatcher(
                full_prompt, effective_mid
            )
            tool_calls_made.extend(fallback_tools)
            sources.extend(fallback_src)

        # 5. Post-execution Safety Guardrails Validation
        validated: ValidatedResponse = self.guardrails.validate_response(raw_response)
        all_warnings = list(dict.fromkeys(query_warnings + validated.safety_warnings))

        # 6. Proactive Suggestions
        executed_tool_names = [call.get("tool", "") for call in tool_calls_made]
        suggestions = self._generate_proactive_suggestions(
            user_message, effective_mid, executed_tool_names
        )

        # 7. Update Persistent Memory
        self.memory.add_message(
            role="user",
            content=full_prompt,
            metadata={"machine_id": effective_mid},
        )
        self.memory.add_message(
            role="assistant",
            content=validated.response,
            metadata={
                "risk_level": validated.risk_level,
                "tool_calls_count": len(tool_calls_made),
            },
        )

        return AgentResponse(
            message=validated.response,
            sources=list(dict.fromkeys(sources)),
            tool_calls_made=tool_calls_made,
            safety_warnings=all_warnings,
            suggestions=suggestions,
            ppe_required=validated.ppe_required,
            risk_level=validated.risk_level,
        )
