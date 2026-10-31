"""Industrial Safety Guardrails and Regulatory Compliance Engine.

Enforces OSHA, NFPA 70E, and ISO 14118 industrial safety protocols across
chatbot responses. Automatically intercepts hazardous queries, injects LOTO
(Lockout/Tagout) directives, and mandates personal protective equipment (PPE).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ValidatedResponse:
    """Represents a chatbot response validated against industrial safety rules.

    Attributes:
        response: Final validated text with any auto-injected safety protocols.
        safety_warnings: List of specific hazard warnings identified.
        ppe_required: List of mandatory Personal Protective Equipment items.
        risk_level: Risk assessment tier ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
        loto_required: Flag indicating if Lockout/Tagout isolation is mandatory.
    """

    response: str
    safety_warnings: list[str] = field(default_factory=list)
    ppe_required: list[str] = field(default_factory=list)
    risk_level: str = "LOW"
    loto_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert validated response to serializable dictionary."""
        return {
            "response": self.response,
            "safety_warnings": self.safety_warnings,
            "ppe_required": self.ppe_required,
            "risk_level": self.risk_level,
            "loto_required": self.loto_required,
        }


class SafetyGuardrails:
    """Industrial safety guardrails engine for predictive maintenance operations."""

    HIGH_RISK_KEYWORDS = [
        "high voltage",
        "lockout",
        "tagout",
        "electrical panel",
        "energized",
        "pressurized",
        "chemical",
        "confined space",
        "hot surface",
        "rotating parts",
        "hydraulic",
        # Extended industrial hazards
        "pneumatic",
        "arc flash",
        "steam line",
        "caustic",
        "acid",
        "refrigerant",
        "ammonia",
        "pinch point",
    ]

    SAFETY_PROTOCOLS: dict[str, list[str]] = {
        "high voltage": [
            "De-energize main supply circuit breaker upstream before opening equipment.",
            "Verify ZERO energy state using a calibrated, CAT-IV rated multimeter.",
            "Apply Lockout/Tagout (LOTO) padlock, hasp, and authorized danger tag.",
            "Wear NFPA 70E compliant Arc Flash face shield and 1000V insulating rubber gloves.",
        ],
        "lockout": [
            "Notify all affected facility operators and production leads prior to shutdown.",
            "Perform standard equipment shutdown sequence via local HMI/stop controls.",
            "Physically isolate all energy-disconnecting devices (breakers, disconnects, valves).",
            "Affix personal safety padlock and serialized tag to each isolation point.",
            "Dissipate stored energy (bleed hydraulic accumulators, discharge capacitors, block gravity loads).",
            "Attempt local restart to physically verify zero-energy isolation before servicing.",
        ],
        "tagout": [
            "Affix standardized, weather-resistant danger tag securely to isolation mechanism.",
            "Record authorized technician name, contact extension, reason, and timestamp.",
            "Never use tagout alone if a physical lockout padlock mechanism is feasible.",
        ],
        "electrical panel": [
            "Never open panel doors while energized unless certified under NFPA 70E Live Work Permit.",
            "Establish and mark arc-flash boundary perimeter (minimum 36-inch clearance).",
            "Wear safety glasses with side shields, full face shield, and voltage-rated gloves.",
            "Ensure only insulated hand tools rated to 1000V (ASTM F1505) are used inside the cabinet.",
        ],
        "energized": [
            "Obtain signed Energized Electrical Work Permit from facility safety supervisor.",
            "Establish restricted approach perimeter to prevent entry by non-certified personnel.",
            "Use insulated tools and arc-resistant test instruments rated for system voltage.",
            "Station a second qualified safety observer trained in CPR and emergency de-energization.",
        ],
        "pressurized": [
            "Isolate primary pressure supply valves and tag out before loosening any fittings.",
            "Slowly crack open bleeder or relief valve to vent system pressure down to 0.0 PSI / Bar.",
            "Verify pressure gauges at both upstream and downstream headers read zero.",
            "Stand clear of line trajectory to guard against projectile hazard during uncoupling.",
        ],
        "hydraulic": [
            "Lower all hydraulic cylinders, booms, and counterweights to ground or mechanically chock them.",
            "De-energize hydraulic power unit (HPU) and bleed high-pressure accumulator circuit.",
            "NEVER use bare hands or thin gloves to check for leaks; use cardboard (fluid injection hazard).",
            "Allow hydraulic fluid to cool below 40°C before breaking pipe joints or replacing filters.",
        ],
        "pneumatic": [
            "Close primary air supply ball valve and engage slide exhaust valve.",
            "Exhaust air reservoir tanks and check receiver pressure gauge reads 0 PSI.",
            "Secure flexible air lines with safety whip checks to prevent hazardous hose whipping.",
        ],
        "chemical": [
            "Consult chemical Safety Data Sheet (SDS) for exposure limits and neutralization protocol.",
            "Verify emergency eyewash station and safety deluge shower are within 10 seconds travel time.",
            "Don chemical splash goggles, neoprene/nitrile heavy gauntlets, and chemical apron.",
            "Ensure continuous active local exhaust ventilation is operating in the work cell.",
        ],
        "confined space": [
            "Obtain authorized Confined Space Entry Permit prior to opening manway or vessel.",
            "Conduct continuous 4-gas atmospheric testing (O2: 19.5-23.5%, LEL < 10%, H2S < 10ppm, CO < 35ppm).",
            "Station a dedicated, trained safety attendant outside the portal at all times.",
            "Equip entrant with full-body safety harness anchored to retrieval tripod winch.",
        ],
        "hot surface": [
            "Allow component to cool naturally or verify surface temperature is below 50°C (122°F).",
            "Measure surface temperature with a calibrated infrared thermal pyrometer before contact.",
            "Wear heavy-duty Kevlar or aluminized heat-resistant gloves if hot work is unavoidable.",
            "Erect high-temperature caution signage to alert adjacent shift workers.",
        ],
        "rotating parts": [
            "Ensure machine spindle, motor shaft, or conveyor is at a complete zero-RPM standstill.",
            "Apply LOTO to motor disconnect switch to prevent automated PLC restarts.",
            "Remove all loose clothing, jewelry, rings, and lanyards; tie back long hair securely.",
            "Never bridge, bypass, or tamper with safety interlock switches, light curtains, or guards.",
        ],
    }

    PPE_CATALOG: dict[str, list[str]] = {
        "electrical": [
            "NFPA 70E Arc Flash Face Shield (Category 2+)",
            "1000V Insulating Rubber Gloves with Leather Protectors",
            "Dielectric Safety Boots (EH Rated)",
            "Safety Glasses with Side Shields (ANSI Z87.1)",
        ],
        "hydraulic_pneumatic": [
            "Full Face Shield",
            "Impact-Resistant Safety Glasses",
            "Heavy-Duty Nitrile / Chemical-Resistant Gloves",
            "Steel-Toe Safety Boots (ASTM F2413)",
        ],
        "mechanical": [
            "Snug-Fitting Mechanic Gloves (Abrasion Resistant)",
            "Steel-Toe Work Boots",
            "ANSI Z87.1 Safety Glasses",
            "Hearing Protection (NRR 25+ Earplugs or Earmuffs)",
            "Hard Hat (ANSI Z89.1 Type I)",
        ],
        "chemical": [
            "Chemical Splash Goggles",
            "Full Face Shield",
            "Nitrile / Butyl Heavy Gauntlets",
            "Chemical-Resistant Apron / Tyvek Suit",
            "NIOSH Half-Mask Respirator (Organic Vapor / Acid Cartridges)",
        ],
        "thermal": [
            "Heat-Resistant Kevlar Thermal Gloves (Rated 250°C+)",
            "Aluminized Thermal Arm Sleeves",
            "Safety Glasses and Face Shield",
        ],
        "general": [
            "ANSI Z87.1 Approved Safety Glasses",
            "Steel-Toe Safety Boots",
            "High-Visibility Safety Vest",
            "Standard Work Gloves",
        ],
    }

    def __init__(self) -> None:
        """Initialize SafetyGuardrails engine."""
        logger.info(
            "SafetyGuardrails initialized with {} risk categories.",
            len(self.SAFETY_PROTOCOLS),
        )

    def _match_keywords(self, text: str) -> list[str]:
        """Identify hazardous keywords present in given text."""
        lowered = text.lower()
        matched = []
        for kw in self.HIGH_RISK_KEYWORDS:
            # Match whole phrase or word boundary
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, lowered):
                matched.append(kw)
        return matched

    def check_query(self, query: str) -> list[str]:
        """Analyze user query and return applicable safety hazard warnings.

        Args:
            query: User's prompt or question.

        Returns:
            List of detected hazard warning strings.
        """
        matched = self._match_keywords(query)
        warnings = []
        for kw in matched:
            warning_text = self.format_safety_warning(kw)
            if warning_text not in warnings:
                warnings.append(warning_text)

        if matched:
            logger.warning(
                "User query triggered safety guardrails for risks: {}", matched
            )
        return warnings

    def format_safety_warning(self, warning_type: str) -> str:
        """Format an industrial safety warning string with ⚠️ emoji and steps.

        Args:
            warning_type: Key from HIGH_RISK_KEYWORDS or SAFETY_PROTOCOLS.

        Returns:
            Formatted multi-line warning string.
        """
        protocol_key = warning_type.lower()
        steps = self.SAFETY_PROTOCOLS.get(protocol_key)

        # Fallback search if exact key isn't present
        if not steps:
            for k, v in self.SAFETY_PROTOCOLS.items():
                if k in protocol_key or protocol_key in k:
                    steps = v
                    protocol_key = k
                    break

        if not steps:
            return (
                f"⚠️ **SAFETY ALERT - [{warning_type.upper()}]**: "
                "Hazardous maintenance procedure detected. Adhere strictly to plant safety guidelines, "
                "verify zero energy state, and wear appropriate PPE."
            )

        header = f"⚠️ **CRITICAL SAFETY PROTOCOL - [{protocol_key.upper()}]**"
        lines = [header, "Before proceeding, you MUST strictly adhere to the following steps:"]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"  {idx}. {step}")

        return "\n".join(lines)

    def get_ppe_requirements(self, task_type: str | list[str]) -> list[str]:
        """Determine required PPE items based on task description or risk types.

        Args:
            task_type: Task string or list of detected risk categories.

        Returns:
            List of required PPE items.
        """
        items: list[str] = []

        if isinstance(task_type, list):
            context = " ".join(task_type).lower()
        else:
            context = task_type.lower()

        is_electrical = any(
            k in context for k in ["electrical", "voltage", "panel", "energized", "breaker", "arc"]
        )
        is_hydraulic = any(
            k in context for k in ["hydraulic", "pressurized", "pneumatic", "pressure", "psi", "fluid"]
        )
        is_chemical = any(
            k in context for k in ["chemical", "acid", "caustic", "solvent", "ammonia", "refrigerant"]
        )
        is_thermal = any(
            k in context for k in ["hot", "surface", "thermal", "heat", "steam", "welding"]
        )
        is_mechanical = any(
            k in context for k in ["rotating", "gear", "bearing", "motor", "shaft", "belt", "lockout"]
        )

        if is_electrical:
            items.extend(self.PPE_CATALOG["electrical"])
        if is_hydraulic:
            items.extend(self.PPE_CATALOG["hydraulic_pneumatic"])
        if is_chemical:
            items.extend(self.PPE_CATALOG["chemical"])
        if is_thermal:
            items.extend(self.PPE_CATALOG["thermal"])
        if is_mechanical:
            items.extend(self.PPE_CATALOG["mechanical"])

        if not items:
            items.extend(self.PPE_CATALOG["general"])

        # Deduplicate while preserving ordering
        seen = set()
        deduped = []
        for it in items:
            if it not in seen:
                seen.add(it)
                deduped.append(it)

        return deduped

    def validate_response(self, response: str) -> ValidatedResponse:
        """Validate LLM response against safety protocols and inject missing warnings.

        Inspects the response for hazardous operations, assesses risk level,
        verifies whether LOTO instructions are required, and appends mandatory
        safety disclosures and PPE checklists if absent.

        Args:
            response: Assistant response text generated by LLM.

        Returns:
            ValidatedResponse dataclass with final response and safety metadata.
        """
        matched_risks = self._match_keywords(response)

        # Assess risk level
        risk_level = "LOW"
        critical_terms = ["high voltage", "energized", "arc flash", "confined space", "steam line"]
        high_terms = ["electrical panel", "hydraulic", "pressurized", "chemical", "rotating parts"]

        if any(term in matched_risks for term in critical_terms):
            risk_level = "CRITICAL"
        elif any(term in matched_risks for term in high_terms):
            risk_level = "HIGH"
        elif matched_risks:
            risk_level = "MEDIUM"

        # Determine if LOTO is mandatory
        loto_triggers = [
            "lockout", "tagout", "electrical panel", "high voltage",
            "rotating parts", "hydraulic", "pneumatic", "energized"
        ]
        loto_required = any(k in matched_risks for k in loto_triggers)

        # Retrieve required PPE
        ppe_list = self.get_ppe_requirements(matched_risks if matched_risks else response)

        # Build list of warnings
        safety_warnings = [self.format_safety_warning(r) for r in matched_risks]

        # Check if response already includes explicit safety warnings or LOTO
        has_warning_emoji = "⚠️" in response or "WARNING" in response.upper() or "CAUTION" in response.upper()
        has_loto_mention = "lockout" in response.lower() or "loto" in response.lower()

        injected_sections: list[str] = []

        # If high/critical risk and no prominent warning present, inject critical warning block
        if risk_level in {"HIGH", "CRITICAL"} and not has_warning_emoji:
            warning_blocks = []
            for risk in matched_risks[:2]:  # Top 2 primary hazards
                warning_blocks.append(self.format_safety_warning(risk))
            if warning_blocks:
                injected_sections.append("\n\n---\n" + "\n\n".join(warning_blocks))

        # If LOTO required but not explicitly mentioned, inject LOTO reminder
        if loto_required and not has_loto_mention:
            loto_injection = (
                "\n\n🛑 **MANDATORY LOTO NOTICE (OSHA 1910.147)**: "
                "Prior to performing any mechanical or electrical maintenance, you MUST disconnect, "
                "lock out, and tag out all primary and secondary energy sources. Always verify zero energy "
                "state before removing protective covers."
            )
            injected_sections.append(loto_injection)

        # If high/critical risk, append mandatory PPE checklist
        if risk_level in {"HIGH", "CRITICAL"} and "PPE" not in response.upper():
            ppe_bullets = "\n".join(f"  - {item}" for item in ppe_list)
            ppe_section = f"\n\n🦺 **REQUIRED PERSONAL PROTECTIVE EQUIPMENT (PPE)**:\n{ppe_bullets}"
            injected_sections.append(ppe_section)

        final_response = response + "".join(injected_sections)

        return ValidatedResponse(
            response=final_response,
            safety_warnings=safety_warnings,
            ppe_required=ppe_list,
            risk_level=risk_level,
            loto_required=loto_required,
        )
