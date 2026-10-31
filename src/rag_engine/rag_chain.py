"""Main RAG execution chain for industrial maintenance and troubleshooting.

Coordinates hybrid retrieval, context synthesis, real-time sensor telemetry integration,
conversation history, safety guardrails, citation generation, and multimodal visual inspection.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import get_settings
from .document_processor import Document
from .retriever import HybridRetriever
from .vector_store import VectorStoreManager

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError:
    ChatGoogleGenerativeAI = None
    AIMessage = None
    BaseMessage = None
    HumanMessage = None
    SystemMessage = None

try:
    import google.generativeai as genai
    from PIL import Image
except ImportError:
    genai = None
    Image = None


@dataclass
class RAGResponse:
    """Structured response object returned by the RAG chain.

    Attributes:
        answer: Generated maintenance troubleshooting guidance.
        sources: List of citation metadata dictionaries (source, page, section, relevance_score).
        confidence: Estimated confidence score between 0.0 and 1.0.
        metadata: Execution metadata (latency, query, model used, retrieval count, etc.).
    """

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGChain:
    """Production RAG pipeline for industrial predictive maintenance and troubleshooting.

    Grounds LLM responses in machine manuals, incorporates live sensor telemetry,
    enforces strict industrial safety protocols (LOTO, PPE), and supports
    visual diagnosis via Gemini multimodal vision.
    """

    SYSTEM_PROMPT = (
        "You are an expert AI industrial maintenance and troubleshooting assistant. "
        "You help technicians diagnose and fix machine problems safely. Always:\n"
        "1) Prioritize safety: Always emphasize Lockout/Tagout (LOTO), Personal Protective "
        "Equipment (PPE), mechanical energy dissipation, and electrical isolation before any physical work.\n"
        "2) Ground answers firmly in the provided maintenance manual context.\n"
        "3) Cite your sources specifically: Include the manual title and page number whenever "
        "referencing instructions (e.g., [Manual: <source>, Page: <page>, Section: <section>]).\n"
        "4) If unsure, or if the provided manual does not contain sufficient information, "
        "explicitly state so and recommend contacting an OEM field specialist or certified engineer.\n"
        "5) Provide clear, structured, step-by-step troubleshooting procedures with symptom analysis, "
        "root causes, necessary tools, and validation steps."
    )

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        vector_store: VectorStoreManager | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the RAG chain with retriever and Gemini LLM.

        Args:
            retriever: Instantiated HybridRetriever. If None, initialized from vector_store.
            vector_store: Optional VectorStoreManager to construct retriever if not supplied.
            model_name: Gemini model name. Defaults to settings.gemini_model.
            temperature: Sampling temperature. Defaults to settings.temperature (0.3).
            api_key: Gemini API key. Defaults to settings.gemini_api_key.
        """
        settings = get_settings()
        self.settings = settings
        self.model_name = model_name or settings.gemini_model
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self.api_key = api_key or settings.gemini_api_key

        # Initialize vector store and retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            vs = vector_store or VectorStoreManager()
            self.retriever = HybridRetriever(vector_store=vs)

        self._llm: Any = None
        self._is_api_key_valid = bool(
            self.api_key
            and self.api_key.strip()
            and self.api_key != "your_gemini_api_key_here"
        )

        if not self._is_api_key_valid:
            logger.warning(
                "Gemini API key is not configured or is placeholder in settings. "
                "Generative LLM synthesis will operate in fallback citation mode until configured."
            )
        else:
            self._init_llm()

    def _init_llm(self) -> None:
        """Instantiate the Gemini LLM client."""
        if ChatGoogleGenerativeAI is not None:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                    max_output_tokens=2048,
                )
                logger.info(
                    f"Initialized LangChain ChatGoogleGenerativeAI with model '{self.model_name}'."
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
                self._llm = None
        elif genai is not None:
            try:
                genai.configure(api_key=self.api_key)
                self._llm = genai.GenerativeModel(self.model_name)
                logger.info(f"Initialized google.generativeai with model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize google.generativeai: {e}")
                self._llm = None
        else:
            logger.warning(
                "Neither langchain-google-genai nor google-generativeai packages are installed."
            )

    def _call_llm_with_retry(
        self,
        messages: list[Any],
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> str:
        """Execute LLM invocation with exponential backoff retry.

        Args:
            messages: List of messages (LangChain format or prompt string).
            max_retries: Number of retry attempts for transient errors.
            initial_delay: Initial sleep duration in seconds.

        Returns:
            Generated response text.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        if not self._is_api_key_valid or self._llm is None:
            raise RuntimeError("Gemini API key is not configured or LLM client not initialized.")

        delay = initial_delay
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                if ChatGoogleGenerativeAI and isinstance(self._llm, ChatGoogleGenerativeAI):
                    res = self._llm.invoke(messages)
                    return str(res.content)
                elif genai and hasattr(self._llm, "generate_content"):
                    prompt_parts = []
                    for m in messages:
                        if hasattr(m, "content"):
                            prompt_parts.append(str(m.content))
                        else:
                            prompt_parts.append(str(m))
                    res = self._llm.generate_content(prompt_parts)
                    return res.text
                else:
                    raise RuntimeError("No compatible LLM client found.")
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini API call failed on attempt {attempt}/{max_retries}: {e}"
                )
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2.0

        raise RuntimeError(f"All {max_retries} Gemini API calls failed: {last_error}") from last_error

    def _calculate_confidence(
        self,
        retrieval_results: list[tuple[Document, float]],
        has_context: bool,
    ) -> float:
        """Estimate response confidence based on retrieval relevance scores.

        Args:
            retrieval_results: List of (Document, score) tuples.
            has_context: Whether relevant context documents were found.

        Returns:
            Float confidence score between 0.0 and 1.0.
        """
        if not has_context or not retrieval_results:
            return 0.35  # Baseline confidence for ungrounded general advice

        scores = [score for _, score in retrieval_results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Weighted blend: highest hit (60%) + average support (40%)
        blended = (max_score * 0.6) + (avg_score * 0.4)

        # Clamp between 0.40 and 0.95
        return round(float(min(0.95, max(0.40, blended))), 2)

    def query(
        self,
        user_question: str,
        sensor_context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> RAGResponse:
        """Query the RAG system to diagnose issues and generate troubleshooting steps.

        Args:
            user_question: Technician's problem description or query.
            sensor_context: Optional string containing live telemetry/anomalies
                (e.g., 'Bearing temperature 88°C, Vibration RMS 7.2 mm/s, ISO Fault Level C').
            conversation_history: Optional list of past chat turns formatted as:
                [{'role': 'user'|'assistant', 'content': '...'}].
            top_k: Number of chunks to retrieve. Defaults to settings.top_k_retrieval (5).
            alpha: Hybrid retrieval weight. Defaults to settings.hybrid_search_alpha (0.7).

        Returns:
            RAGResponse containing generated answer, cited sources, and confidence score.
        """
        start_time = time.perf_counter()
        effective_k = top_k or self.settings.top_k_retrieval
        effective_alpha = alpha if alpha is not None else self.settings.hybrid_search_alpha

        logger.info(f"RAG query received: '{user_question}' (top_k={effective_k}, alpha={effective_alpha})")

        # 1. Retrieve relevant manual sections
        retrieval_results = self.retriever.hybrid_search(
            query=user_question,
            top_k=effective_k,
            alpha=effective_alpha,
        )

        has_relevant_context = bool(retrieval_results and retrieval_results[0][1] >= 0.15)
        sources = self.retriever.format_sources(retrieval_results)

        # 2. Format Context Excerpts
        if has_relevant_context:
            context_str = self.retriever.format_retrieval_context(retrieval_results)
            context_instruction = (
                "Ground your answer in the provided technical manual excerpts below. "
                "Cite specific pages and sections in your answer."
            )
        else:
            context_str = (
                "NO RELEVANT MANUAL EXCERPTS FOUND IN DATABASE FOR THIS SPECIFIC MACHINE QUERY."
            )
            context_instruction = (
                "No specific technical manual sections were matched. You must begin your response with "
                "the following disclaimer banner:\n"
                "> [!NOTE]\n"
                "> **General Advisory**: No direct OEM manual match was found in the local repository. "
                "The following recommendations are based on standard industrial maintenance best practices.\n\n"
                "Then provide industry-standard safety and troubleshooting guidelines."
            )

        # 3. Format Real-time Sensor Telemetry (if supplied)
        sensor_block = ""
        if sensor_context and sensor_context.strip():
            sensor_block = (
                f"\n\n--- REAL-TIME SENSOR TELEMETRY & SYSTEM ALERTS ---\n"
                f"{sensor_context.strip()}\n"
                f"Note: Explicitly analyze how these live sensor readings relate to the symptoms."
            )

        # 4. Construct Prompt Messages
        user_prompt_content = (
            f"{context_instruction}\n\n"
            f"--- TECHNICAL MANUAL CONTEXT ---\n"
            f"{context_str}"
            f"{sensor_block}\n\n"
            f"--- TECHNICIAN QUERY ---\n"
            f"{user_question}"
        )

        confidence = self._calculate_confidence(retrieval_results, has_relevant_context)

        # 5. Handle Missing API Key Gracefully
        if not self._is_api_key_valid or self._llm is None:
            logger.warning("No active Gemini API key. Returning formatted retrieval context.")
            fallback_answer = (
                "> [!WARNING]\n"
                "> **Gemini API Key Required for Generative Synthesis**: "
                "Please configure `GEMINI_API_KEY` in your `.env` file to enable conversational responses.\n\n"
                "### Retrieved Maintenance Documentation:\n\n"
            )
            if has_relevant_context:
                fallback_answer += context_str
            else:
                fallback_answer += (
                    "No relevant manual documentation was found for this query in the vector database."
                )

            latency = time.perf_counter() - start_time
            return RAGResponse(
                answer=fallback_answer,
                sources=sources,
                confidence=confidence,
                metadata={
                    "latency_seconds": round(latency, 3),
                    "model": "retrieval_only_fallback",
                    "retrieved_chunks": len(retrieval_results),
                    "has_context": has_relevant_context,
                },
            )

        # 6. Build Messages List for Chat Model
        messages: list[Any] = []
        if SystemMessage and HumanMessage and AIMessage:
            messages.append(SystemMessage(content=self.SYSTEM_PROMPT))

            # Incorporate Conversation History
            if conversation_history:
                for turn in conversation_history[-self.settings.max_conversation_history :]:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role in ("assistant", "ai"):
                        messages.append(AIMessage(content=content))

            messages.append(HumanMessage(content=user_prompt_content))
        else:
            # Flat string fallback
            messages = [f"SYSTEM: {self.SYSTEM_PROMPT}\n\nUSER: {user_prompt_content}"]

        # 7. Call LLM with Retry
        try:
            answer_text = self._call_llm_with_retry(messages)
        except Exception as e:
            logger.error(f"Error during RAG generation: {e}")
            answer_text = (
                f"> [!CAUTION]\n"
                f"> **Inference Error**: An error occurred while communicating with the AI service: {e}\n\n"
                f"### Relevant Manual Excerpts Retrieved:\n\n{context_str}"
            )

        latency = time.perf_counter() - start_time
        return RAGResponse(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            metadata={
                "latency_seconds": round(latency, 3),
                "model": self.model_name,
                "retrieved_chunks": len(retrieval_results),
                "has_context": has_relevant_context,
            },
        )

    def query_with_image(
        self,
        user_question: str,
        image_path: str | Path,
        sensor_context: str | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """Diagnose machine faults by analyzing a physical component image along with manual context.

        Sends the image and troubleshooting context to Gemini Vision for multimodal analysis
        (e.g., assessing gear wear, thermal discoloration, bearing pitting, or leakage).

        Args:
            user_question: Problem description or diagnostic request.
            image_path: Local path to the component image file.
            sensor_context: Optional live sensor telemetry string.
            top_k: Number of manual chunks to retrieve.

        Returns:
            RAGResponse containing visual inspection findings and repair procedures.
        """
        start_time = time.perf_counter()
        img_path = Path(image_path).resolve()

        if not img_path.is_file():
            raise FileNotFoundError(f"Inspection image not found: {img_path}")

        logger.info(f"Multimodal visual inspection query with image: '{img_path.name}'")

        # 1. Retrieve relevant manual sections based on question
        retrieval_results = self.retriever.hybrid_search(
            query=user_question,
            top_k=top_k or self.settings.top_k_retrieval,
        )
        sources = self.retriever.format_sources(retrieval_results)
        context_str = self.retriever.format_retrieval_context(retrieval_results)

        sensor_block = ""
        if sensor_context and sensor_context.strip():
            sensor_block = (
                f"\n\n--- REAL-TIME SENSOR TELEMETRY ---\n"
                f"{sensor_context.strip()}\n"
            )

        prompt_text = (
            f"You are conducting a visual physical inspection of an industrial machine component.\n\n"
            f"--- RELEVANT TECHNICAL MANUAL PROCEDURES ---\n"
            f"{context_str}"
            f"{sensor_block}\n\n"
            f"--- INSPECTION REQUEST ---\n"
            f"{user_question}\n\n"
            f"Instructions:\n"
            f"1. Examine the image carefully for physical anomalies (wear patterns, cracks, misalignment, leakage, corrosion, overheating).\n"
            f"2. Cross-reference visible defects with the manual specs provided.\n"
            f"3. State immediate safety precautions (PPE, shutdown, LOTO).\n"
            f"4. Recommend definitive corrective actions."
        )

        confidence = self._calculate_confidence(retrieval_results, bool(retrieval_results))

        if not self._is_api_key_valid:
            latency = time.perf_counter() - start_time
            return RAGResponse(
                answer=(
                    "> [!WARNING]\n"
                    "> **Gemini API Key Required**: Please configure `GEMINI_API_KEY` to enable "
                    "multimodal visual diagnosis.\n\n"
                    f"### Retrieved Manual Guidance:\n\n{context_str}"
                ),
                sources=sources,
                confidence=confidence,
                metadata={
                    "latency_seconds": round(latency, 3),
                    "model": "retrieval_only_fallback",
                    "image": str(img_path),
                },
            )

        # Encode image
        mime_type = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
        image_bytes = img_path.read_bytes()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")

        answer_text = ""
        try:
            # Use LangChain multimodal message structure
            if ChatGoogleGenerativeAI and HumanMessage:
                vision_llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                )
                multimodal_message = HumanMessage(
                    content=[
                        {"type": "text", "text": f"{self.SYSTEM_PROMPT}\n\n{prompt_text}"},
                        {
                            "type": "image_url",
                            "image_url": f"data:{mime_type};base64,{base64_data}",
                        },
                    ]
                )
                res = vision_llm.invoke([multimodal_message])
                answer_text = str(res.content)
            elif genai and Image:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model_name)
                pil_image = Image.open(img_path)
                res = model.generate_content([self.SYSTEM_PROMPT, prompt_text, pil_image])
                answer_text = res.text
            else:
                answer_text = (
                    "> [!CAUTION]\n"
                    "> Multimodal dependencies not available. Please install langchain-google-genai or google-generativeai and Pillow."
                )
        except Exception as e:
            logger.error(f"Multimodal vision query failed: {e}")
            answer_text = (
                f"> [!CAUTION]\n"
                f"> **Visual Analysis Failed**: {e}\n\n"
                f"### Manual Excerpts for Question:\n\n{context_str}"
            )

        latency = time.perf_counter() - start_time
        return RAGResponse(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            metadata={
                "latency_seconds": round(latency, 3),
                "model": self.model_name,
                "image": str(img_path),
            },
        )
