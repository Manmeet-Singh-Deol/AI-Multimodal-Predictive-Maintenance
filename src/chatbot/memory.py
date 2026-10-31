"""Conversation Memory and State Persistence Module for Maintenance Chatbot.

Provides in-memory message buffering, automated context compression / summarization,
SQLite session persistence, and seamless LangChain message format conversion.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseMessage = Any  # type: ignore[assignment,misc]

from config.settings import get_settings


@dataclass
class ChatMessage:
    """Represents a single conversational turn.

    Attributes:
        role: Sender role ('user', 'assistant', 'system').
        content: Text content of the message.
        timestamp: ISO 8601 formatted timestamp string.
        metadata: Optional dictionary with auxiliary context (e.g., machine_id, tool_calls).
    """

    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chat message to dictionary representation."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ConversationMemory:
    """Manages chat buffer history, persistence in SQLite, and context summarization."""

    def __init__(
        self,
        max_history: int | None = None,
        db_path: str | Path | None = None,
        session_id: str = "default_session",
    ) -> None:
        """Initialize ConversationMemory.

        Args:
            max_history: Maximum number of messages to retain before summarization.
                         Defaults to Settings.max_conversation_history.
            db_path: Path to SQLite database file. Defaults to data/chat_history.db.
            session_id: Active session identifier string.
        """
        settings = get_settings()
        self.max_history = (
            max_history if max_history is not None else settings.max_conversation_history
        )
        self.session_id = session_id

        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = (settings.raw_path.parent / "chat_history.db").resolve()

        self.messages: list[ChatMessage] = []
        self.summary_context: str = ""

        self._init_database()
        logger.info(
            "ConversationMemory initialized (max_history={}, db='{}', session='{}')",
            self.max_history,
            self.db_path,
            self.session_id,
        )

    def _init_database(self) -> None:
        """Create SQLite schema if it does not already exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversations_session 
                    ON conversations(session_id);
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize SQLite database at '{}': {}", self.db_path, exc)

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new message to the active conversation buffer.

        Automatically triggers summarization if the message count exceeds max_history.

        Args:
            role: 'user', 'assistant', or 'system'.
            content: Message string.
            metadata: Optional dictionary with tool records or telemetry references.
        """
        if not content:
            return

        msg = ChatMessage(
            role=role.strip().lower(),
            content=content.strip(),
            metadata=metadata or {},
        )
        self.messages.append(msg)
        logger.debug("Added message to memory: role='{}' (total: {})", role, len(self.messages))

        # Check if history exceeds buffer capacity
        if len(self.messages) > self.max_history:
            self.summarize_if_long()

    def get_history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Retrieve recent conversation history as dictionaries.

        Args:
            last_n: Optional count of most recent messages to retrieve.

        Returns:
            List of message dictionaries with 'role', 'content', 'timestamp', and 'metadata'.
        """
        target_msgs = self.messages[-last_n:] if last_n else self.messages
        return [msg.to_dict() for msg in target_msgs]

    def get_messages(self) -> list[ChatMessage]:
        """Return the raw list of ChatMessage dataclass instances."""
        return list(self.messages)

    def get_formatted_history(self, last_n: int | None = None) -> str:
        """Format the dialogue history into a string suitable for LLM prompting.

        Args:
            last_n: Optional limit on recent turns.

        Returns:
            Human-readable formatted dialogue transcript.
        """
        sections: list[str] = []

        if self.summary_context:
            sections.append(f"[PREVIOUS SESSION SUMMARY]:\n{self.summary_context}\n")

        target_msgs = self.messages[-last_n:] if last_n else self.messages
        for msg in target_msgs:
            role_label = "Technician" if msg.role == "user" else "Assistant"
            sections.append(f"{role_label}: {msg.content}")

        return "\n".join(sections)

    def to_langchain_messages(self) -> list[Any]:
        """Convert memory messages into LangChain BaseMessage objects.

        Returns:
            List of SystemMessage, HumanMessage, and AIMessage instances.
        """
        if not HAS_LANGCHAIN:
            logger.warning("langchain_core not installed; cannot convert to LangChain messages.")
            return []

        lc_messages: list[Any] = []

        if self.summary_context:
            lc_messages.append(
                SystemMessage(
                    content=f"Context from earlier in this troubleshooting session: {self.summary_context}"
                )
            )

        for msg in self.messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role in {"assistant", "ai"}:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))

        return lc_messages

    def clear(self) -> None:
        """Reset in-memory conversation buffer and running summary."""
        self.messages.clear()
        self.summary_context = ""
        logger.info("In-memory conversation cleared for session '{}'", self.session_id)

    def summarize_if_long(self) -> None:
        """Compress older messages into a summary when history exceeds max_history.

        Keeps the most recent (max_history // 2) messages intact for immediate context
        and condenses the earlier turns into summary_context.
        """
        if len(self.messages) <= self.max_history:
            return

        keep_count = max(4, self.max_history // 2)
        archive_msgs = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]

        # Extract topics, machines, and action summaries from archived messages
        archive_summaries: list[str] = []
        for msg in archive_msgs:
            role = "User" if msg.role == "user" else "Agent"
            snippet = msg.content[:160].replace("\n", " ")
            if len(msg.content) > 160:
                snippet += "..."
            archive_summaries.append(f"{role}: {snippet}")

        new_summary_chunk = "; ".join(archive_summaries)
        if self.summary_context:
            self.summary_context = f"{self.summary_context} | {new_summary_chunk}"
        else:
            self.summary_context = new_summary_chunk

        # Cap length of running summary
        if len(self.summary_context) > 1200:
            self.summary_context = "..." + self.summary_context[-1100:]

        logger.info(
            "Summarized {} older messages. Buffer reduced to {} messages.",
            len(archive_msgs),
            len(self.messages),
        )

    def save_to_db(self, session_id: str | None = None) -> None:
        """Persist current conversation messages to the SQLite database.

        Args:
            session_id: Target session identifier. Defaults to self.session_id.
        """
        sid = session_id or self.session_id
        if not self.messages:
            logger.debug("No messages to persist for session '{}'", sid)
            return

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                for msg in self.messages:
                    meta_str = json.dumps(msg.metadata) if msg.metadata else "{}"
                    cursor.execute(
                        """
                        INSERT INTO conversations (session_id, role, content, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (sid, msg.role, msg.content, msg.timestamp, meta_str),
                    )
                conn.commit()
                logger.info("Persisted {} messages to SQLite for session '{}'", len(self.messages), sid)
        except Exception as exc:
            logger.error("Failed to save conversation to SQLite database: {}", exc)

    def load_from_db(self, session_id: str | None = None) -> None:
        """Restore conversation history from SQLite for a given session.

        Args:
            session_id: Target session identifier. Defaults to self.session_id.
        """
        sid = session_id or self.session_id
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT role, content, timestamp, metadata
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY id ASC;
                    """,
                    (sid,),
                )
                rows = cursor.fetchall()

            self.messages.clear()
            for r in rows:
                meta = {}
                if r["metadata"]:
                    try:
                        meta = json.loads(r["metadata"])
                    except Exception:
                        meta = {}
                self.messages.append(
                    ChatMessage(
                        role=r["role"],
                        content=r["content"],
                        timestamp=r["timestamp"],
                        metadata=meta,
                    )
                )

            logger.info("Loaded {} messages from SQLite for session '{}'", len(self.messages), sid)
        except Exception as exc:
            logger.error("Failed to load conversation from SQLite database: {}", exc)
