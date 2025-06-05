# singletons/session_manager.py
from typing import Dict, List, Optional
import time
import threading
from dataclasses import dataclass
from singletons.logger import get_logger

logger = get_logger()


@dataclass
class ConversationMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    metadata: Optional[dict] = None


class ConversationSession:
    def __init__(self, session_id: str, max_history: int = 10):
        self.session_id = session_id
        self.messages: List[ConversationMessage] = []
        self.max_history = max_history
        self.last_accessed = time.time()
        self.context_documents = []  # Store recent RAG context

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_accessed = time.time()

        # Keep only recent messages
        # user + assistant = 2 messages per exchange
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]

    def get_conversation_history(self) -> str:
        """Format conversation history for LLM context"""
        history = []
        for msg in self.messages[-10:]:  # Maintain context window of last 10 exchanges
            role_name = "Người dùng" if msg.role == "user" else "Trợ lý"
            history.append(f"{role_name}: {msg.content}")
        return "\n-----".join(history)

    def update_context_documents(self, documents):
        """Update the RAG context documents for this session"""
        self.context_documents = documents
        self.last_accessed = time.time()


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.lock = threading.RLock()
        self.cleanup_interval = 3600  # 1 hour
        self.session_timeout = 5 * 60   # 5 minutes

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationSession(session_id)
                logger.info(f"Created new conversation session: {session_id}")
            else:
                self.sessions[session_id].last_accessed = time.time()
                logger.info(f"Retrieved existing conversation session: {session_id}")
            return self.sessions[session_id]

    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if current_time - session.last_accessed > self.session_timeout
            ]
            for sid in expired_sessions:
                del self.sessions[sid]
                logger.info(f"Cleaned up expired session: {sid}")


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
