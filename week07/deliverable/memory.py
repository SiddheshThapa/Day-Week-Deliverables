import logging
from .config import settings
from .schemas import ConversationTurn

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Stores conversation history per session.
    Each session_id maps to its own independent list of turns.
    """

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}      #each session gets its own lists of conversation



    def get_history(self, session_id: str) -> list[dict]:           #returns msg list for that session or else empty list
        return self._sessions.get(session_id, [])




    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:         #saves one Q&A exchange to the session; if history exceeds cap 10 turns , keeps only the most recent ones.
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({
            "role": "user",
            "content": user_message
        })
        self._sessions[session_id].append({
            "role": "assistant",
            "content": assistant_message
        })

        max_messages = settings.max_history_turns * 2
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]
            logger.info(f"Session '{session_id}': trimmed history to {settings.max_history_turns} turns")



    def turn_count(self, session_id: str) -> int:  # counts each turn = user+assistant pair
        return len(self._sessions.get(session_id, [])) // 2



    def clear_session(self, session_id: str) -> None:  # wipes one session's history
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session '{session_id}' cleared")



    def list_sessions(self) -> list[str]:               #returns all active session IDs
        return list(self._sessions.keys())