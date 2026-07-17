from dataclasses import dataclass


@dataclass
class Citation:
    index: int
    text: str
    similarity: float
    topic: str | None = None


@dataclass
class EvalScores:
    faithfulness: float
    context_precision: float


@dataclass
class ConversationTurn:
    """Represents one complete Q&A turn in a conversation."""
    role: str       # "user" or "assistant"
    content: str


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    grounded: bool
    query: str
    retrieval_method: str = "hybrid"
    eval_scores: EvalScores | None = None
    conversation_turn: int = 0  # which turn in the conversation this is