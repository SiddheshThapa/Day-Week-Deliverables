import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_THIS_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Settings:
    # ── API ──────────────────────────────────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # ── Embedding ────────────────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    # ── Chroma ───────────────────────────────────────────────────
    chroma_persist_dir: str = str(_THIS_DIR / "chroma_db")
    collection_name: str = "week06_documents"

    # ── Retrieval ────────────────────────────────────────────────
    default_top_k: int = 3
    rerank_top_n: int = 5
    bm25_weight: float = 0.4
    dense_weight: float = 0.6

    # ── Generation ───────────────────────────────────────────────
    chat_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.2
    max_output_tokens: int = 500
    min_similarity_threshold: float = 0.3

    # ── Evaluation ───────────────────────────────────────────────
    faithfulness_threshold: float = 0.7
    precision_threshold: float = 0.6

    # ── Caching ──────────────────────────────────────────────────
    cache_max_size: int = 100

    # ── Guardrails ───────────────────────────────────────────────
    guardrail_min_confidence: float = 0.4

    # ── RAGAS ────────────────────────────────────────────────────
    ragas_llm_model: str = "gemini-2.5-flash"

    # ── Memory (for when we return to Week 6) ───────────────────
    max_history_turns: int = 10

    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")


settings = Settings()
settings.validate()