"""
Centralized configuration for the search engine.
Production rule: NEVER scatter magic strings/numbers across files.
Every tunable value lives here, in ONE place.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)  # frozen=True = values can't be changed after creation (prevents accidental mutation)
class Settings:
    # ── API ─────────────────────────────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # ── Embedding model ─────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768          # smaller = cheaper storage, still strong quality
    # NOTE: must stay IDENTICAL between ingestion and querying — mismatched
    # dimensions silently produce garbage similarity scores, not an error

    # ── Chroma persistence ───────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"  # data survives script restarts
    collection_name: str = "documents"

    # ── Retrieval defaults ───────────────────────────────────
    default_top_k: int = 3

    def validate(self) -> None:
        """Fail FAST and LOUD at startup, not silently mid-request."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )


settings = Settings()
settings.validate()  # runs the moment this module is imported anywhere