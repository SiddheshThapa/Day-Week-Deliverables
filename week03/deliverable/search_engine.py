"""
The public-facing service. This is what other code (a CLI, a FastAPI route
in Week 10, another module) actually imports and calls — it should NEVER
need to know Chroma or Gemini exist underneath.
"""

import logging
from .vector_store import VectorStore
from .knowledge_base import KNOWLEDGE_BASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self):
        self.store = VectorStore()

    def ingest_knowledge_base(self) -> int:
        """Loads the static knowledge base into the vector store."""
        return self.store.upsert_documents(KNOWLEDGE_BASE)

    def search(
        self, query: str, top_k: int = 3, topic: str | None = None
    ) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        return self.store.search(query=query.strip(), top_k=top_k, topic_filter=topic)

    def stats(self) -> dict:
        return {"total_documents": self.store.count()}