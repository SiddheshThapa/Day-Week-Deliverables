"""
Wraps the Gemini embedding API behind a clean interface.
WHY a wrapper at all: if Google changes their SDK tomorrow, you fix ONE
file instead of hunting through every place embeddings are called.
"""

import logging
from typing import Any
from google import genai
from google.genai import types
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from .config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)


@register_embedding_function
class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    A Chroma-compatible embedding function.
    Newer Chroma versions require custom embedding functions to implement
    name(), get_config(), and build_from_config() — these are used internally
    for validation and for persisting WHICH embedding function a collection
    was created with, so Chroma can detect mismatches on reload.
    """

    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT"):
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []

        try:
            response = _client.models.embed_content(
                model=settings.embedding_model,
                contents=input,
                config=types.EmbedContentConfig(
                    task_type=self.task_type,
                    output_dimensionality=settings.embedding_dimensions,
                ),
            )
            return [e.values for e in response.embeddings]

        except Exception as exc:
            logger.error(f"Embedding failed for {len(input)} text(s): {exc}")
            raise

    @staticmethod
    def name() -> str:
        # Required by Chroma — a stable identifier for this embedding function type
        return "gemini-embedding-function"

    def get_config(self) -> dict[str, Any]:
        # Required by Chroma — lets it persist/restore this function's settings
        return {"task_type": self.task_type}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "GeminiEmbeddingFunction":
        # Required by Chroma — reconstructs this function from saved config
        # when you reopen a persistent collection in a NEW script run
        return GeminiEmbeddingFunction(task_type=config["task_type"])


# ── Two pre-configured instances, one per use case ────────────
document_embedder = GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
query_embedder = GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")