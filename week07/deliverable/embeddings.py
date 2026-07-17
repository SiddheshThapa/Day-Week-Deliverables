import logging
from typing import Any
from google import genai
from google.genai import types
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from .config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

class GeminiEmbeddingFunction(EmbeddingFunction):
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
            logger.error(f"Embedding failed: {exc}")
            raise

    @staticmethod
    def name() -> str:
        return "gemini-embedding-function-w6"

    def get_config(self) -> dict[str, Any]:
        return {"task_type": self.task_type}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(task_type=config["task_type"])


document_embedder = GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")
query_embedder = GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")

