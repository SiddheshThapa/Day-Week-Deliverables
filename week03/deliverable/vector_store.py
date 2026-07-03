"""
The core engine: a persistent, production-grade wrapper around ChromaDB.
This is the file you'd actually unit-test and reuse across multiple projects.
"""

import logging
import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .embeddings import document_embedder, query_embedder
from .knowledge_base import DocumentRecord

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wraps a persistent Chroma collection. Persistent means data survives
    between script runs — written to disk, not lost when the process exits.
    """

    def __init__(self):
        # PersistentClient (not the plain Client we used in Week 3 Day 4)
        # writes to disk at `path` — this is the actual production difference.
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # get_or_create_collection: idempotent — safe to call every startup.
        # If the collection already exists (e.g. you restarted the app),
        # it reuses it instead of crashing or duplicating data.
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=document_embedder,
            metadata={"hnsw:space": "cosine"},  # explicit similarity metric
        )

        logger.info(
            f"VectorStore ready — collection '{settings.collection_name}' "
            f"has {self._collection.count()} documents"
        )

    def upsert_documents(self, documents: list[DocumentRecord]) -> int:
        """
        'Upsert' = insert if new, update if the ID already exists.
        This is the production-safe alternative to .add(), which would
        error or duplicate on re-running ingestion.
        """
        if not documents:
            return 0

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        try:
            self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            logger.info(f"Upserted {len(documents)} document(s)")
            return len(documents)

        except Exception as exc:
            logger.error(f"Upsert failed: {exc}")
            raise

    def search(
        self,
        query: str,
        top_k: int | None = None,
        topic_filter: str | None = None,
    ) -> list[dict]:
        """
        Returns a list of result dicts, each with text/metadata/distance.
        Keeping the return type a plain list[dict] (not raw Chroma output)
        means callers never need to know Chroma's internal response shape —
        that's an implementation detail this class hides on purpose.
        """
        k = top_k or settings.default_top_k

        # IMPORTANT: queries must use query_embedder, not document_embedder.
        # Chroma's .query() embeds the query text FOR us, using whichever
        # embedding_function the collection was created with (document_embedder).
        # To use the QUERY-optimized embedding instead, we embed manually here
        # and pass the vector directly via query_embeddings.
        query_vector = query_embedder([query])[0]

        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": k,
        }
        if topic_filter:
            query_kwargs["where"] = {"topic": topic_filter}

        try:
            raw = self._collection.query(**query_kwargs)
        except Exception as exc:
            logger.error(f"Search failed for query '{query}': {exc}")
            raise

        # Defensive check: an empty collection returns empty nested lists,
        # not an error — handle that gracefully instead of crashing on [0].
        if not raw["documents"] or not raw["documents"][0]:
            return []

        results = []
        for doc, meta, distance in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            results.append({
                "text": doc,
                "metadata": meta,
                "distance": distance,
                "similarity": 1 - distance,  # convert distance back to an intuitive 0-1 score
            })
        return results

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Wipes the collection. Dangerous — only ever call this explicitly, never by default."""
        self._client.delete_collection(settings.collection_name)
        logger.warning(f"Collection '{settings.collection_name}' was reset")