import logging
import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .embeddings import document_embedder, query_embedder
from .knowledge import DOCUMENTS, DocumentRecord

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=document_embedder,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"VectorStore ready — {self._collection.count()} docs in '{settings.collection_name}'")

    def ingest_if_empty(self) -> int:
        if self._collection.count() > 0:
            logger.info("Collection not empty — skipping ingestion")
            return 0
        logger.info(f"Ingesting {len(DOCUMENTS)} documents...")
        ids = [d["id"] for d in DOCUMENTS]
        texts = [d["text"] for d in DOCUMENTS]
        metas = [d["metadata"] for d in DOCUMENTS]
        self._collection.upsert(ids=ids, documents=texts, metadatas=metas)
        logger.info(f"Ingested {len(DOCUMENTS)} documents successfully")
        return len(DOCUMENTS)

    def search(self, query: str, top_k: int | None = None, topic_filter: str | None = None) -> list[dict]:
        k = top_k or settings.default_top_k
        query_vector = query_embedder([query])[0]
        kwargs = {"query_embeddings": [query_vector], "n_results": k}
        if topic_filter:
            kwargs["where"] = {"topic": topic_filter}
        try:
            raw = self._collection.query(**kwargs)
        except Exception as exc:
            logger.error(f"Search failed: {exc}")
            raise
        if not raw["documents"] or not raw["documents"][0]:
            return []
        results = []
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": round(1 - dist, 4),
            })
        return results

    def get_all_documents(self) -> list[dict]:
        raw = self._collection.get()
        if not raw["documents"]:
            return []
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(raw["documents"], raw["metadatas"])
        ]

    def count(self) -> int:
        return self._collection.count()