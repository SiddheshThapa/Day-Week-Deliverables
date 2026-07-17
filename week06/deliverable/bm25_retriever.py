import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, documents: list[dict]):
        self._documents = documents
        tokenized = [doc["text"].lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        scored = sorted(
            zip(self._documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        results = []
        for doc, score in scored[:top_k]:
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "bm25_score": float(score),
            })
        return results