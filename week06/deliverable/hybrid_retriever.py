import logging
import json
import numpy as np
from google import genai
from google.genai import types

from .config import settings
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)
_client = genai.Client(api_key=settings.gemini_api_key)


def _gemini_rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return candidates

    candidates_text = "\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(candidates)
    )
    prompt = f"""You are a relevance scoring system.
Score each document's relevance to the query on a scale of 0-10.
0 = completely irrelevant, 10 = perfectly answers the query.
Return ONLY a JSON object: {{"scores": [7, 2, 9, 1, 5]}}
The scores array must have exactly {len(candidates)} numbers.

Query: {query}
Documents:
{candidates_text}"""

    try:
        response = _client.models.generate_content(
            model=settings.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=500,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        try:
            raw_text = (response.text or "").strip()
        except Exception:
            raw_text = ""

        if not raw_text:
            logger.warning("Reranker returned empty response — using hybrid score order")
            for c in candidates:
                c["rerank_score"] = c.get("hybrid_score", 0.0)
            return candidates
        data = json.loads(raw_text)
        scores = data.get("scores", [])
        if len(scores) != len(candidates):
            logger.warning("Reranker score count mismatch — falling back to hybrid order")
            return candidates
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
    except Exception as exc:
        logger.warning(f"Reranking failed: {exc} — using hybrid score order")
        for c in candidates:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
        return candidates


class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        all_docs = vector_store.get_all_documents()                 #builds my bm25 index once at startup
        self._bm25 = BM25Retriever(all_docs)
        logger.info("HybridRetriever ready")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or settings.default_top_k
        candidate_n = settings.rerank_top_n

        bm25_results = self._bm25.search(query, top_k=candidate_n)
        dense_results = self._vector_store.search(query, top_k=candidate_n)             #gets bm25 and dense result top 5 result



        candidates: dict[str, dict] = {}
        for r in dense_results:
            candidates[r["text"]] = {
                "text": r["text"],
                "metadata": r["metadata"],                                              #merging of both results into 1 no duplicates
                "dense_score": r["similarity"],
                "bm25_score": 0.0,
            }
        for r in bm25_results:
            if r["text"] in candidates:
                candidates[r["text"]]["bm25_score"] = r["bm25_score"]
            else:
                candidates[r["text"]] = {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "dense_score": 0.0,
                    "bm25_score": r["bm25_score"],
                }
        candidate_list = list(candidates.values())




        dense_scores = np.array([c["dense_score"] for c in candidate_list])
        bm25_scores = np.array([c["bm25_score"] for c in candidate_list])                                 #normalizing both scores to 0-1 then mixing (60% meaning  and  40% keyword)
        dense_norm = dense_scores / (dense_scores.max() + 1e-9)
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        combined = (settings.dense_weight * dense_norm) + (settings.bm25_weight * bm25_norm)

        for i, c in enumerate(candidate_list):
            c["hybrid_score"] = float(combined[i])

        candidate_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        shortlist = candidate_list[:candidate_n]
        for c in shortlist:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
        reranked = shortlist

        logger.info(f"Hybrid search: {min(k, len(reranked))} results for '{query}'")
        return reranked[:k]


