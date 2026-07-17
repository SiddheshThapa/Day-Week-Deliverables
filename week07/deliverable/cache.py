import logging
from .config import settings

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Stores past query→answer pairs so identical questions don't
    trigger a full search + generation pipeline again.
    """

    def __init__(self):
        self._cache: dict[str, str] = {}

    def get(self, query: str) -> str | None:                #checks if this exact question was answered before
        normalized = query.strip().lower()
        cached_answer = self._cache.get(normalized)
        if cached_answer:
            logger.info(f"Cache HIT for: '{query}'")
        return cached_answer


    def set(self, query: str, answer: str) -> None:         #stores a new ques ans pair; if cache hits 100 size, deletes the oldest entry .
        normalized = query.strip().lower()

        if len(self._cache) >= settings.cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.info(f"Cache full — evicted oldest entry: '{oldest_key}'")

        self._cache[normalized] = answer
        logger.info(f"Cache SET for: '{query}'")


    def clear(self) -> None:                # wipes the entire cache
        self._cache.clear()
        logger.info("Cache cleared")