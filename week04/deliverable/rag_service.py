"""
The orchestrator. This is the file that actually implements "RAG":
retrieve -> ground-check -> assemble -> generate -> attach citations.

Notice this file imports SearchEngine from Week 3 instead of reimplementing
retrieval — that's the entire point of building Week 3 as a clean package.
"""

import logging
import sys
import importlib.util
from pathlib import Path

# ── Load Week 3's ENTIRE day5 package under a renamed identity ──
# Both folders are literally named "day5" — a plain `sys.path` + import
# trick fails because Python can't tell the two "day5" packages apart.
# Loading just search_engine.py alone also fails, because that file uses
# RELATIVE imports (`.vector_store`, `.embeddings`, etc.) which only work
# if Python believes the file is part of a real, registered package.
#
# The fix: register Week 3's day5 folder itself as a package named
# "week3_day5" BEFORE importing search_engine from inside it. That way,
# search_engine.py's relative imports resolve against "week3_day5",
# completely separate from this file's own "day5" package.

_WEEK03_DAY5_DIR = Path(__file__).resolve().parents[2] / "week03" / "deliverable"
_WEEK03_INIT_PATH = _WEEK03_DAY5_DIR / "__init__.py"

if "week3_day5" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "week3_day5",
        _WEEK03_INIT_PATH,
        submodule_search_locations=[str(_WEEK03_DAY5_DIR)],  # makes it a real package, not a single file
    )
    _week3_day5_package = importlib.util.module_from_spec(_spec)
    sys.modules["week3_day5"] = _week3_day5_package
    _spec.loader.exec_module(_week3_day5_package)

# Now this works, because "week3_day5.search_engine" resolves its OWN
# internal relative imports against the "week3_day5" package correctly.
_search_engine_spec = importlib.util.spec_from_file_location(
    "week3_day5.search_engine", _WEEK03_DAY5_DIR / "search_engine.py"
)
_search_engine_module = importlib.util.module_from_spec(_search_engine_spec)
sys.modules["week3_day5.search_engine"] = _search_engine_module
_search_engine_spec.loader.exec_module(_search_engine_module)

SearchEngine = _search_engine_module.SearchEngine

from .config import generation_settings
from .llm_client import LLMClient
from .prompts import GROUNDED_QA_SYSTEM_PROMPT, build_context_block, build_user_prompt
from .schemas import Citation, RAGResponse

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.search_engine = SearchEngine()   # Week 3's retrieval, reused as-is
        self.llm = LLMClient()

    def answer(self, query: str, top_k: int = 3, topic: str | None = None) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        # ── STEP 1: RETRIEVE (delegated entirely to Week 3) ─────
        raw_results = self.search_engine.search(query, top_k=top_k, topic=topic)

        # ── STEP 2: GROUND-CHECK before spending an LLM call ────
        usable_chunks = [
            r for r in raw_results
            if r["similarity"] >= generation_settings.min_similarity_threshold
        ]

        if not usable_chunks:
            logger.info(f"No sufficiently similar context for query: '{query}'")
            return RAGResponse(
                answer="I don't have enough information to answer that based on the available documents.",
                citations=[],
                grounded=False,
                query=query,
            )

        # ── STEP 3: ASSEMBLE PROMPT ──────────────────────────────
        context_block = build_context_block(usable_chunks)
        user_prompt = build_user_prompt(query, context_block)

        # ── STEP 4: GENERATE ─────────────────────────────────────
        answer_text = self.llm.generate(GROUNDED_QA_SYSTEM_PROMPT, user_prompt)

        # ── STEP 5: ATTACH STRUCTURED CITATIONS ──────────────────
        citations = [
            Citation(
                index=i + 1,
                text=chunk["text"],
                similarity=chunk["similarity"],
                topic=chunk["metadata"].get("topic"),
            )
            for i, chunk in enumerate(usable_chunks)
        ]

        return RAGResponse(
            answer=answer_text,
            citations=citations,
            grounded=True,
            query=query,
        )

    def stats(self) -> dict:
        return self.search_engine.stats()