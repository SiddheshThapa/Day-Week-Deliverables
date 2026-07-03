"""
Prompt templates live here, separate from orchestration logic.
Production rule: prompts get iterated on CONSTANTLY during development —
keeping them in one file means you tune wording without touching business
logic, and you can unit test prompt formatting independently.
"""

GROUNDED_QA_SYSTEM_PROMPT = """You are a precise, factual assistant.
Rules you must follow:
1. Answer ONLY using the provided context below.
2. Cite every claim with its source number, like [1] or [1][2].
3. If the context does not contain the answer, respond exactly with:
   "I don't have enough information to answer that based on the available documents."
4. Never speculate or use outside knowledge, even if you know the answer."""


def build_context_block(chunks: list[dict]) -> str:
    """
    Turns retrieved chunks into the numbered context block the LLM sees.
    Kept as a standalone function so it's independently testable —
    you can verify formatting WITHOUT calling any API.
    """
    lines = [f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(chunks)]
    return "\n".join(lines)


def build_user_prompt(query: str, context_block: str) -> str:
    return f"""Context:
{context_block}

Question: {query}"""