"""
Raw data, completely separate from logic.
Production rule: data and code live in different files. This lets
non-engineers (or a future ingestion pipeline from a real database/CMS)
update content WITHOUT touching the search engine's code at all.

In a real product, this file would instead be a database query, an API
call to a CMS, or a folder of files being scanned — this hardcoded list
is just a stand-in for that data source so the example stays runnable.
"""

from typing import TypedDict


class DocumentRecord(TypedDict):
    id: str
    text: str
    metadata: dict


KNOWLEDGE_BASE: list[DocumentRecord] = [
    {
        "id": "doc_001",
        "text": "Python is a versatile language used in AI, web development, and data science.",
        "metadata": {"topic": "programming", "difficulty": "beginner"},
    },
    {
        "id": "doc_002",
        "text": "LangChain helps developers build LLM applications quickly using composable chains.",
        "metadata": {"topic": "ai_tools", "difficulty": "intermediate"},
    },
    {
        "id": "doc_003",
        "text": "FAISS is a library for fast vector similarity search developed by Meta.",
        "metadata": {"topic": "ai_tools", "difficulty": "intermediate"},
    },
    {
        "id": "doc_004",
        "text": "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "metadata": {"topic": "geography", "difficulty": "beginner"},
    },
    {
        "id": "doc_005",
        "text": "Mount Everest is the tallest mountain in the world at 8,849 meters.",
        "metadata": {"topic": "geography", "difficulty": "beginner"},
    },
    {
        "id": "doc_006",
        "text": "Neural networks are inspired by the structure and function of the human brain.",
        "metadata": {"topic": "ai_concepts", "difficulty": "intermediate"},
    },
    {
        "id": "doc_007",
        "text": "RAG retrieves relevant documents before generating an answer, improving factual accuracy.",
        "metadata": {"topic": "ai_concepts", "difficulty": "advanced"},
    },
]