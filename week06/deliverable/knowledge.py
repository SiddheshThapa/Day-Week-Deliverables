from typing import TypedDict


class DocumentRecord(TypedDict):
    id: str
    text: str
    metadata: dict


DOCUMENTS: list[DocumentRecord] = [
    {
        "id": "doc_001",
        "text": "RAG (Retrieval-Augmented Generation) retrieves relevant documents before generating an answer, improving factual accuracy and reducing hallucination.",
        "metadata": {"topic": "ai_concepts", "difficulty": "intermediate"},
    },
    {
        "id": "doc_002",
        "text": "Neural networks are inspired by the structure and function of the human brain, using layers of nodes to learn patterns from data.",
        "metadata": {"topic": "ai_concepts", "difficulty": "beginner"},
    },
    {
        "id": "doc_003",
        "text": "Large Language Models (LLMs) are trained on massive text datasets and can generate, summarize, translate, and reason about text.",
        "metadata": {"topic": "ai_concepts", "difficulty": "beginner"},
    },
    {
        "id": "doc_004",
        "text": "Embeddings convert text into dense vectors of numbers that capture semantic meaning — similar texts produce similar vectors.",
        "metadata": {"topic": "ai_concepts", "difficulty": "intermediate"},
    },
    {
        "id": "doc_005",
        "text": "Hallucination in LLMs refers to generating confident but factually incorrect information not grounded in any source.",
        "metadata": {"topic": "ai_concepts", "difficulty": "intermediate"},
    },
    {
        "id": "doc_006",
        "text": "LangChain helps developers build LLM applications quickly using composable chains and pre-built integrations.",
        "metadata": {"topic": "ai_tools", "difficulty": "intermediate"},
    },
    {
        "id": "doc_007",
        "text": "LangGraph is a framework for building stateful, multi-step AI agents using a graph of nodes and edges.",
        "metadata": {"topic": "ai_tools", "difficulty": "advanced"},
    },
    {
        "id": "doc_008",
        "text": "FAISS (Facebook AI Similarity Search) is a library for fast vector similarity search over large datasets.",
        "metadata": {"topic": "ai_tools", "difficulty": "intermediate"},
    },
    {
        "id": "doc_009",
        "text": "ChromaDB is an open-source vector database that stores embeddings with metadata and supports filtered semantic search.",
        "metadata": {"topic": "ai_tools", "difficulty": "intermediate"},
    },
    {
        "id": "doc_010",
        "text": "BM25 is a keyword-based ranking algorithm that scores documents by term frequency and inverse document frequency.",
        "metadata": {"topic": "ai_tools", "difficulty": "intermediate"},
    },
    {
        "id": "doc_011",
        "text": "Python is a versatile programming language widely used in AI, data science, web development, and automation.",
        "metadata": {"topic": "programming", "difficulty": "beginner"},
    },
    {
        "id": "doc_012",
        "text": "FastAPI is a modern Python framework for building REST APIs quickly with automatic validation and documentation.",
        "metadata": {"topic": "programming", "difficulty": "intermediate"},
    },
    {
        "id": "doc_013",
        "text": "Pydantic is a Python library for data validation using type hints — it catches wrong data types before they cause bugs.",
        "metadata": {"topic": "programming", "difficulty": "beginner"},
    },
    {
        "id": "doc_014",
        "text": "Hybrid search combines BM25 keyword search with dense vector search, using score fusion to get the best of both methods.",
        "metadata": {"topic": "rag_techniques", "difficulty": "advanced"},
    },
    {
        "id": "doc_015",
        "text": "Cross-encoder reranking takes a shortlist of retrieved documents and scores each query-document pair jointly for more accurate relevance.",
        "metadata": {"topic": "rag_techniques", "difficulty": "advanced"},
    },
    {
        "id": "doc_016",
        "text": "HyDE (Hypothetical Document Embeddings) generates a fake answer to a query first, then uses that fake answer's embedding to search — improving retrieval quality.",
        "metadata": {"topic": "rag_techniques", "difficulty": "advanced"},
    },
    {
        "id": "doc_017",
        "text": "Multi-query retrieval rephrases a question multiple ways using an LLM, searches all versions, and merges results to improve recall.",
        "metadata": {"topic": "rag_techniques", "difficulty": "advanced"},
    },
    {
        "id": "doc_018",
        "text": "Chunking splits long documents into smaller pieces before embedding — sentence-based chunking preserves meaning better than fixed-size chunking.",
        "metadata": {"topic": "rag_techniques", "difficulty": "intermediate"},
    },
    {
        "id": "doc_019",
        "text": "Faithfulness measures whether an answer's claims are actually supported by the retrieved context — a score of 1.0 means no hallucination.",
        "metadata": {"topic": "evaluation", "difficulty": "advanced"},
    },
    {
        "id": "doc_020",
        "text": "Context precision measures what fraction of retrieved chunks were actually relevant to answering the question.",
        "metadata": {"topic": "evaluation", "difficulty": "advanced"},
    },
]