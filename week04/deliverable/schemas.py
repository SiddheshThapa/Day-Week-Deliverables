"""
Typed structures for what flows through this service.
Production rule: don't pass raw dicts between layers — a typed shape
means your editor catches typos, and anyone reading this file instantly
understands the data contract without reading every function body.
"""

from dataclasses import dataclass


@dataclass
class Citation:
    index: int          # the [1], [2], [3] shown to the user
    text: str
    similarity: float
    topic: str | None = None


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    grounded: bool       # False if we answered without enough retrieved context
    query: str