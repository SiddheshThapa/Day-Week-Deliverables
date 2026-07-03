"""Makes day5 a proper Python package so other weeks can import from it."""

from .search_engine import SearchEngine

__all__ = ["SearchEngine"]