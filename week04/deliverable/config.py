"""
Generation-specific settings. Retrieval settings (embedding model, Chroma
path) intentionally stay OUT of this file — they belong to Week 3's
config.py. This file only owns what's NEW for the generation layer.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class GenerationSettings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # gemini-2.5-flash-lite: cheapest + highest free-tier limits, good fit
    # for grounded Q&A which doesn't need heavy multi-step reasoning
    chat_model: str = "gemini-2.5-flash"

    temperature: float = 0.2          # low = stay factual, don't get creative with facts
    max_output_tokens: int = 500

    # Safety net: if retrieval returns nothing usable, don't even call the LLM
    min_similarity_threshold: float = 0.3

    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")


generation_settings = GenerationSettings()
generation_settings.validate()