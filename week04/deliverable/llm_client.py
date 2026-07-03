"""
Thin wrapper around Gemini's chat completion API with automatic retry.
"""

import logging
import time
from google import genai
from google.genai import types

from .config import generation_settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=generation_settings.gemini_api_key)


class LLMClient:
    def __init__(self):
        self.model = generation_settings.chat_model

    def generate(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        last_error = None

        for attempt in range(max_retries):
            try:
                response = _client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=generation_settings.temperature,
                        max_output_tokens=generation_settings.max_output_tokens,
                    ),
                )
                return response.text

            except Exception as exc:
                last_error = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    wait = 2 ** attempt   # attempt 0=1s, attempt 1=2s, attempt 2=4s
                    logger.warning(f"503 on attempt {attempt + 1}/{max_retries}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"Generation failed (non-retryable): {exc}")
                    return "I'm having trouble generating a response right now. Please try again."

        logger.error(f"Generation failed after {max_retries} attempts: {last_error}")
        return "I'm having trouble generating a response right now. Please try again."