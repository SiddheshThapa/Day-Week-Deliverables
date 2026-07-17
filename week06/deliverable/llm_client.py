import logging
import time                                                 #retry backoff delays
from google import genai
from typing import Generator
from google.genai import types

from .config import settings

logger = logging.getLogger(__name__)
_client = genai.Client(api_key=settings.gemini_api_key)




class LLMClient:                                            #retries only on server-overload errors (503) with increasing wait time, never crashes
    def __init__(self):                                     #GENERATE FN FOR HANDING WHOLE ANSWER BACK
        self.model = settings.chat_model

    def generate(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        last_error = None
        for attempt in range(max_retries):
            try:
                response = _client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=settings.temperature,
                        max_output_tokens=settings.max_output_tokens,
                    ),
                )
                return response.text
            except Exception as exc:
                last_error = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    wait = 2 ** attempt
                    logger.warning(f"503 attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"Generation failed (non-retryable): {exc}")
                    return "I'm having trouble generating a response right now. Please try again."
        logger.error(f"Generation failed after {max_retries} attempts: {last_error}")
        return "I'm having trouble generating a response right now. Please try again."




    def stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:          #stream fn word by word ans generation
        """
        Streaming generation. Yields text chunks as they arrive from
        the API, instead of waiting for the full response.
        """
        try:
            for chunk in _client.models.generate_content_stream(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=settings.temperature,
                        max_output_tokens=settings.max_output_tokens,
                    ),
            ):
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.warning(f"Streaming failed: {exc} — falling back to standard generation")
            result = self.generate(system_prompt, user_prompt)
            yield result
