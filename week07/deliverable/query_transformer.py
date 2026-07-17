import logging
from google import genai
from google.genai import types

from .config import settings

logger = logging.getLogger(__name__)
_client = genai.Client(api_key=settings.gemini_api_key)




def _call_llm(prompt: str) -> str:
    try:
        response = _client.models.generate_content(             #helper function called by both fn below to send prompt , return text , fails soft without crash
            model=settings.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=200,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"Query transformation failed: {exc}")
        return ""



def generate_multi_queries(query: str, n: int = 3) -> list[str]:
    prompt = f"""Generate {n} different phrasings of this question. 
Return ONLY the questions, one per line, no numbering, no extra text.
Question: {query}"""
    response = _call_llm(prompt)                                    #ask llm to reword question in 3 ways then search with all 4
    if not response:
        return [query]
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    return [query] + lines[:n]




def generate_hyde_query(query: str) -> str:
    prompt = f"""Write a short, confident, factual answer to this question.
Write as if you are certain, even if you're not sure.
Keep it under 3 sentences.      
Question: {query}"""
    response = _call_llm(prompt)                                #fake answer generation then search using this instead of question
    if not response:
        return query
    return response