import logging
import json
from google import genai
from google.genai import types

from .config import settings
from .schemas import EvalScores

logger = logging.getLogger(__name__)
_client = genai.Client(api_key=settings.gemini_api_key)






def _call_llm_json(prompt: str) -> dict:                #shared helper called by 2 functions below
    try:
        response = _client.models.generate_content(
            model=settings.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=500,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as exc:
        logger.error(f"Evaluator call failed: {exc}")
        return {}




def calculate_faithfulness(answer: str, context_chunks: list[dict]) -> float:               #Everything (all chunks combined, full answer) goes in one prompt, and Gemini itself does the claim-splitting internally and returns all claim judgments in one reply.
    context_text = "\n".join(c["text"] for c in context_chunks)
    prompt = f"""Break this answer into individual factual claims.
For each claim, check if it is supported by the context.
Return JSON only: {{"claims": [{{"claim": "...", "supported": true}}]}}

Context: {context_text}
Answer: {answer}"""
    data = _call_llm_json(prompt)
    claims = data.get("claims", [])
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c.get("supported", False))
    score = round(supported / len(claims), 3)
    logger.info(f"Faithfulness: {score}")
    return score




def calculate_context_precision(query: str, context_chunks: list[dict]) -> float:               # judging each chunk against the query.
    relevant = 0                                                                                # if 4 chunks then 4 separate calls to gemini
    for chunk in context_chunks:
        prompt = f"""Is this context chunk relevant to answering the question?
Return JSON only: {{"relevant": true}} or {{"relevant": false}}

Question: {query}
Context: {chunk['text']}"""
        data = _call_llm_json(prompt)
        if data.get("relevant", False):
            relevant += 1
    score = round(relevant / len(context_chunks), 3) if context_chunks else 0.0
    logger.info(f"Context Precision: {score}")
    return score





def evaluate(query: str, answer: str, context_chunks: list[dict]) -> EvalScores:
    return EvalScores(
        faithfulness=calculate_faithfulness(answer, context_chunks),
        context_precision=calculate_context_precision(query, context_chunks),
    )