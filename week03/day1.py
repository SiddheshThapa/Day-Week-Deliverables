# WEEK 4 - DAY 1: RAG Architecture (using Gemini)
# Topics: the full pipeline, grounding, citations

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = "gemini-2.5-flash-lite"
# ── 1. WITHOUT RAG ─────────────────────────────────────────────
def ask_without_rag(question: str) -> str:
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# ── 2. WITH RAG ────────────────────────────────────────────────
def ask_with_rag(question: str, context: str) -> str:
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the given context."

Context: {context}

Question: {question}
"""
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── 3. WITH CITATION ──────────────────────────────────────────
def ask_with_citation(question: str, context: str) -> str:
    prompt = f"""
Answer the question using ONLY the context below.
After your answer, quote the exact sentence from context you used as [Source: "..."]

Context: {context}

Question: {question}
"""
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":

    private_doc = """
Our company, TechCorp, was founded in 2021 by Riya Sharma.
We have 45 employees and our main product is a CRM tool called FlowDesk.
Our headquarters is in Pune, India.
"""

    question = "Who founded TechCorp and when?"

    print("=== WITHOUT RAG ===")
    print(ask_without_rag(question))

    print("\n=== WITH RAG ===")
    print(ask_with_rag(question, private_doc))

    print("\n=== WITH RAG + CITATION ===")
    print(ask_with_citation(question, private_doc))