# WEEK 4 - DAY 4: Context Strategies (using Gemini)
# Topics: stuffing, map-reduce, refine

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = "gemini-2.5-flash-lite"

def ask(prompt: str) -> str:
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

chunks = [
    "TechCorp was founded in 2021 by Riya Sharma.",
    "FlowDesk is TechCorp's CRM product, used by 500+ businesses.",
    "TechCorp raised $2M in seed funding in 2022 from local investors."
]

def stuffing(chunks: list[str], question: str) -> str:
    context = "\n".join(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return ask(prompt)

def map_reduce(chunks: list[str], question: str) -> str:
    summaries = []
    for chunk in chunks:
        summary = ask(f"Summarize this in one line relevant to: '{question}'\nText: {chunk}")
        summaries.append(summary)

    combined = "\n".join(summaries)
    final_prompt = f"Based on these summaries, answer: {question}\n\nSummaries:\n{combined}"
    return ask(final_prompt)

def refine(chunks: list[str], question: str) -> str:
    answer = ""
    for i, chunk in enumerate(chunks):
        if i == 0:
            prompt = f"Question: {question}\nContext: {chunk}\nAnswer:"
        else:
            prompt = f"""
Question: {question}
Current answer: {answer}
New context: {chunk}
Refine the answer using the new context if useful, otherwise keep it the same.
"""
        answer = ask(prompt)
    return answer

if __name__ == "__main__":

    question = "What funding has TechCorp raised and from whom?"

    print("=== STUFFING ===")
    print(stuffing(chunks, question))

    print("\n=== MAP-REDUCE ===")
    print(map_reduce(chunks, question))

    print("\n=== REFINE ===")
    print(refine(chunks, question))