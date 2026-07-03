# WEEK 4 - DAY 3: Retrieval + Prompt Assembly (using Gemini)
# Topics: top-k retrieval, context assembly, grounded prompts

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = "gemini-2.5-flash-lite"

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag_docs")

documents = [
    "TechCorp was founded in 2021 by Riya Sharma in Pune, India.",
    "TechCorp's main product is FlowDesk, a CRM tool for small businesses.",
    "TechCorp has 45 employees as of 2024.",
    "FlowDesk integrates with Gmail, Slack, and WhatsApp.",
    "The CEO of TechCorp is Riya Sharma, and the CTO is Aman Verma.",
]

def ingest():
    ids = [f"doc{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)

def retrieve(query: str, k: int = 2) -> list[str]:
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]

def assemble_prompt(query: str, chunks: list[str]) -> str:
    context = "\n".join(f"- {c}" for c in chunks)
    return f"""
Answer the question using ONLY the context below.
If you cannot answer from context, say "I don't know."

Context:
{context}

Question: {query}
"""

def rag_answer(query: str, k: int = 2) -> str:
    chunks = retrieve(query, k)
    prompt = assemble_prompt(query, chunks)
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":

    ingest()
    print(f"Ingested {collection.count()} documents")

    query = "Who is the CTO of TechCorp?"

    print(f"\n=== Query ===\n{query}")

    print("\n=== Retrieved Chunks ===")
    chunks = retrieve(query)
    for c in chunks:
        print(f"  - {c}")

    print("\n=== Final RAG Answer ===")
    answer = rag_answer(query)
    print(answer)