
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


client = OpenAI(
    api_key  = os.getenv("API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)

MODEL_1 = "llama-3.3-70b-versatile"
MODEL_2 = "llama-3.1-8b-instant"

def call_llm(model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model    = model,
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def call_with_temp(prompt: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model       = MODEL_1,
        messages    = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens  = 50
    )
    return response.choices[0].message.content

def stream_response(prompt: str):
    stream = client.chat.completions.create(
        model    = MODEL_1,
        messages = [{"role": "user", "content": prompt}],
        stream   = True
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
    print()

if __name__ == "__main__":

    prompt = "What is machine learning? Answer in 2 lines."

    print("=== Model 1 — LLaMA 70b ===")
    print(call_llm(MODEL_1, prompt))

    print("\n=== Model 2 — Gemma 9b ===")
    print(call_llm(MODEL_2, prompt))

    print("\n=== Temperature Sweep ===")
    for temp in [0.0, 0.5, 1.0]:
        print(f"\nTemp {temp}:")
        print(call_with_temp("Give me one creative word.", temp))

    print("\n=== Streaming ===")
    stream_response("Tell me a fun fact about AI.")