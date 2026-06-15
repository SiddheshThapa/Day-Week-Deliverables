
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key  = os.getenv("API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)
MODEL = "llama-3.3-70b-versatile"

def ask(prompt: str, system: str = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model      = MODEL,
        messages   = messages,
        max_tokens = 200
    )
    return response.choices[0].message.content


def zero_shot():
    prompt = "Classify this as positive or negative: 'I love this product!'"
    print(ask(prompt))


def few_shot():
    prompt = """
Classify as positive or negative:

"I love this!"     → positive
"This is terrible" → negative
"Worst purchase"   → negative

"Absolutely amazing product!" →
"""
    print(ask(prompt))


def chain_of_thought():
    prompt = """
A store sells apples for $2 each.
John buys 5 apples and pays with $20.
How much change does he get?

Think step by step.
"""
    print(ask(prompt))


def role_prompt():
    system = "You are a senior AI engineer. Give short, technical answers only."
    prompt = "What is RAG?"
    print(ask(prompt, system=system))

if __name__ == "__main__":

    print("=== Zero Shot ===")
    zero_shot()

    print("\n=== Few Shot ===")
    few_shot()

    print("\n=== Chain of Thought ===")
    chain_of_thought()

    print("\n=== Role Prompt ===")
    role_prompt()