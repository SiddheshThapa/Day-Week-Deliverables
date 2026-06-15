from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

class LLMResponse(BaseModel):
    model: str
    content: str
    tokens: int

def demo_valid():
    r = LLMResponse(model="llama3", content="AI is great", tokens=42)
    print(f"Valid: {r}")
    print(f"Just content: {r.content}")

def demo_invalid():
    try:
        r = LLMResponse(model="llama3", content="AI", tokens="wrong")
    except ValidationError as e:
        print(f"Caught error: {e.error_count()} error(s)")
        print(f"Detail: {e.errors()[0]['msg']}")

class Message(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    model: str
    message: Message
    tokens: int

def demo_nested():
    r = ChatResponse(
        model="llama3",
        message={"role": "assistant", "content": "Hello!"},
        tokens=10
    )
    print(f"Nested: {r.message.content}")

def demo_env():
    key = os.getenv("GROQ_API_KEY")
    if key:
        print(f"Key loaded: {key[:8]}...")
    else:
        print("No key found — check .env")

def demo_pandas():
    data = {
        "model":   ["llama3", "mistral", "gemma"],
        "tokens":  [100, 200, 150],
        "cost":    [0.01, 0.02, 0.015]
    }
    df = pd.DataFrame(data)
    print(f"\n{df}")
    print(f"\nAvg tokens: {df['tokens'].mean()}")
    print(f"Cheapest:   {df.loc[df['cost'].idxmin(), 'model']}")

if __name__ == "__main__":
    print("=== Valid Model ===")
    demo_valid()
    print("\n=== Invalid Model ===")
    demo_invalid()
    print("\n=== Nested Model ===")
    demo_nested()
    print("\n=== ENV Keys ===")
    demo_env()
    print("\n=== Pandas ===")
    demo_pandas()