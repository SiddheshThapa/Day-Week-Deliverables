import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def estimate_cost(tokens: int, price_per_1k: float = 0.002) -> float:
    return (tokens / 1000) * price_per_1k

CONTEXT_WINDOWS = {
    "llama3-8b-8192"  : 8_192,
    "llama3-70b-8192" : 8_192,
    "gpt-4o"          : 128_000,
    "claude-3-5-sonnet": 200_000,
}

def check_fits(text: str, model: str) -> bool:
    tokens = count_tokens(text)
    limit  = CONTEXT_WINDOWS.get(model, 4096)
    fits   = tokens < limit
    print(f"Model     : {model}")
    print(f"Tokens    : {tokens}")
    print(f"Limit     : {limit}")
    print(f"Fits?     : {' Yes' if fits else ' Too long'}")
    return fits

if __name__ == "__main__":
    text = "Artificial Intelligence is transforming every industry in the world."
    print("=== Token Count ===")
    t = count_tokens(text)
    print(f"Text   : {text}")
    print(f"Tokens : {t}")
    print("\n=== Cost Estimate ===")
    cost = estimate_cost(t)
    print(f"Cost for {t} tokens: ${cost:.6f}")
    big_text = text * 100
    big_tokens = count_tokens(big_text)
    print(f"Big doc tokens: {big_tokens}")
    print(f"Big doc cost  : ${estimate_cost(big_tokens):.4f}")
    print("\n=== Context Window Check ===")
    check_fits(text, "llama3-8b-8192")