from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import httpx
import asyncio
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: Message

class GroqResponse(BaseModel):
    model: str
    choices: list[Choice]

async def call_groq(user_prompt: str) -> GroqResponse:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GROQ_URL, headers=headers, json=body)
        response.raise_for_status()
        return GroqResponse(**response.json())

async def safe_call(prompt: str):
    try:
        result = await call_groq(prompt)
        answer = result.choices[0].message.content
        print(f"\n {answer}")
    except httpx.HTTPStatusError as e:
        print(f" API Error: {e.response.status_code} — check your API key")
    except ValidationError as e:
        print(f" Response format unexpected: {e.error_count()} error(s)")
    except Exception as e:
        print(f" Something went wrong: {e}")

async def main():
    print("=" * 40)
    print("  Week 1 CLI Tool — Powered by Groq")
    print("  Type 'quit' to exit")
    print("=" * 40)
    if not API_KEY:
        print(" API_KEY missing from .env")
        return
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "quit":
            print("Bye!")
            break
        if not prompt:
            print("Please type something!")
            continue
        await safe_call(prompt)

if __name__ == "__main__":
    asyncio.run(main())