

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import httpx

load_dotenv()

client = OpenAI(
    api_key  = os.getenv("API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)
MODEL = "llama-3.3-70b-versatile"

class MovieReview(BaseModel):
    title     : str
    sentiment : str
    score     : int
    summary   : str

class PersonInfo(BaseModel):
    name  : str
    age   : int
    skills: list[str]

def extract_movie_review(text: str) -> MovieReview:
    prompt = f"""
Extract movie review info from this text.
Return ONLY a JSON object with keys: title, sentiment, score (1-10), summary.
Text: {text}
"""
    response = client.chat.completions.create(
        model           = MODEL,
        messages        = [{"role": "user", "content": prompt}],
        response_format = {"type": "json_object"},
        max_tokens      = 200
    )
    raw = response.choices[0].message.content
    return MovieReview(**json.loads(raw))

def extract_person(text: str) -> PersonInfo:
    prompt = f"""
Extract person info from this text.
Return ONLY a JSON object with keys: name, age, skills (list).
Text: {text}
"""
    response = client.chat.completions.create(
        model           = MODEL,
        messages        = [{"role": "user", "content": prompt}],
        response_format = {"type": "json_object"},
        max_tokens      = 200
    )
    raw = response.choices[0].message.content
    return PersonInfo(**json.loads(raw))



def get_weather(city: str) -> str:
    url      = f"https://wttr.in/{city}?format=3"
    response = httpx.get(url)
    return response.text                

def ask_weather(user_question: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name"       : "get_weather",
                "description": "Get current weather for a city",
                "parameters" : {
                    "type"      : "object",
                    "properties": {
                        "city": {"type": "string", "description": "city name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [{"role": "user", "content": user_question}]

    response = client.chat.completions.create(
        model    = MODEL,
        messages = messages,
        tools    = tools
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args      = json.loads(tool_call.function.arguments)
    city      = args["city"]
    print(f"LLM decided to call: get_weather(city='{city}')")

    weather_result = get_weather(city)
    print(f"Real weather data : {weather_result}")

    messages.append(response.choices[0].message)
    messages.append({
        "role"        : "tool",
        "tool_call_id": tool_call.id,
        "content"     : weather_result
    })

    final = client.chat.completions.create(
        model    = MODEL,
        messages = messages
    )
    print(f"LLM final answer  : {final.choices[0].message.content}")

if __name__ == "__main__":

    print("=== Movie Review Extraction ===")
    review = extract_movie_review(
        "Interstellar is a masterpiece! Amazing visuals and story. Definitely a 9/10."
    )
    print(f"Title    : {review.title}")
    print(f"Sentiment: {review.sentiment}")
    print(f"Score    : {review.score}")
    print(f"Summary  : {review.summary}")

    print("\n=== Person Extraction ===")
    person = extract_person(
        "My name is Arjun, I am 25 years old and I know Python, AI and LangChain."
    )
    print(f"Name  : {person.name}")
    print(f"Age   : {person.age}")
    print(f"Skills: {person.skills}")

    print("\n=== Real Function Calling ===")
    ask_weather("What is the weather like in Mumbai right now?")