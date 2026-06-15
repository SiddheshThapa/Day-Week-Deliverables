

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key  = os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)
MODEL = "llama-3.3-70b-versatile"

class Summary(BaseModel):
    summary  : str
    key_point: str

class Classification(BaseModel):
    category  : str
    confidence: str

class Extraction(BaseModel):
    people : list[str]
    places : list[str]
    topics : list[str]

def summarize(text: str) -> Summary:
    prompt = f"""
Summarize this text.
Return ONLY JSON with keys: summary, key_point.
Text: {text}
"""
    r = client.chat.completions.create(
        model           = MODEL,
        messages        = [{"role": "user", "content": prompt}],
        response_format = {"type": "json_object"},
        max_tokens      = 200
    )
    return Summary(**json.loads(r.choices[0].message.content))

def classify(text: str) -> Classification:
    prompt = f"""
Classify this text into one category: tech, sports, politics, entertainment.
Return ONLY JSON with keys: category, confidence (high/medium/low).
Text: {text}
"""
    r = client.chat.completions.create(
        model           = MODEL,
        messages        = [{"role": "user", "content": prompt}],
        response_format = {"type": "json_object"},
        max_tokens      = 100
    )
    return Classification(**json.loads(r.choices[0].message.content))

def extract(text: str) -> Extraction:
    prompt = f"""
Extract entities from this text.
Return ONLY JSON with keys: people (list), places (list), topics (list).
Text: {text}
"""
    r = client.chat.completions.create(
        model           = MODEL,
        messages        = [{"role": "user", "content": prompt}],
        response_format = {"type": "json_object"},
        max_tokens      = 200
    )
    return Extraction(**json.loads(r.choices[0].message.content))



def run_eval():
    print("\n=== Mini Eval ===")

    samples = [
        {
            "text"    : "Elon Musk launched a new Tesla model in California.",
            "expected": "tech"
        },
        {
            "text"    : "Virat Kohli scored a century in Mumbai yesterday.",
            "expected": "sports"
        },
        {
            "text"    : "The prime minister announced new AI policy in Delhi.",
            "expected": "politics"
        }
    ]

    passed = 0
    for i, sample in enumerate(samples):
        result = classify(sample["text"])
        ok     = result.category.lower() == sample["expected"]
        status = " PASS" if ok else " FAIL"
        if ok:
            passed += 1
        print(f"Sample {i+1}: {status} | got={result.category} expected={sample['expected']}")

    print(f"\nScore: {passed}/{len(samples)}")

def main():
    print("=" * 45)
    print("  Week 2 Text Utility — Powered by Groq")
    print("  Commands: summarize / classify / extract / eval / quit")
    print("=" * 45)

    while True:
        command = input("\nCommand: ").strip().lower()

        if command == "quit":
            print("Bye!")
            break

        elif command == "eval":
            run_eval()

        elif command in ["summarize", "classify", "extract"]:
            text = input("Enter text: ").strip()
            if not text:
                print("Please enter some text!")
                continue

            if command == "summarize":
                r = summarize(text)
                print(f"Summary  : {r.summary}")
                print(f"Key point: {r.key_point}")

            elif command == "classify":
                r = classify(text)
                print(f"Category  : {r.category}")
                print(f"Confidence: {r.confidence}")

            elif command == "extract":
                r = extract(text)
                print(f"People: {r.people}")
                print(f"Places: {r.places}")
                print(f"Topics: {r.topics}")
        else:
            print("Unknown command! Use: summarize / classify / extract / eval / quit")

if __name__ == "__main__":
    main()