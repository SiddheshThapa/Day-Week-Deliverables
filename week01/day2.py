from dataclasses import dataclass

def greet(name: str) -> str:
    return f"Hello {name}"

@dataclass
class LLMResponse:
    model: str
    content: str
    tokens: int

words = ["hello", "AI", "world"]
lengths = [len(w) for w in words]

def make_chunks(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i:i+size]

def save_text(text: str):
    with open("week01/output.txt", "w") as f:
        f.write(text)

class APIKeyMissingError(Exception):
    pass

if __name__ == "__main__":
    print(greet("AI Engineer"))
    r = LLMResponse(model="llama3", content="AI is cool", tokens=10)
    print(r)
    print(lengths)
    for chunk in make_chunks("Hello World from AI", size=5):
        print(chunk)
    save_text("Day 2 done!")
    print("File saved!")