# WEEK 4 - DAY 2: Document Loading & Preprocessing
# Topics: PDF loading, cleaning, metadata attachment
# Note: no LLM used here, so nothing changes from the Groq version

from pypdf import PdfReader
import re

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text.strip()

def create_document(text: str, source: str, doc_type: str) -> dict:
    return {
        "text"     : clean_text(text),
        "metadata" : {
            "source": source,
            "type"  : doc_type
        }
    }

def load_documents(file_paths: list[str]) -> list[dict]:
    documents = []
    for path in file_paths:
        if path.endswith(".pdf"):
            text = load_pdf(path)
            doc_type = "pdf"
        else:
            text = load_text_file(path)
            doc_type = "text"
        documents.append(create_document(text, source=path, doc_type=doc_type))
    return documents

if __name__ == "__main__":

    sample_text = """TechCorp Company Overview


    Founded in 2021    by Riya Sharma.

    We   have 45 employees!!"""

    with open("sample.txt", "w") as f:
        f.write(sample_text)

    print("=== Raw Text ===")
    raw = load_text_file("sample.txt")
    print(repr(raw))

    print("\n=== Cleaned Text ===")
    cleaned = clean_text(raw)
    print(cleaned)

    print("\n=== Document with Metadata ===")
    doc = create_document(raw, source="sample.txt", doc_type="text")
    print(doc)

    print("\n=== Loading Multiple Documents ===")
    docs = load_documents(["sample.txt"])
    for d in docs:
        print(f"Source: {d['metadata']['source']} | Type: {d['metadata']['type']}")
        print(f"Text: {d['text'][:60]}...")