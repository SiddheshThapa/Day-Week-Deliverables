"""
Thin CLI front door — same principle as Week 3's cli.py.
All real logic lives in rag_service.py, fully reusable later by a
FastAPI endpoint (Week 10) without changing a single line here.
"""

import logging
from .rag_service import RAGService
from .schemas import RAGResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_response(response: RAGResponse) -> None:
    print(f"\n🤖 {response.answer}")

    if not response.grounded:
        return

    print("\n📚 Sources:")
    for c in response.citations:
        print(f"  [{c.index}] ({c.topic}, sim={c.similarity:.3f}) {c.text}")


def main():
    print("=" * 55)
    print("  Production RAG Service")
    print("  Type 'quit' to exit, 'stats' for store info")
    print("=" * 55)

    service = RAGService()

    if service.stats()["total_documents"] == 0:
        n = service.search_engine.ingest_knowledge_base()
        print(f"Ingested {n} documents.\n")
    else:
        print(f"Using existing store with {service.stats()['total_documents']} documents.\n")

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if query.lower() == "quit":
            print("Bye!")
            break

        if query.lower() == "stats":
            print(f"  {service.stats()}")
            continue

        if not query:
            continue

        try:
            response = service.answer(query)
            print_response(response)
        except Exception as exc:
            logger.error(f"Failed to answer query: {exc}")
            print(f"\n⚠️  Something went wrong: {exc}")


if __name__ == "__main__":
    main()