"""
Thin entrypoint — a CLI is just ONE possible "front door" to the service.
Notice this file contains almost NO logic — that's intentional. All real
logic lives in search_engine.py / vector_store.py, fully reusable by
a future FastAPI app (Week 10) without touching this file at all.
"""

import logging
from .search_engine import SearchEngine

logger = logging.getLogger(__name__)


def print_results(results: list[dict]) -> None:
    if not results:
        print("  No results found.")
        return
    for r in results:
        print(f"  [{r['similarity']:.3f}] ({r['metadata']['topic']}) {r['text']}")


def main():
    print("=" * 55)
    print("  Production Semantic Search Engine")
    print("  Commands: search <query> | topic <query>:<topic> | stats | quit")
    print("=" * 55)

    engine = SearchEngine()

    # Only ingest if the store is empty — avoids re-embedding (and re-paying
    # for API calls) every single time you restart the app.
    if engine.stats()["total_documents"] == 0:
        n = engine.ingest_knowledge_base()
        print(f"Ingested {n} documents into persistent storage.\n")
    else:
        print(f"Loaded existing store with {engine.stats()['total_documents']} documents.\n")

    while True:
        try:
            command = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if command.lower() == "quit":
            print("Bye!")
            break

        elif command.lower() == "stats":
            print(f"  {engine.stats()}")

        elif command.startswith("topic "):
            try:
                rest = command[6:]
                query, topic = rest.split(":")
                results = engine.search(query.strip(), topic=topic.strip())
                print_results(results)
            except ValueError:
                print("  Format: topic <query>:<topic>")
            except Exception as exc:
                logger.error(f"Search error: {exc}")
                print(f"  Error: {exc}")

        elif command.startswith("search "):
            query = command[7:]
            try:
                results = engine.search(query)
                print_results(results)
            except Exception as exc:
                logger.error(f"Search error: {exc}")
                print(f"  Error: {exc}")

        else:
            print("  Unknown command.")


if __name__ == "__main__":
    main()