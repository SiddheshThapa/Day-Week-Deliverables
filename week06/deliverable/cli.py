"""
Week 6 CLI — streaming output, conversation history display.
Commands:
  ask <question>        → standard answer (non-streaming)
  chat <question>       → streaming answer (typing effect)
  multi <question>      → multi-query retrieval
  hyde <question>       → HyDE retrieval
  eval <question>       → answer + evaluation scores
  history                → show conversation history
  clear                  → clear conversation memory
  stats                  → document count + session info
  quit                   → exit
"""

import logging
from .rag_service import RAGService
from .schemas import RAGResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SESSION_ID = "main_session"


def print_response(response: RAGResponse, skip_answer: bool = False) -> None:
    if not skip_answer:
        print(f"\n [{response.retrieval_method}] {response.answer}")

    if response.grounded:
        print("\n Sources:")
        for c in response.citations:
            print(f"  [{c.index}] ({c.topic}, score={c.similarity}) {c.text}")

    if response.eval_scores:
        print(f"\n Evaluation (Turn {response.conversation_turn}):")
        print(f"  Faithfulness     : {response.eval_scores.faithfulness:.2f}")
        print(f"  Context Precision: {response.eval_scores.context_precision:.2f}")

    print(f"\n  [Turn {response.conversation_turn} in this session]")


def main():
    print("=" * 65)
    print("  Week 6 — Conversational RAG (LCEL + Memory + Streaming)")
    print("  ask | chat | multi | hyde | eval | history | clear | stats | quit")
    print("=" * 65)

    service = RAGService()
    print(f"\nStore ready — {service.stats()['total_documents']} documents loaded.\n")

    while True:
        try:
            command = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not command:
            continue

        if command.lower() == "quit":
            print("Bye!")
            break

        if command.lower() == "stats":
            print(f"  {service.stats()}")
            continue

        if command.lower() == "clear":
            service.clear_memory(SESSION_ID)
            print("  Conversation memory cleared.")
            continue

        if command.lower() == "history":
            history = service.memory.get_history(SESSION_ID)
            if not history:
                print("  No conversation history yet.")
            else:
                print(f"\n  Conversation history ({len(history)//2} turns):")
                for msg in history:
                    role = "You" if msg["role"] == "user" else "Bot"
                    print(f"  [{role}] {msg['content'][:80]}...")
            continue

        if command.startswith("chat "):
            query = command[5:]
        elif command.startswith("multi "):
            query = command[6:]
        elif command.startswith("hyde "):
            query = command[5:]
        elif command.startswith("eval "):
            query = command[5:]
        elif command.startswith("ask "):
            query = command[4:]
        else:
            print("  Unknown command. Use: ask | chat | multi | hyde | eval | history | clear | stats | quit")
            continue

        try:
            response = service.answer(
                query,
                session_id=SESSION_ID,
                use_multi_query=command.startswith("multi "),
                use_hyde=command.startswith("hyde "),
                run_eval=command.startswith("eval "),
                stream=command.startswith("chat "),
            )
            print_response(response, skip_answer=command.startswith("chat "))
        except Exception as exc:
            logger.error(f"Failed: {exc}")
            print(f"\n  Something went wrong: {exc}")


if __name__ == "__main__":
    main()