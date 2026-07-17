import logging
from langchain_core.messages import HumanMessage, AIMessage

from .config import settings
from .vector_store import VectorStore
from .hybrid_retriever import HybridRetriever
from .query_transformer import generate_multi_queries, generate_hyde_query
from .evaluator import evaluate
from .memory import ConversationMemory
from .cache import QueryCache
from .lcel_chain import build_lcel_chain
from .llm_client import LLMClient
from .prompts import GROUNDED_QA_SYSTEM_PROMPT, build_context_block, build_user_prompt
from .schemas import Citation, RAGResponse, EvalScores

logger = logging.getLogger(__name__)


class RAGService:                   #boots up everything
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.ingest_if_empty()
        self.retriever = HybridRetriever(self.vector_store)

        self.memory = ConversationMemory()
        self.cache = QueryCache()

        self.chain = build_lcel_chain(self.retriever)

        self.llm = LLMClient()



    def answer(                             #session id / cache / stream
            self,
            query: str,
            session_id: str = "default",
            use_multi_query: bool = False,
            use_hyde: bool = False,
            run_eval: bool = False,
            stream: bool = False,
    ) -> RAGResponse:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        query = query.strip()

        cached_answer = self.cache.get(query)
        if cached_answer:
            return RAGResponse(
                answer=cached_answer,
                citations=[],
                grounded=True,
                query=query,
                retrieval_method="cache",
                conversation_turn=self.memory.turn_count(session_id),
            )

        search_query = query
        if use_hyde:
            search_query = generate_hyde_query(query)

        if use_multi_query:
            queries = generate_multi_queries(query)
            all_results: dict[str, dict] = {}
            for q in queries:
                for r in self.retriever.search(q):
                    all_results[r["text"]] = r
            retrieved = list(all_results.values())[:settings.default_top_k]
        else:
            retrieved = self.retriever.search(search_query)

        usable = [r for r in retrieved if r.get("rerank_score", r.get("hybrid_score", 0)) > 0]

        if not usable:
            return RAGResponse(
                answer="I don't have enough information to answer that based on the available documents.",
                citations=[],
                grounded=False,
                query=query,
                retrieval_method="hybrid+multiquery" if use_multi_query else "hybrid",
                conversation_turn=self.memory.turn_count(session_id),
            )

        history = self.memory.get_history(session_id)

        lc_history = []                     #prompt template needs langchain format not plain dict
        for msg in history:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            else:
                lc_history.append(AIMessage(content=msg["content"]))

        context_block = build_context_block(usable)

        chain_input = {
            "question": query,
            "history": lc_history,
            "context": context_block,
        }

        if stream:                      #streaming is for the chat command, invoke is for ask/multi/hyde/eval commands
            answer_text = ""
            print("🤖 ", end="", flush=True)
            for chunk in self.chain.stream(chain_input):
                print(chunk, end="", flush=True)
                answer_text += chunk
            print()
        else:
            answer_text = self.chain.invoke(chain_input)



        self.memory.add_turn(session_id, query, answer_text)            #saves this Q&A exchange into the conversation history
        self.cache.set(query, answer_text)                              #saves this Q&A exchange into the cache

        citations = [
            Citation(
                index=i + 1,
                text=c["text"],
                similarity=round(c.get("rerank_score", c.get("hybrid_score", 0.0)), 3),
                topic=c["metadata"].get("topic"),
            )
            for i, c in enumerate(usable)
        ]

        eval_scores = None
        if run_eval:
            eval_scores = evaluate(query, answer_text, usable)

        return RAGResponse(
            answer=answer_text,
            citations=citations,
            grounded=True,
            query=query,
            retrieval_method="hybrid+multiquery" if use_multi_query else "hybrid",
            eval_scores=eval_scores,
            conversation_turn=self.memory.turn_count(session_id),
        )

    def clear_memory(self, session_id: str = "default") -> None:
        self.memory.clear_session(session_id)

    def stats(self) -> dict:
        return {
            "total_documents": self.vector_store.count(),
            "active_sessions": len(self.memory.list_sessions()),
        }
