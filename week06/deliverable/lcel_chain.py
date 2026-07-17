import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from .config import settings
from .hybrid_retriever import HybridRetriever
from .prompts import build_context_block, GROUNDED_QA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_lcel_chain(retriever: HybridRetriever):               #langchain needs llm wrapper
    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
    )

    prompt = ChatPromptTemplate.from_messages([             #takes system rules then combines context, question and history
        ("system", GROUNDED_QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def pass_context(inputs: dict) -> str:               #langchain cant call retriever search alone needs a wrapper fn.
        return inputs["context"]

    context_runnable = RunnableLambda(pass_context)                       #RunnableLambda converts wrapper to runnable component


    parser = StrOutputParser()                                    # to convert geminis reply from langchain object to python string

    chain = (
            {
                "context": context_runnable,
                "question": RunnableLambda(lambda x: x["question"]),
                "history": RunnableLambda(lambda x: x.get("history", [])),
            }
            | prompt
            | llm
            | parser
    )

    return chain