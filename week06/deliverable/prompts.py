#system prompt constant

GROUNDED_QA_SYSTEM_PROMPT = """You are a precise, factual assistant.
Rules you must follow:
1. Answer ONLY using the provided context below.
2. Cite every claim with its source number like [1] or [1][2].
3. If the context does not contain the answer, respond exactly with:
   "I don't have enough information to answer that based on the available documents."
4. Never speculate or use outside knowledge, even if you know the answer."""




def build_context_block(chunks: list[dict]) -> str:
    lines = [f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(chunks)]            #numbers each chunk as [1], [2], [3] so citations can refer them
    return "\n".join(lines)




def build_user_prompt(query: str, context_block: str) -> str:                   #glues the numbered context + question into one final prompt
    return f"""Context: 
{context_block}                                                                            
    
Question: {query}"""




def build_conversational_prompt(                                            
    query: str,
    context_block: str,
    history: list[dict],
) -> str:
    """
    Builds a prompt that includes conversation history, so the model
    can answer follow-up questions like "what did I just ask?"
    """
    history_text = ""
    if history:
        turns = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            turns.append(f"{role}: {turn['content']}")
        history_text = "\n\nConversation so far:\n" + "\n".join(turns)

    return f"""Context:
{context_block}{history_text}

Current Question: {query}"""




