# =========================
# Workshop-ready RAG example
# =========================

import os
import sys
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------
# Environment setup
# -------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence HuggingFace warnings
load_dotenv()

# Add current folder to Python path so local packages can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MCP data
from task_2_mcp_and_agents.task2_solution import mcp_embeddings, mcp_names, mcps

# -------------------------
# OpenAI client (chat only)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HAS_KEY = bool(OPENAI_API_KEY) and OPENAI_API_KEY != "TODO"

if HAS_KEY:
    print("✅ OpenAI key loaded")
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    # Retrieval still works without a key; only the LLM summary is skipped.
    print("⚠️  No OPENAI_API_KEY in .env (still 'TODO'?). Answers will be the raw retrieved text.")
    client = None

# -------------------------
# SentenceTransformer model (embeddings)
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Chat memory
# -------------------------
chat_history = []

# -------------------------
# MCP selection
# -------------------------
def select_mcp(query: str, threshold: float = 0.2):
    """Select MCP handlers based on query similarity and print similarity scores."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_embedding, mcp_embeddings)[0]

    # Print MCP names and their similarity scores
    print("MCP similarities:")
    for i, sim in enumerate(sims):
        print(f"  {mcp_names[i]}: {sim:.2f} -> {mcps[mcp_names[i]]['description'][:80]}...")

    # Select MCPs above threshold
    ranked_indices = np.argsort(sims)[::-1]
    selected_indices = [i for i in ranked_indices if sims[i] >= threshold]

    if not selected_indices:
        print("No MCP exceeded the threshold. Returning empty list.")
        selected_indices = []

    return [mcps[mcp_names[i]]['handler'] for i in selected_indices]


def stream_openai_summary(query: str, combined_context: str, history=None):
    """
    Stream a summarized answer from OpenAI chat model for a given query and context.
    `history` is the earlier conversation (chat_history without the current query),
    so follow-up questions can refer back to it.

    Yields:
        str: partial chunks of the assistant's response as they arrive.
    """
    if client is None:
        # No API key: return the retrieved text as-is.
        yield combined_context
        return

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes MCP responses. Use the earlier conversation to understand follow-up questions."},
            *(history or []),
            {"role": "user", "content": f"User question: {query}\n\nMCP responses:\n{combined_context}"}
        ],
        stream=True
    )

    # Yield chunks as they arrive
    for event in completion:
        for choice in event.choices:
            delta_text = choice.delta.content
            if delta_text:
                yield delta_text


# -------------------------
# Full RAG pipeline
# -------------------------
def full_rag_pipeline(query: str, threshold: float = 0.1):
    """Run query through MCPs and summarize with OpenAI chat model."""
    previous_turns = chat_history[-6:]  # last few messages, sent to the LLM as context
    chat_history.append({"role": "user", "content": query})
    handlers = select_mcp(query, threshold)

    # Get MCP responses
    responses = []
    for handler in handlers:
        try:
            responses.append(handler(query))
        except Exception as e:
            responses.append(f"[Error from {handler.__name__}]: {e}")

    # Combine responses
    if responses == []:
        # Fallback: no MCP is confident, so don't ask the LLM to guess.
        print("MCP CANNOT ANSWER THAT")
        final_answer = "We cannot answer that."
        chat_history.append({"role": "assistant", "content": final_answer})
        return final_answer

    combined_context = "\n---\n".join(["".join(res) for res in responses])
    print(f"MCP responses: {responses}")  # Optional logging

    final_answer = "".join(stream_openai_summary(query=query, combined_context=combined_context, history=previous_turns))
    chat_history.append({"role": "assistant", "content": final_answer})

    return final_answer

# -------------------------
# Demo / Test
# -------------------------
if __name__ == "__main__":
    queries = [
        "How does the notification service work?",
        "Who can reset user passwords?",
        "Send me an alert when tickets update",
        "I cant log in anymore",
        "What is the weather in Bergen tomorrow?",
        "Can a woodchucker chuck wood if the wood chusk wood?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        answer = full_rag_pipeline(q, threshold=0.1)
        print(f"A: {answer}\n")
