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
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

print(f"✅ OpenAI token loaded: {OPENAI_API_KEY[:8]}...")
client = OpenAI(api_key=OPENAI_API_KEY)

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


def stream_openai_summary(query: str, combined_context: str):
    """
    Stream a summarized answer from OpenAI chat model for a given query and context.
    
    Yields:
        str: partial chunks of the assistant's response as they arrive.
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes MCP responses."},
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
        print("MCP CANNOT ANSWER THAT")
        combined_context = "We cannot answer that."
    else:
        combined_context = "\n---\n".join(["".join(res) for res in responses])
    
    print(f"MCP responses: {responses}")  # Optional logging

    final_answer = "".join(stream_openai_summary(query=query, combined_context=combined_context))
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
        "What time is it?",
        "Can a woodchucker chuck wood if the wood chusk wood?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        answer = full_rag_pipeline(q, threshold=0.1)
        print(f"A: {answer}\n")
