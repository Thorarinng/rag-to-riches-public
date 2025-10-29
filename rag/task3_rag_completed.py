# =========================
# Workshop Exercise: RAG with SentenceTransformer + OpenAI
# =========================

import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Environment setup
# -------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Add current folder to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MCP data
from task_2_mcp_and_agents.task2_solution import mcp_embeddings, mcp_names, mcps

# -------------------------
# Initialize OpenAI client (chat only)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# TODO: Initialize the sentence transformer model
# =========================
model = None  # e.g., SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# TODO: Initialize chat memory
# =========================
chat_history = []

# =========================
# TODO: MCP selection function
# =========================
def select_mcp(query: str, threshold: float = 0.2):
    """
    TODO:
    1. Encode the query using the model.
    2. Compute cosine similarity with all MCP embeddings.
    3. Select MCPs that exceed the threshold.
    4. If none exceed, return empty list.
    5. Return the corresponding MCP handlers.
    """
    None

# =========================
# TODO: Full RAG pipeline
# =========================
def full_rag_pipeline(query: str):
    """
    TODO:
    1. Add user query to chat_history.
    2. Use select_mcp to pick relevant MCPs.
    3. Execute each MCP handler and collect responses.
    4. Combine responses.
    5. Use OpenAI chat model to summarize / finalize answer.
    6. Store assistant response in chat_history.
    7. Return the final combined response.
    """
    None

# =========================
# TODO: Demo / Test
# =========================
if __name__ == "__main__":
    queries = [
        "How does the notification service work?",
        "Who can reset user passwords?",
        "Send me an alert when tickets update"
    ]

    for q in queries:
        print(f"\nQ: {q}")
        # TODO: Call full_rag_pipeline and print the answer
        print(f"A: {None}\n")
