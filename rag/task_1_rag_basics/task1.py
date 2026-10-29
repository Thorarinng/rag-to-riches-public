# =========================
# FAISS + SentenceTransformer RAG Example (Workshop-ready)
# =========================

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")  # needed by NLTK 3.9+
from nltk.tokenize import sent_tokenize

# -------------------------
# Documents
# -------------------------
docs = [
    {"title": "Ticketing Service", "content": "Handles user requests like password resets and hardware issues."},
    {"title": "User Management Service", "content": "Stores employee accounts and roles. Admins can assign roles."},
    {"title": "Notification Service", "content": "Sends emails and Slack alerts when tickets update."},
    {"title": "Knowledge Base", "content": "Stores FAQs and solutions for common IT problems."}
]

# =========================
# Setup: embedding model (already done for you)
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# TODO 1: Apply chunking to all documents
# =========================
all_chunks = []
chunk_titles = []

# TODO: split each document content into chunks and store in all_chunks
# TODO: keep track of chunk titles in chunk_titles

# =========================
# TODO 2: Encode chunks into embeddings
# =========================
embeddings = None  # TODO: encode all_chunks into embeddings

# =========================
# TODO 3: Build FAISS index and add embeddings
index = None  # TODO: create FAISS index and add embeddings

# =========================
# TODO 4: Implement retrieval function
def ask_rag(query: str, k=1):
    """
    1. Encode the query into an embedding using the SentenceTransformer model.
    2. Search the FAISS index to find the top-k most similar document chunks.
    3. Return the corresponding document chunk contents.
    """
    # TODO: encode query, search index, and return results
    # Hint: use model.encode([query], convert_to_numpy=True) and index.search(...)

    return None

# =========================
# TODO 5: Demo / Test
if __name__ == "__main__":
    queries = [
        "How does the notification service send alerts?",
        "Who can assign roles?"
    ]

    for q in queries:
        results = None  # TODO: call ask_rag with query
        print(f"\nQ: {q}")
        print("A:")
        # TODO: print results
