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

for doc in docs:
    chunks = sent_tokenize(doc["content"])
    all_chunks.extend(chunks)
    chunk_titles.extend([doc["title"]] * len(chunks))

print("All chunks:", all_chunks)

# =========================
# TODO 2: Encode chunks into embeddings
# =========================
embeddings = model.encode(all_chunks, convert_to_numpy=True)

# =========================
# TODO 3: Build FAISS index and add embeddings
# =========================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# =========================
# TODO 4: Implement retrieval function
# =========================
def ask_rag(query: str, k=1):
    """
    1. Encode the query into an embedding using the SentenceTransformer model.
    2. Search the FAISS index to find the top-k most similar document chunks.
    3. Return the corresponding document chunk contents.
    """
    q_emb = model.encode([query], convert_to_numpy=True)
    scores, indices = index.search(q_emb, k)
    return {s: all_chunks[i] for s, i in zip(scores[0], indices[0])}

# =========================
# TODO 5: Demo / Test
# =========================
if __name__ == "__main__":
    queries = [
        "How does the notification service send alerts?",
        "Who can assign roles?"
    ]

    for q in queries:
        results = ask_rag(q, k=3)  # retrieve top 3 chunks
        print(f"\nQ: {q}")
        print("A:")
        for s, r in results.items():
            print(f"- L2-dist: {s:.2f}, results: {r}")
