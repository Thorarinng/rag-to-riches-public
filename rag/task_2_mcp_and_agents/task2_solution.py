import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize


# =========================
# TODO 1: Load datasets
# =========================
# Hint: create 'datasets.json' with MCP datasets
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "..", "data", "datasets.json")

with open(data_path, "r") as f:
    datasets = json.load(f)

# =========================
# TODO 2: Initialize embedding model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# TODO 3: Create FAISS indexes per MCP
# =========================
mcp_indexes = {}
mcp_texts = {}
for mcp_name, data in datasets.items():

    all_chunks = [sentence for item in data for sentence in sent_tokenize(item['content'])]
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    
    # TODO: create FAISS index and add embeddings
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    mcp_indexes[mcp_name] = index
    mcp_texts[mcp_name] = all_chunks

# =========================
# TODO 4: Define MCPs with description and handler
# =========================
# Subtasks:
#  - Write description for each MCP
#  - Implement handler as a function that fetches data from FAISS
mcps = {
    "ticketing": {
        "description": (
            "Responsible for handling all end-user IT requests and issues, including password resets, "
            "account lockouts, hardware problems, VPN access, and software installation requests. "
            "This MCP ensures that users can regain access to systems quickly, receive necessary "
            "hardware or software support, and maintain smooth day-to-day operations."
        ),
        "handler": lambda query: fetch_from_mcp("ticketing", query)
    },
    "user_management": {
        "description": (
            "Manages employee accounts, including creation, deactivation, role assignments, and "
            "permissions auditing. This MCP ensures proper access control, compliance with security "
            "policies, and maintains a clear audit trail of all user-related changes within the organization."
        ),
        "handler": lambda query: fetch_from_mcp("user_management", query)
    },
    "notifications": {
        "description": (
            "Handles sending notifications and alerts through email, Slack, or other channels. "
            "Includes ticket updates, maintenance alerts, security warnings, and system outage notifications. "
            "This MCP ensures that users and teams are informed about important events in a timely manner."
        ),
        "handler": lambda query: fetch_from_mcp("notifications", query)
    }
}

# =========================
# TODO 5: Compute embeddings for MCP descriptions
# =========================
mcp_names = list(mcps.keys())
mcp_descs = [mcps[name]["description"] for name in mcp_names]

# TODO: encode MCP descriptions for semantic similarity
mcp_embeddings = model.encode(mcp_descs, convert_to_numpy=True)
faiss.normalize_L2(mcp_embeddings)

# =========================
# TODO 6: MCP selection function
# =========================
def select_mcp(query: str):
    # TODO: encode query
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    # TODO: compute cosine similarity with mcp_embeddings
    similarities = cosine_similarity(q_emb, mcp_embeddings)[0]
    
    best_idx = np.argmax(similarities)
    best_mcp = mcp_names[best_idx]
    return mcps[best_mcp]["handler"]

# =========================
# TODO 7: Fetch top-k relevant content from MCP dataset
# =========================
def fetch_from_mcp(mcp_name, query, k=1):
    # TODO: encode query and search FAISS index
    print(f"Logging: Preparing to execute MCP='{mcp_name}' for query: '{query}'")
    index = mcp_indexes[mcp_name]
    texts = mcp_texts[mcp_name]
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)

    print(f"Vector Search result in '{mcp_name}' with score: {scores[0]}")
    
    # TODO: return list of top-k results
    results = [texts[i] for i in idxs[0]]
    return results


if __name__ == "__main__":
    # --- Demo ---

    # =========================
    # TODO 8: Test the pipeline
    # =========================
    queries = [
        "I forgot my password and cannot login",
        "Who can assign roles to a new employee?",
        "Can I get Slack alerts for ticket updates?",
        "Request VPN access",
        "Reset permissions for my account"
    ]

    for q in queries:
        print(f"Q: {q}")

        # TODO: select MCP and fetch results
        handler = select_mcp(q)
        results = handler(q)
        
        print(f"A: {results}\n")
