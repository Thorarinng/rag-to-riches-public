import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# =========================
# TODO 0: Load datasets
# =========================
# Hint: create 'datasets.json' with MCP datasets
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "..", "data", "datasets.json")

with open(data_path, "r") as f:
    datasets = json.load(f)

# =========================
# TODO 1: Initialize embedding model
# =========================
# Hint: use SentenceTransformer("all-MiniLM-L6-v2")
model = None

# =========================
# TODO 2: Create FAISS indexes per MCP
# =========================
mcp_indexes = {}
mcp_texts = {}
for mcp_name, data in datasets.items():
    # TODO: Chunk texts .. (sent_tokenize)
    texts = [item["content"] for item in data]

    # TODO: encode texts into embeddings
    embeddings = None

    # TODO: create FAISS index and add embeddings
    dim = None
    index = None


    mcp_indexes[mcp_name] = index
    mcp_texts[mcp_name] = texts

# =========================
# TODO 3: Define MCPs with description and handler
# =========================
# Subtasks:
#  - Write description for each MCP
#  - Implement handler as a function that fetches data from FAISS
mcps = {
    "ticketing": {
        "description": None,
        "handler": lambda query: None
    },
    "user_management": {
        "description": None,
        "handler": lambda query: None
    },
    "notifications": {
        "description": None,
        "handler": lambda query: None
    }
}

# =========================
# TODO 4: Compute embeddings for MCP descriptions
# =========================
mcp_names = list(mcps.keys())
mcp_descs = [mcps[name]["description"] for name in mcp_names]

# TODO: encode MCP descriptions for semantic similarity
mcp_embeddings = None

# =========================
# TODO 5: MCP selection function
# =========================
def select_mcp(query: str):
    """
    TODO:
    1. Encode the query using the embedding model
    2. Compute cosine similarity with MCP description embeddings
    3. Select the best matching MCP
    4. Return the corresponding MCP handler
    """
    pass

# =========================
# TODO 6: Fetch top-k relevant content from MCP dataset
# =========================
def fetch_from_mcp(mcp_name, query, k=1):
    """
    TODO:
    1. Encode the query
    2. Search the FAISS index for the MCP
    3. Return the top-k document contents
    """
    pass

# =========================
# TODO 7: Demo / Test
# =========================
queries = [
    "I forgot my password and cannot login",
    "Who can assign roles to a new employee?",
    "Can I get Slack alerts for ticket updates?",
    "Request VPN access",
    "Reset permissions for my account"
]

for q in queries:
    # TODO:
    # 1. Select the appropriate MCP
    # 2. Fetch top-k results using the MCP handler
    handler = None
    results = None

    print(f"Q: {q}")
    print(f"A: {results}\n")
