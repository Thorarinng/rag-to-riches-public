# visualize_docs.py
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import umap
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

nltk.download("punkt")

# -------------------------
# Documents
# -------------------------
docs = [
    {"title": "Ticketing Service", "content": "Handles user requests like password resets and hardware issues."},
    {"title": "User Management Service", "content": "Stores employee accounts and roles. Admins can assign roles."},
    {"title": "Notification Service", "content": "Sends emails and Slack alerts when tickets update."},
    {"title": "Knowledge Base", "content": "Stores FAQs and solutions for common IT problems."}
]

# -------------------------
# Chunk the documents
# -------------------------
all_chunks = []
chunk_titles = []

for doc in docs:
    chunks = sent_tokenize(doc["content"])
    all_chunks.extend(chunks)
    chunk_titles.extend([doc["title"]] * len(chunks))

# -------------------------
# Encode chunks
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks, convert_to_numpy=True)

# -------------------------
# Visualization function
# -------------------------
def visualize_embeddings(embeddings, chunk_titles, all_chunks, title="Document Chunk Embeddings in 2D"):
    reducer = umap.UMAP(n_neighbors=2, min_dist=0.1, metric='euclidean', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    colors_map = {
        "Ticketing Service": "red",
        "User Management Service": "blue",
        "Notification Service": "green",
        "Knowledge Base": "orange"
    }

    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, color=colors_map.get(chunk_titles[i], "gray"),
                    label=chunk_titles[i] if i == 0 or chunk_titles[i] not in chunk_titles[:i] else "")
        plt.text(x + 0.01, y + 0.01, all_chunks[i], fontsize=9)

    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.show()

# -------------------------
# Run visualization
# -------------------------
visualize_embeddings(embeddings, chunk_titles, all_chunks)
