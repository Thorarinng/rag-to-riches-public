# Workshop: Hands-on RAG with MCPs, Agents, and Chat

---

## Section 1: Basic Vector Search (20 min)

**Goal:** Understand embeddings, FAISS indexing, and simple retrieval.

### User Asks

- Encode documents into embeddings.
- Store embeddings in a FAISS index.
- Retrieve the most relevant document given a query.
- Experiment with top-k results.

### TODOs

- [ ] Install libraries: `sentence-transformers`, `faiss-cpu`, `numpy`
- [ ] Define your documents (`title` + `content`)
- [ ] Encode documents into embeddings using `SentenceTransformer`
- [ ] Create a FAISS index and add embeddings
- [ ] Write a function `ask_rag(query: str, k=1)` that returns the top-k results
- [ ] Test the function with 2–3 example queries
- [ ] Optional: Visualize cosine similarity vs L2 distance

## Optional Experiments

### 1. Change Embedding Models

- Try different SentenceTransformers, e.g., `all-mpnet-base-v2`, `multi-qa-MiniLM-L6-cos-v1`.
- Compare retrieval results for the same queries.
- **Learning:** Understand how model choice affects semantic similarity.

### 2. Test Similarity Measures

- Switch from L2 to **cosine similarity** (normalize embeddings first).
- Observe differences in which documents are retrieved.
- **Learning:** Learn how distance metrics influence semantic search.

### 3. Dynamic Document Addition

- Add new documents at runtime with `index.add()`.
- Test if retrieval updates correctly after adding new documents.
- **Learning:** Explore real-time indexing and vector updates.

---

## Section 2: MCPs and Agent Selection (40 min)

**Goal:** Learn to dynamically route queries to the right MCP/agent using embeddings instead of static keywords.

### User Asks

- Define multiple MCPs with a description of their responsibilities.
- Compute embeddings for each MCP’s description.
- Convert user queries into embeddings.
- Compare query embeddings to MCP embeddings (cosine similarity) and select the best MCP.
- Route queries to the selected MCP.
- Experiment with top-k MCP suggestions if needed.

### TODOs

- [ ] Define at least 3 MCPs (`ticketing`, `user_management`, `notifications`) with a `handler` function and a textual `description`
- [ ] Compute embeddings for MCP descriptions using `SentenceTransformer`
- [ ] Write a function `select_mcp(query: str)` that:
  - Converts the query to an embedding
  - Computes cosine similarity with each MCP embedding
  - Returns the MCP handler with the highest similarity
- [ ] Test `select_mcp` with queries like `"I forgot my password"` and `"Assign a new role"`
- [ ] Optional: Implement top-2 or top-3 MCP suggestions and let the system choose among them

---

## Section 3: Full RAG Pipeline with Multi-Turn Chat (50 min)

**Goal:** Combine everything into a fully functional RAG system with multi-agent orchestration and optional chat memory.

### User Asks

- Maintain a chat memory for multi-turn conversations.
- Use `select_mcp` to pick the right MCP dynamically for each query.
- Inside the MCP, retrieve relevant documents using FAISS.
- Return a synthesized response (optional: aggregate top-k results).
- Experiment with multi-agent collaboration: one MCP retrieves, another summarizes.
- Optionally handle fallback scenarios if no MCP is confident.

### TODOs

- [ ] Initialize a `chat_history` list to store previous queries and answers
- [ ] Write `full_rag_pipeline(query: str)` that:
  - Adds user query to chat history
  - Uses `select_mcp` to pick the best MCP
  - Calls the MCP handler
  - Adds the MCP response to chat history
  - Returns the response
- [ ] Test multi-turn interaction with at least 2–3 queries in sequence
- [ ] Optional: Modify MCPs to aggregate top-k FAISS results and summarize
- [ ] Optional: Implement fallback MCP or default response for low similarity queries
- [ ] Optional: Integrate an LLM to rewrite or summarize responses

---

## Optional Challenges

- **Dynamic Query Chunking:** Split long queries or documents into smaller chunks before embedding and retrieval. Aggregate results from multiple chunks to improve accuracy.

- **Multi-Agent Orchestration:** Use multiple MCPs/agents to collaborate on a query. For example, one MCP retrieves documents, another summarizes, and another adds external context.

- **Visualize Query → MCP → FAISS → Response:** Show a flow of how a query travels through the system. Highlight which MCP was chosen and which documents contributed to the final answer.

- **Benchmark Routing with Multiple MCPs and Large Datasets:** Measure performance (speed, accuracy) when scaling MCPs and documents. Experiment with `top_k`, embeddings, and MCP selection strategies.

## Sources

- SentenceTransformer
  - https://sbert.net/docs/package_reference/sentence_transformer/index.html
- FAISS
  - https://github.com/facebookresearch/faiss
