# Workshop: Hands-on RAG with MCPs, Agents, and Chat

---

## Setup (do this before we start)

You need **Python 3.10 or newer** (check with `python3 --version`).

```bash
make setup                   # creates venv, installs packages, creates .env
# On macOS the default python3 is often 3.9. If so, use a newer one:
# make setup PY=python3.12
source venv/bin/activate     # Windows: venv\Scripts\activate
```

- **Task 1 and 2** need no API key.
- **Task 3** uses OpenAI for the final answer. Put the key from your mentor in `.env`
  as `OPENAI_API_KEY=...` (`make setup` creates the file with `OPENAI_API_KEY=TODO`).
  Without a key, Task 3 still routes and retrieves, but the LLM step is skipped.

Files for each task:

| Task | Your file | Reference solution |
| --- | --- | --- |
| 1 | `rag/task_1_rag_basics/task1.py` | `task1_solution.py` |
| 2 | `rag/task_2_mcp_and_agents/task2.py` | `task2_solution.py` |
| 3 | `rag/task3_rag_completed.py` | `task3_rag_completed_solution.py` |

Run a file from the repo root, e.g. `python rag/task_1_rag_basics/task1.py`.
Task 3 imports the Task 2 solution, so you can do Task 3 even if Task 2 is unfinished.

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
- [ ] Write `select_mcp(query, threshold)` that returns **every** MCP handler whose cosine similarity is above the threshold (best first), or an empty list
- [ ] Write `full_rag_pipeline(query: str)` that:
  - Adds user query to chat history
  - Uses `select_mcp` to pick the relevant MCPs
  - Calls each MCP handler and combines the results
  - Asks the OpenAI chat model to write the answer (send the earlier `chat_history` too, so follow-up questions work)
  - If no MCP matches, answers "We cannot answer that." without calling the LLM
  - Adds the answer to chat history and returns it
- [ ] Test multi-turn interaction with at least 2–3 queries in sequence
- [ ] Test the fallback with an off-topic question, e.g. "What is the weather in Bergen tomorrow?"
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
