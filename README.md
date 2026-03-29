# 🤖 Agentic RAG Assistant with Self-Correction

A production-grade, containerized **RAG (Retrieval-Augmented Generation) system** leveraging **LangGraph**, **FastAPI**, and **Groq (Llama 3.3)**. This system features an advanced retrieval pipeline and an autonomous self-correction loop designed to eliminate hallucinations in real-time.

---

## 🏗️ System Architecture

The core logic is managed by a **LangGraph state machine**, ensuring a predictable yet flexible flow from initialization to final output.

---

## 🔍 Advanced Retrieval Strategy

To ensure the LLM receives the most relevant and diverse context, the system employs a **multi-stage retrieval process**:

- **Diverse Retrieval (MMR)**: Uses Maximal Marginal Relevance with a `λ_mult` of 0.5 to balance semantic relevance with result diversity, preventing the model from processing redundant information.
- **Multi-Query Expansion**: Generates multiple versions of the user's query to catch relevant documents that might use different terminology.
- **Contextual Compression**: Employs an `LLMChainExtractor` to strip away noise, passing only the essential document snippets to the generator.

---

## 🌟 Key Features

- **Self-Correction Loop**: A dedicated "Checker" node evaluates the generated answer against the context. If a hallucination is detected, the "Rewrite" node fixes it using a temperature-stable configuration.
- **Per-User Vector Isolation**: Automatically partitions **ChromaDB** storage by `user_id`, ensuring strict data isolation and session persistence.
- **Hybrid Backend**: Combines the lightning-fast inference of Groq (Llama-3.3-70b) with local HuggingFace embeddings (`all-MiniLM-L6-v2`).

---

## 🛠️ Tech Stack

| Component       | Technology                                         |
|-----------------|---------------------------------------------------|
| Orchestration   | LangGraph                                         |
| LLM             | Llama 3.3 70B (via Groq)                          |
| Retriever       | MMR + MultiQuery + Contextual Compression         |
| Embeddings      | HuggingFace (`all-MiniLM-L6-v2`)                  |
| Vector Store    | ChromaDB                                          |
| API / Web       | FastAPI + Vanilla JS/HTML5                        |
| Deployment      | Docker                                           |

---

## 🐳 Docker Deployment

The application is fully containerized for seamless deployment.

### 1. Build & Start Services

```bash
docker-compose up --build
```

### 2. Access the Application

- **Frontend**: [http://localhost:8000](http://localhost:8000)  
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6️⃣ Why This System is Robust

- **Retriever**: Multi-layered (MQR + MMR + LLM compression) → highly relevant context.
- **Hallucination Check**: Strict JSON output + LLM verification → faithful answers.
- **Chroma DB + user_id**: Persistent, isolated, scalable knowledge bases.
- **Regeneration Loop**: Ensures corrections until the answer is hallucination-free.
- **Frontend Integration**: FastAPI + LangGraph → visual and interactive experience.

---

## 📝 Workflow Diagram

```
START
 │
 ▼
init → initialize_state (sets regenerated=False, docs=[], verdict="Not_Sure")
 │
 ▼
retrieve → get_retriever(user_id) → ContextualCompressionRetriever → fetch docs
 │
 ▼
generate → generate_answer_chain → answer + final_status
 │
 ├── if final_status="not_in_docs" → END
 │
 ▼
check_answer → check_answer_quality_chain → verdict + reason
 │
 ├── if verdict="Not Supported" & regenerated=False → rewrite_answer
 │      │
 │      └─> rewrite_chain → answer (regenerated=True) → check_answer
 │
 ▼
END → format_output(state) → user-friendly message
```

---

## Author

**Arindam Das** | Machine Learning & AI Engineer

---

## License

MIT License

