---
title: Zenthi-AI OS
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Zenthi-AI: Agentic Multi-Model AI Operating System

Zenthi-AI OS is a production-ready, local-first conversational AI Operating System dashboard built from scratch. It intelligently orchestrates multiple expert AI models (`qwen2.5-coder:3b`, `riven/smolvlm:latest`, and `qwen2.5:1.5b-instruct`) using a 10-step agentic lifecycle, loading only the required expert model in VRAM for each query.

---

## 🛠️ Multi-Agent Architecture

Every incoming request follows a structured 10-step lifecycle to optimize context, memory, and model selection:

1. **Input Validation**: Sanitizes queries and handles visual base64 attachments.
2. **Intent Classification**: Evaluated by the **Router Agent** (running `qwen2.5:1.5b-instruct`) to detect intent (`CODE`, `VISION`, `RAG`, `SEARCH`, `KNOWLEDGE`, `COMPLEX`).
3. **Workflow Planning**: For `COMPLEX` multi-step tasks, the **Planner Agent** designs a sequence of tools to run.
4. **Memory Integration**: Fetches session-based conversational history.
5. **RAG Retrieval**: Retrieves context chunks from local document collections indexed in ChromaDB.
6. **Web Search Integration**: Gathers search context from local SearXNG metasearch API.
7. **Prompt Compilation**: Assembles the exact context, history, and user query.
8. **Expert Dispatching**: Directs the compiled prompt to the assigned expert model in Ollama.
9. **Output Validation**: Evaluated by the **Validation Agent** to clean up any syntax or leakages.
10. **Finalization**: Saves the response in the session memory and returns it to the dashboard.

---

## 📊 Routing & Classification Accuracy Benchmarks

We executed a comprehensive benchmark evaluation on **500 unique test queries** (100 per category) running on a local **RTX 5060 GPU**:

* **Overall Routing Accuracy**: **72.60%**
* **Average Latency**: **651.54 ms** per query

| Intent Category | Routing Accuracy (%) | Assigned Expert Model |
| :--- | :--- | :--- |
| **CODE** (Programming, math, algorithms) | **100.00%** | `qwen2.5-coder:3b` |
| **VISION** (Visual prompts, image understanding) | **100.00%** | `riven/smolvlm:latest` |
| **SEARCH** (News, current events, live updates) | **99.00%** | `qwen2.5:1.5b-instruct` |
| **RAG** (Context queries, files, uploads) | **43.00%** | `qwen2.5:1.5b-instruct` |
| **KNOWLEDGE** (General writing, history, summaries) | **21.00%** | `qwen2.5:1.5b-instruct` |

---

## 📂 Project Organization

```text
Zenthi-AI/
├── agents/                   # Modular agent subclasses and orchestrator
│   ├── ollama_client.py      # Ollama API connection
│   ├── router_agent.py       # Intent router agent
│   ├── planner_agent.py      # Action sequencing agent
│   ├── agents.py             # Helper agents (RAG, search, memory, validation)
│   └── orchestrator.py       # 10-step lifecycle manager
├── api/                      # FastAPI endpoints (chat, upload, health)
│   └── main.py               # Main api driver, mounts static web UI at root /
├── frontend/                 # Glassmorphic dashboard web UI
│   ├── index.html            # Landing page
│   ├── app.js                # Frontend connector (attachment previews & agent traces)
│   └── style.css             # Main styling rules
├── rag/                      # ChromaDB vector store helpers
├── search/                   # SearXNG wrapper utilities
├── memory/                   # Session chat history manager
├── Dockerfile                # Hugging Face deployment container configuration
├── start.sh                  # Container entrypoint script preloading VRAM models
└── requirements.txt          # Python dependencies
```

---

## 🐳 Running Locally & Deployment

### 1. Run via Docker (Same as Hugging Face Spaces)
Build and run the self-contained container:
```bash
docker build -t zenthi-ai .
docker run -p 7860:7860 --gpus all zenthi-ai
```
Open `http://localhost:7860` to access the chat dashboard.

### 2. Manual Local Setup
1. **Activate Environment**:
   ```bash
   conda activate dgpu-core
   pip install -r requirements.txt
   ```
2. **Launch Ollama Server**:
   Ensure Ollama is running and models are downloaded:
   - `qwen2.5:1.5b-instruct`
   - `qwen2.5-coder:3b`
   - `riven/smolvlm:latest`
3. **Run API Server**:
   ```bash
   python api/main.py
   ```
4. **Access UI**: Open `frontend/index.html` in your browser.

---

## ⚖️ Licenses & Compliance

This project is dual-licensed:
* **RAG Engine, Multi-Agent Framework, API Backend, & Web UI Codebase**: Licensed under the **MIT License**.
* **LLM Model Weights & Quantizations**: Licensed under the **Apache License 2.0** (in compliance with the base Qwen2.5 license).
