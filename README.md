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

# Zenthi-AI: Custom Small Language Model (SLM) Project

Zenthi-AI is a production-ready, custom fine-tuned Small Language Model conversational assistant built from scratch for a college final-year project. It integrates local retrieval-augmented generation (RAG), live web search metasearch engines, session-based memory history, and a responsive dark-themed dashboard.

---

## Technical Stack & Features
* **Model Base**: `Qwen/Qwen2.5-0.5B-Instruct`
* **Fine-Tuning (QLoRA)**: Fine-tuned on a merged, cleaned dataset of Alpaca, Dolly 15K, OpenHermes, UltraChat, and ShareGPT.
* **Quantization**: GGUF format for local deployment (quantized size under 500 MB).
* **Local RAG**: ChromaDB database utilizing local `all-MiniLM-L6-v2` Sentence Transformer embeddings. Handles text chunking and indexing for PDF, TXT, and Markdown files.
* **Live Web Search**: Custom SearXNG metasearch engine client wrapper with robust public-instance and offline mock fallbacks.
* **Session Memory**: Windowed conversation manager preserving context and identity boundaries.
* **Backend API**: FastAPI server exposing endpoints for chat routing, file ingestion, and health metrics.
* **Web UI**: Modern dark glassmorphic HTML/CSS/JS dashboard.

---

## Directory Structure
```text
project/
├── datasets/                 # Merged/cleaned datasets, Alpaca/Dolly/train JSON splits
├── synthetic_data/           # Synthetic instruction generation pipelines
├── training/                 # LoRA fine-tuning scripts and merging pipelines
├── evaluation/               # Model validation, benchmark scripts, metric logs
├── rag/                      # ChromaDB helper scripts and ingestion models
├── search/                   # SearXNG wrapper, search ranking and scraping utilities
├── memory/                   # Chat history and windowed memory logic
├── api/                      # FastAPI endpoints (chat, upload, health)
├── frontend/                 # Web UI dashboard (HTML/CSS/JS)
├── ollama/                   # Modelfile configuration for Ollama deployment
├── docs/                     # Guides, reports, and architectures
└── requirements.txt          # Python dependencies
```

---

## Setup & Running Locally

### 1. Environment Activation
Activate your conda environment containing deep learning dependencies:
```bash
conda activate dgpu-aiml
pip install -r requirements.txt
```

### 2. Dataset Cleaning & Ingestion
Clean and compile all datasets (Alpaca, Dolly, OpenHermes, UltraChat, ShareGPT) into unified train/val json structures:
```bash
python scripts/prepare_dataset.py
```

### 3. Identity Synthetic Data Generation
Generate synthetic reasoning, programming, and identity dataset turns defining the Zenthi-AI persona:
```bash
python synthetic_data/generator.py
```

### 4. Running Model Fine-Tuning
Execute local parameter-efficient QLoRA training on your GPU:
```bash
python training/train.py
```

### 5. Model Merging & Exporting
Merge LoRA adapters into base weights for GGUF compilation:
```bash
python training/export_gguf.py
```

### 6. Packaging into Ollama
Create your customized local Ollama model mapping the Zenthi-AI system prompt and chat template:
```bash
ollama create Zenthi-AI -f ollama/Modelfile
```

### 7. Run backend FastAPI API & Frontend Server
Launch backend server:
```bash
python api/main.py
```
Open `frontend/index.html` in any browser to start chat sessions, upload documents, and toggle auto/hybrid routing settings!
>>>>>>> 513ed1f (Initial commit of Zenthi-AI codebase)
