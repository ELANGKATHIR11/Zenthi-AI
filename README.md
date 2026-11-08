---
title: Zenthi-AI OS
emoji: 🧠
colorFrom: purple
colorTo: indigo
app_port: 7860
pinned: false
license: apache-2.0
---

# Zenith-AI: Autonomous Multi-Agent AI Operating System & Zenith-4B Foundation Engine

**Zenith-AI** is a production-grade, local-first conversational AI Operating System and multi-agent dashboard powered by **Zenith-4B**—a fine-tuned, decoupled SLM+VLM foundation model with native multimodal vision, deep chain-of-thought algorithmic synthesis, and zero-hallucination document extraction.

Integrated with **POLAR-FLASH**, an ultra-low-bitrate compression runtime featuring polar-coordinate KV-cache quantization, thermal throttling safeguards, and INT2-FP16 GEMM matrix engines.

---

## 🌟 Key Highlights

- **Zenith-4B Model**: Pure fine-tuned combined SLM+VLM model with code, structured extraction, and visual reasoning merged via TIES consensus.
- **POLAR-FLASH Engine**: Sub-2-bit KV-cache quantization with polar coordinates $(\rho, \theta)$, achieving $<0.35$ MSE and saving up to $4\times$ KV memory.
- **Agentic OS Architecture**: 10-step multi-agent orchestration pipeline (Router, Planner, RAG, Web Search, Memory, Hybrid Execution, Validation).
- **Public Model Weights**: Available on Hugging Face at [`KATHIR2006/Zenith-4B`](https://huggingface.co/KATHIR2006/Zenith-4B).
- **Open-Source Codebase**: Maintained on GitHub at [`ELANGKATHIR11/Zenthi-AI`](https://github.com/ELANGKATHIR11/Zenthi-AI).

---

## 🏆 Industry-Grade Benchmark Results

Comprehensive head-to-head empirical evaluation across 4 core capabilities:

| Evaluation Pillar / Metric | Base Qwen3.5-4B | Zenith-4B (Fine-Tuned) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **1. Gen AI & Logical Reasoning** | 43.33% | 43.33% | Parity |
| **2. OCR & Document Extraction** | 16.67% | **77.78%** | **+61.11%** 🚀 |
| **3. Coding (Unit Test Pass Rate)** | 0.00% | **100.00%** | **+100.00%** 🏆 |
| *— Coding AST Syntax Validity* | 100.00% | 100.00% | 100% Valid |
| **4. Image & Diagram Analysis** | 66.67% | **100.00%** | **+33.33%** 📈 |
| **Overall Composite Score** | **31.67%** | **80.28%** | **+48.61% Overall Gain** |
| **Inference Throughput** | 61.37 tok/s | **84.08 tok/s** | **+37.0% Faster** |
| **Total Benchmark Latency** | 78.22s | **24.76s** | **3.15x Faster Completion** |

*Evaluation reports generated automatically and stored in [`reports/industry_benchmark_report.json`](reports/industry_benchmark_report.json) and [`reports/polarflash_scientific_benchmark.json`](reports/polarflash_scientific_benchmark.json).*

---

## ⚡ POLAR-FLASH Production Framework

The repository includes the complete production-grade `polarflash` library:

- **`polarflash_core`**: Spec format v1.0 parser, bitstream pack/unpack, and streaming runtime.
- **`polarflash_quant`**: Polar coordinate quantization $(\rho, \theta)$, 2-bit non-uniform KV cache, W4A16 GEMV kernel, INT2-FP16 GEMM with roofline models.
- **`polarflash_rt`**: Dynamic cache eviction (H2O / heavy-hitter policy), thermal throttler with automatic duty-cycle sleep, and speculative decode engine.
- **`polarflash_train`**: Decoupled multi-adapter trainer and TIES weight merger with sign-majority election.
- **`polarflash_bench`**: Scientific evaluation suite for perplexity, throughput, MSE validation, and automated latency profiling.

---

## 🛠️ Multi-Agent Architecture Lifecycle

Incoming requests trigger a high-efficiency 10-step lifecycle:

```text
User Request ──► 1. Input Sanitization
                     │
                     ▼
             2. Intent Router (Zenith-4B) ──► [CODE | VISION | RAG | SEARCH | KNOWLEDGE | COMPLEX]
                     │
                     ▼
             3. Planner Agent ──► Action DAG Sequencing
                     │
                     ├─► 4. Memory Integration (Session History)
                     ├─► 5. RAG Retrieval (Qdrant Dense Vector Store)
                     └─► 6. Web Search Integration (SearXNG Engine)
                     │
                     ▼
             7. Prompt Compilation & Context Assembly
                     │
                     ▼
             8. Hybrid Execution (Zenith-4B SLM + VLM Engine)
                     │
                     ▼
             9. Validation & Sanitization Agent
                     │
                     ▼
            10. Memory Persistence & UI Finalization
```

---

## 📂 Repository Structure

```text
Zenith-AI/
├── agents/                   # Multi-agent subclasses and orchestrator
│   ├── ollama_client.py      # Ollama inference client
│   ├── router_agent.py       # Intent router agent
│   ├── planner_agent.py      # Action sequencing agent
│   ├── agents.py             # RAG, search, memory, validation agents
│   └── orchestrator.py       # 10-step lifecycle manager
├── polarflash_core/          # Spec v1.0 binary format, packaging, bitstream
├── polarflash_quant/         # Polar coordinate & sub-2-bit KV quantizers, INT2 GEMM
├── polarflash_rt/            # Thermal throttling, streaming runtime, cache eviction
├── polarflash_train/         # Decoupled adapter trainer & TIES merger
├── polarflash_bench/         # Scientific benchmarker & roofline analyzer
├── models/Zenith-4B/         # Standalone Zenith-4B model weights & projector
│   ├── config.json           # Model configuration
│   ├── chat_template.jinja   # Chat formatting template
│   ├── mmproj-zenith-4b.gguf # Multimodal projector
│   ├── MODEL_CARD.md         # Digital signature and model card
│   └── benchmarks/           # Industry benchmark reports
├── reports/                  # Benchmark metrics & JSON verification logs
├── frontend/                 # Interactive glassmorphic dashboard
├── api/                      # FastAPI endpoints (chat, search, RAG, health)
├── ollama/                   # Modelfiles for local Ollama serving
├── scripts/                  # Packaging, benchmarking, evaluation scripts
├── tests/                    # Production test suite (21/21 passing)
├── app.py                    # Root dashboard application
├── pyproject.toml            # POLAR-FLASH packaging configuration
├── start.ps1                 # Native Windows launcher
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ELANGKATHIR11/Zenthi-AI.git
cd Zenthi-AI

# Install Python requirements and POLAR-FLASH framework
pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Ollama Local Serving

```bash
# Register Zenith-4B with Ollama
ollama create zenith-4b:latest -f ollama/Modelfile.zenith4b

# Run interactive inference
ollama run zenith-4b:latest
```

### 4. Launch Dashboard

- **Windows (PowerShell)**:
  ```powershell
  .\start.ps1
  ```
- **Direct Python**:
  ```bash
  python app.py
  ```
Open `http://localhost:7860` in your web browser.

---

## 📥 Model Download (Hugging Face)

The full Zenith-4B model weights and projector are publicly available on Hugging Face:

- **Hugging Face Model Hub**: [https://huggingface.co/KATHIR2006/Zenith-4B](https://huggingface.co/KATHIR2006/Zenith-4B)
- **Hugging Face Space**: [https://huggingface.co/spaces/KATHIR2006/zenthi-ai](https://huggingface.co/spaces/KATHIR2006/zenthi-ai)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "KATHIR2006/Zenith-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
```

---

## 🤝 Acknowledgments & Attribution

Special credit and acknowledgment to **Qwen 3.5 / 2.5 by Alibaba Cloud** for developing and open-sourcing the foundational base model architecture that made the fine-tuning, domain adaptation, and development of Zenith-4B possible.

---

## ⚖️ License

- **Framework, Agents, & Application Code**: Licensed under the **MIT License**.
- **Model Architecture & Weights**: Licensed under the **Apache License 2.0** (in accordance with base model terms).
