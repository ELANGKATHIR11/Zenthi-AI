Project Zenthi-AI: Custom SLM + VLM Hybrid Architecture & Engineering Plan (Revised)
Revision notes: this version corrects the KV-cache/context-window mismatch, separates vision-track training from the code/extraction souping pass, adds a decontamination and evaluation phase, and stratifies the data synthesis targets. Changes from the original are marked inline.
Executive Overview
Zenthi-AI is a proprietary, unified Small Language Model + Vision Language Model (SLM+VLM) tailored for edge and consumer hardware, targeting 3.0–3.5 GB VRAM so it runs on 4 GB laptop GPUs and low-spec workstations.
Rather than relying on off-the-shelf generalist weights, Zenthi-AI is engineered through multi-task adapter distillation, model souping (TIES/DARE merging), and quantization-aware vision projection, using the local repositories and datasets already present in this workspace:
• Base Foundation: Qwen3.5-4B-Q4_K_M.gguf & mmproj-Qwen3.5-4B-BF16.gguf
• Training & Merging Framework: Soup-main (soup-cli with layer streaming and multi-adapter merging)
• Data Synthesis & Quality Auditing Engine: crewAI-main
• Knowledge Retrieval Engine: Embedded Qdrant Vector Store

1. Target Specifications & Efficiency Goals
Metric
Target Specification
Comparison vs. Raw Qwen 3.5 4B
Model Architecture
Unified Decoupled Vision-Language Transformer
Same architecture, domain-specialized weights
Quantization Format
IQ4_XS / Q4_K_M (Base) + Q8_0 (mmproj)
~20% reduction in total footprint
Total Disk Size
< 2.65 GB total (Base LLM + mmproj)
Down from 3.15 GB
Runtime VRAM (Active, 8k ctx)
~2.8 GB – 3.4 GB
Operates smoothly on 4 GB GPUs (GTX 1650, RTX 3050 4GB)
Default Context Window (revised)
8,192–16,384 tokens on 4 GB devices
KV cache scales ~linearly with context; 32k is not safe at 4 GB — see §1.1
Extended Context Window
32,768 tokens (RoPE-scaled from 256k)
Available on 8 GB+ cards only, or with aggressive KV quantization
Core Capabilities
OCR, Deep Coding, Web Scraping Extraction, Document Synthesis, Visual Reasoning
Zero prompt-drift; strictly aligned to production tasks
1.1 Context Window vs. KV Cache Budget (new)
The original plan quoted a single KV cache figure (~0.6 GB) at 8k context but advertised a 32k window — these are inconsistent, since KV cache size grows roughly linearly with sequence length. The table below makes the tradeoff explicit so the context window claim is defensible per device tier.
Context Length
KV Cache (FP16)
KV Cache (Q8 cache)
Fits 4 GB ceiling?
4,096
~0.30 GB
~0.15 GB
Yes
8,192
~0.60 GB
~0.30 GB
Yes
16,384
~1.20 GB
~0.60 GB
Marginal — depends on quant tier
32,768
~2.40 GB
~1.20 GB
No on IQ4_XS tier; only with Q8 KV cache and Tier 1 weights on 6–8 GB cards
Decision: ship 8k as the guaranteed default on 4 GB devices, enable 16k with Q8 KV cache as an opt-in, and gate 32k to the 6–8 GB VRAM profile only. This replaces the single-number VRAM claim in the original plan.
2. The 6-Pillar Domain Specialization
3. Text Understanding & Reasoning — native chain-of-thought (<think>) fine-tuning for step-by-step problem solving.
4. High-Accuracy OCR — direct transcription of screenshots, receipts, handwritten notes, tables, and low-contrast UI elements.
5. Software Engineering & Coding — algorithmic reasoning, syntax correction, type annotations, unit-test generation.
6. Structured Web Scraping Extraction — converting messy HTML/Markdown/web text into strict Pydantic/JSON schemas without hallucinated attributes.
7. Generative AI & Document Generation — executive reports, technical documentation, tables, export-ready Markdown/LaTeX.
8. Visual Scene & Diagram Analysis — architecture flowcharts, UI wireframes, chart metric extraction.
9. Detailed Execution Roadmap
Phase 1: High-Density Data Engine (crewAI-main)
The Data Crew:
• CodeSpecialistAgent — extracts/generates coding and debugging challenges from datasets/CodeParrot-Small and HumanEval.
• VisionAnnotationAgent — processes images in datasets/vision_images to generate dense OCR ground truths, bounding descriptions, and UI breakdown QA pairs.
• WebExtractorAgent — takes raw HTML/SearXNG snippets and generates queries paired with clean JSON extraction outputs.
• QualityAuditorAgent — evaluates every synthetic sample, removing formatting leaks (<|im_start|>) and vague answers.
Decontamination step (new): before training, strip any sample whose source problem overlaps with HumanEval or other public eval sets used later in Phase 2.5. Training on eval data invalidates every downstream benchmark number.
Stratified data targets (revised): 15,000 total examples spread evenly across 6 domains is too thin per domain after quality filtering. Set explicit per-domain minimums, re-running synthesis for any domain that falls short after auditing:
Domain
Minimum examples
Code / algorithms
4,000
OCR / vision-language
3,000
Web extraction (HTML→JSON)
3,000
Reasoning / chain-of-thought
2,000
Document generation
2,000
Diagram / chart analysis
1,000
Output artifact: datasets/zenthi_hybrid_train.jsonl, decontaminated, stratified, target ~15,000+ examples (scale up per domain as auditor yield requires).
Phase 2: Layer-Streamed Training (Soup-main)
• Train with stream_layers: true, keeping the base model in system RAM and streaming layers to GPU VRAM one at a time — avoids OOM within a 4 GB VRAM ceiling.
• Train three focused LoRA task vectors:
• adapter-code — code, algorithms, regex.
• adapter-vision — OCR, document layout parsing, visual reasoning (trained against the language backbone, in coordination with the projector track in Phase 4).
• adapter-extraction — web scraping HTML-to-JSON and report generation.
• Checkpointing (new): checkpoint every N steps (not just per-epoch) given the long-running nature of layer-streamed training on constrained VRAM — a crash mid-epoch on a multi-hour streamed run is expensive to lose.
Phase 2.5: Per-Adapter Evaluation (new phase)
Before any merging happens, run each adapter — and later the merged model — against a fixed, held-out eval slice per domain (never seen during training or data synthesis). This phase exists because nothing in the original plan measured whether souping preserves per-domain performance; without it, Phase 5's quantization matrix could be built on a regressed model.
• Build held-out sets per domain (small, curated, excluded from Phase 1 output).
• Score each single-adapter checkpoint independently.
• Record baseline numbers to compare against the merged model in Phase 3.
Phase 3: Model Souping & Weight Merging (revised)
The original plan merged all three adapters (code, vision, extraction) together via TIES/DARE. This is risky because TIES/DARE operate on weight-space deltas, and adapter-code and adapter-extraction share a compatible modality and objective (both operate purely on text), while adapter-vision's useful signal is split between the language backbone and the mmproj projector (Phase 4) — small-magnitude vision-relevant deltas can be overwritten by larger code-adapter deltas during merge.
• Step 1: Merge adapter-code + adapter-extraction via DARE + TIES — compatible modality, lower interference risk.
• Step 2: Validate adapter-vision independently (language-backbone LoRA + optimized projector) against the Phase 2.5 OCR/vision held-out set before it touches the merge.
• Step 3: Only fold adapter-vision into the merged weights once its standalone OCR accuracy is confirmed; re-run Phase 2.5 eval on the fully merged model and compare against all three single-adapter baselines.
• Rollback point (new): if the merged model regresses on any domain beyond an agreed threshold (e.g., >3% drop), adjust DARE drop-rates and re-merge rather than restarting data synthesis.
Phase 4: Vision Projector Optimization (mmproj)
• Current mmproj-Qwen3.5-4B-BF16.gguf is 630 MB in uncompressed BF16.
• Profile vision patch token density.
• Quantize projector weights to Q8_0 (~630 MB → ~340 MB).
• Explicit validation (revised): measure OCR accuracy before and after Q8_0 quantization on the held-out OCR set from Phase 2.5 — don't assume "near-zero loss"; confirm it with a number.
Phase 5: Low-Spec Quantization Matrix
• Tier 1: Zenthi-AI-Hybrid-4B-Q4_K_M (~2.5 GB) — balanced performance for 6–8 GB VRAM GPUs, supports up to 16k–32k context per §1.1.
• Tier 2: Zenthi-AI-Hybrid-4B-IQ4_XS (~2.1 GB) — extreme compression via importance matrices, for 4 GB VRAM GPUs or CPU+RAM execution, default 8k context per §1.1.
• Run Phase 2.5 eval sets against both quantized tiers before shipping, to catch any quantization-induced regression separately from the merge-induced regression already checked in Phase 3.
Phase 6: System Integration & Serving
• Construct the unified ollama/Modelfile registering zenthi-ai:latest.
• Configure native context limits: num_ctx 8192 default (4 GB profile), num_ctx 16384 with Q8 KV cache opt-in, scaling to 32768 only on the 6–8 GB profile.
• Validate live end-to-end inference through api/main.py and the glassmorphic frontend dashboard.
10. Hardware & Resource Budgeting (RTX 5060 + 4 GB Low-Spec Profile)
Resource
RTX 5060 (Host Dev Machine)
Target 4 GB Low-Spec Device (8k ctx, default)
Available VRAM
8.52 GB
4.00 GB
Model Weights (IQ4_XS / Q4_K_M)
2.50 GB
2.10 GB
Vision Projector (Q8_0 / BF16)
0.63 GB
0.34 GB
KV Cache (8k context, Q8 cache)
~0.60 GB
~0.30 GB
CUDA Activation Workspace
~0.50 GB
~0.35 GB
Total Peak Runtime VRAM
~4.23 GB
~3.09 GB
Status
Fast (Zero Paging)
Fits comfortably under 4 GB at 8k default context
Note: at 16k context with Q8 KV cache, the 4 GB device total rises to ~3.39 GB (still fits); at 32k it rises to ~3.99 GB, leaving effectively no margin — hence 32k is gated to the 6–8 GB profile in §1.1, not promised as a default on 4 GB hardware.
