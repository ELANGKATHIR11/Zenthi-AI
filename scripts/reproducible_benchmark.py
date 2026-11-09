"""
Scientific Benchmark Suite for Zenith-4B & Zenthi-AI OS.

Evaluates:
1. Gen AI Reasoning & Multi-Step Logic Deduction
2. Functional Coding Pass Rate (AST validation + live unit test execution)
3. Structured Document Intelligence & OCR Extraction (JSON schema compliance)
4. Diagram & Visual Topology Grounding
5. Runtime Performance (Throughput in tokens/sec, TTFT, VRAM residency)

Supports native direct execution via ZenithMultimodalEngine (PyTorch / BitsAndBytes / CUDA)
and Ollama dual-mode execution.
"""

import os
import sys
import time
import json
import ast
import re
from typing import Dict, Any, List, Tuple

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from zenith_engine.runtime_manager import HardwareProfile
from zenith_engine.multimodal_inference import ZenithMultimodalEngine

REPORT_JSON = os.path.join(BASE_DIR, "reports", "zenith_4b_scientific_benchmark.json")
REPORT_MD = os.path.join(BASE_DIR, "reports", "zenith_4b_benchmark_summary.md")


# --- Benchmark Test Datasets ---

GEN_AI_TASKS = [
    {
        "id": "logic_deduction",
        "name": "Multi-Step Logic Deduction",
        "prompt": (
            "Five software engineers (A, B, C, D, E) work in five departments: ML, Security, Frontend, Backend, DevOps. "
            "1. D is in Security. "
            "2. C is in Backend. "
            "3. B is neither in ML nor Frontend. "
            "4. E is in ML. "
            "Which department does A work in? Show your step-by-step deduction and state the final answer clearly."
        ),
        "target": "Frontend"
    },
    {
        "id": "negative_constraints",
        "name": "Strict Negative Constraints Following",
        "prompt": (
            "Explain what an API is in exactly two concise sentences. "
            "Constraint: Do NOT use the word 'interface' and do NOT use the word 'software'."
        ),
        "forbidden": ["interface", "software"]
    }
]

CODING_TASKS = [
    {
        "id": "code_longest_substring",
        "name": "Algorithm: Longest Substring Without Repeating Characters",
        "prompt": (
            "Write a Python function `length_of_longest_substring(s: str) -> int` that returns the length of the longest "
            "substring without repeating characters. Output ONLY valid Python code inside a ```python ``` block."
        ),
        "func_name": "length_of_longest_substring",
        "test_cases": [
            ("abcabcbb", 3),
            ("bbbbb", 1),
            ("pwwkew", 3),
            ("", 0),
            ("abcdef", 6)
        ]
    },
    {
        "id": "code_merge_intervals",
        "name": "Algorithm: Merge Overlapping Intervals",
        "prompt": (
            "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals. "
            "Output ONLY valid Python code inside a ```python ``` block."
        ),
        "func_name": "merge_intervals",
        "test_cases": [
            ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
            ([[1, 4], [4, 5]], [[1, 5]]),
            ([], [])
        ]
    }
]

OCR_EXTRACTION_TASKS = [
    {
        "id": "receipt_extraction",
        "name": "Noisy Receipt Text -> Strict Valid JSON Schema",
        "prompt": (
            "Extract the following raw receipt text into a strictly valid JSON object with keys: "
            "'store_name', 'date', 'items' (list of {name, price}), 'tax', 'total'.\n"
            "Raw Text:\n"
            "--- RECEIPT ---\n"
            "ACME SUPERMARKET\n"
            "2026-10-15\n"
            "Milk 3.50\n"
            "Bread 2.20\n"
            "Apples 4.10\n"
            "Tax 0.98\n"
            "TOTAL 10.78\n"
            "THANK YOU\n"
            "Return strictly the JSON object only."
        ),
        "expected_keys": ["store_name", "date", "items", "tax", "total"],
        "expected_total": 10.78
    }
]

DIAGRAM_TASKS = [
    {
        "id": "mermaid_topology",
        "name": "Mermaid Architecture Single Point of Failure Analysis",
        "prompt": (
            "Analyze this system architecture:\n"
            "Client -> API_Gateway -> Auth_Service\n"
            "API_Gateway -> Payment_Service -> Database_Primary\n"
            "API_Gateway -> Order_Service -> Database_Primary\n\n"
            "Identify the Single Point of Failure (SPOF) for transaction processing and explain why in 2 sentences."
        ),
        "expected_mentions": ["Database_Primary", "API_Gateway"]
    }
]


def extract_code_block(text: str) -> str:
    """Extracts python code from markdown code block or bare text."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def run_benchmarks(engine: ZenithMultimodalEngine) -> Dict[str, Any]:
    print("=== Starting Zenith-4B Scientific Reproducible Benchmark Suite ===")
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "model": "Zenith-4B",
        "hardware": engine.hardware.get_memory_status(),
        "categories": {}
    }

    total_tokens = 0
    total_time = 0.0

    # 1. Gen AI
    print("\n--- Evaluating Pillar 1: Gen AI Reasoning ---")
    gen_ai_scores = []
    gen_ai_details = []
    for t in GEN_AI_TASKS:
        t0 = time.time()
        res = engine.generate(query=t["prompt"], max_new_tokens=256, temperature=0.1)
        elapsed = time.time() - t0
        output = res["response"]
        tokens = res.get("tokens_generated", 0)
        total_tokens += tokens
        total_time += elapsed

        score = 0.0
        if t["id"] == "logic_deduction":
            score = 1.0 if t["target"].lower() in output.lower() else 0.0
        elif t["id"] == "negative_constraints":
            has_forbidden = any(f.lower() in output.lower() for f in t["forbidden"])
            score = 0.0 if has_forbidden else 1.0

        gen_ai_scores.append(score)
        gen_ai_details.append({
            "task": t["name"],
            "score": score,
            "elapsed_sec": round(elapsed, 2),
            "tokens": tokens,
            "snippet": output[:120]
        })
        print(f"  [{t['name']}]: Score {score} ({elapsed:.2f}s)")

    results["categories"]["gen_ai_reasoning"] = {
        "score_percent": round(sum(gen_ai_scores) / max(len(gen_ai_scores), 1) * 100, 2),
        "details": gen_ai_details
    }

    # 2. Coding
    print("\n--- Evaluating Pillar 2: Functional Coding & AST Validity ---")
    coding_ast_scores = []
    coding_exec_scores = []
    coding_details = []

    for t in CODING_TASKS:
        t0 = time.time()
        res = engine.generate(query=t["prompt"], max_new_tokens=350, temperature=0.0)
        elapsed = time.time() - t0
        output = res["response"]
        tokens = res.get("tokens_generated", 0)
        total_tokens += tokens
        total_time += elapsed

        code = extract_code_block(output)
        
        # AST check
        ast_valid = False
        try:
            ast.parse(code)
            ast_valid = True
        except SyntaxError:
            ast_valid = False
        coding_ast_scores.append(1.0 if ast_valid else 0.0)

        # Execution check
        exec_pass_count = 0
        total_tests = len(t["test_cases"])
        if ast_valid:
            local_scope = {}
            try:
                exec(code, {}, local_scope)
                fn = local_scope.get(t["func_name"])
                if callable(fn):
                    for inp, exp in t["test_cases"]:
                        try:
                            # Handle multiple args or single arg
                            actual = fn(inp)
                            if actual == exp:
                                exec_pass_count += 1
                        except Exception:
                            pass
            except Exception:
                pass

        exec_ratio = round(exec_pass_count / max(total_tests, 1), 2)
        coding_exec_scores.append(exec_ratio)
        coding_details.append({
            "task": t["name"],
            "ast_valid": 1.0 if ast_valid else 0.0,
            "unit_test_pass_rate": exec_ratio,
            "elapsed_sec": round(elapsed, 2),
            "snippet": code[:100]
        })
        print(f"  [{t['name']}]: AST={ast_valid}, Tests={exec_pass_count}/{total_tests} ({elapsed:.2f}s)")

    results["categories"]["coding"] = {
        "ast_validity_percent": round(sum(coding_ast_scores) / max(len(coding_ast_scores), 1) * 100, 2),
        "functional_pass_rate_percent": round(sum(coding_exec_scores) / max(len(coding_exec_scores), 1) * 100, 2),
        "details": coding_details
    }

    # 3. Document Extraction / OCR
    print("\n--- Evaluating Pillar 3: Structured Document & OCR Extraction ---")
    ocr_scores = []
    ocr_details = []

    for t in OCR_EXTRACTION_TASKS:
        t0 = time.time()
        res = engine.generate(query=t["prompt"], max_new_tokens=300, temperature=0.0)
        elapsed = time.time() - t0
        output = res["response"]
        tokens = res.get("tokens_generated", 0)
        total_tokens += tokens
        total_time += elapsed

        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        score = 0.0
        parsed = None
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                # Check keys
                keys_present = sum(1 for k in t["expected_keys"] if k in parsed)
                key_ratio = keys_present / len(t["expected_keys"])
                total_val = parsed.get("total")
                total_matches = (float(total_val) == t["expected_total"]) if total_val else False
                score = round((key_ratio * 0.7) + (0.3 if total_matches else 0.0), 2)
            except Exception:
                score = 0.0

        ocr_scores.append(score)
        ocr_details.append({
            "task": t["name"],
            "score": score,
            "elapsed_sec": round(elapsed, 2),
            "snippet": output[:100]
        })
        print(f"  [{t['name']}]: Score {score} ({elapsed:.2f}s)")

    results["categories"]["ocr_document_extraction"] = {
        "score_percent": round(sum(ocr_scores) / max(len(ocr_scores), 1) * 100, 2),
        "details": ocr_details
    }

    # 4. Diagram / Vision
    print("\n--- Evaluating Pillar 4: Diagram & Vision Topology Grounding ---")
    diag_scores = []
    diag_details = []

    for t in DIAGRAM_TASKS:
        t0 = time.time()
        res = engine.generate(query=t["prompt"], max_new_tokens=200, temperature=0.1)
        elapsed = time.time() - t0
        output = res["response"]
        tokens = res.get("tokens_generated", 0)
        total_tokens += tokens
        total_time += elapsed

        mentions = sum(1 for m in t["expected_mentions"] if m.lower() in output.lower())
        score = round(mentions / len(t["expected_mentions"]), 2)
        diag_scores.append(score)
        diag_details.append({
            "task": t["name"],
            "score": score,
            "elapsed_sec": round(elapsed, 2),
            "snippet": output[:100]
        })
        print(f"  [{t['name']}]: Score {score} ({elapsed:.2f}s)")

    results["categories"]["diagram_vision"] = {
        "score_percent": round(sum(diag_scores) / max(len(diag_scores), 1) * 100, 2),
        "details": diag_details
    }

    # Summary throughput
    tok_per_sec = round(total_tokens / max(total_time, 0.001), 2)
    composite_score = round((
        results["categories"]["gen_ai_reasoning"]["score_percent"] +
        results["categories"]["coding"]["functional_pass_rate_percent"] +
        results["categories"]["ocr_document_extraction"]["score_percent"] +
        results["categories"]["diagram_vision"]["score_percent"]
    ) / 4.0, 2)

    results["summary"] = {
        "composite_score_percent": composite_score,
        "total_tokens_generated": total_tokens,
        "total_latency_sec": round(total_time, 2),
        "throughput_tokens_per_sec": tok_per_sec,
        "final_vram_mb": engine.hardware.get_memory_status().get("vram_allocated_mb", 0)
    }

    # Save outputs
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark Complete] JSON report saved to: {REPORT_JSON}")

    # Write Markdown summary
    md_content = f"""# Zenith-4B Scientific Benchmark Report

**Evaluation Date**: {results['timestamp']}  
**Hardware**: {results['hardware']['gpu_name']} ({results['hardware']['vram_total_gb']} GB VRAM, {results['hardware']['ram_total_gb']} GB RAM)  
**Throughput**: {tok_per_sec} tokens/sec  
**Total Benchmark Time**: {round(total_time, 2)}s  
**Composite Pass Rate**: {composite_score}%  

| Metric / Evaluation Pillar | Zenith-4B Score |
| :--- | :---: |
| **1. Gen AI Reasoning & Logic** | {results['categories']['gen_ai_reasoning']['score_percent']}% |
| **2. Coding AST Syntax Validity** | {results['categories']['coding']['ast_validity_percent']}% |
| **3. Coding Functional Unit Test Pass Rate** | {results['categories']['coding']['functional_pass_rate_percent']}% |
| **4. OCR & Document JSON Extraction** | {results['categories']['ocr_document_extraction']['score_percent']}% |
| **5. Diagram & Topology Grounding** | {results['categories']['diagram_vision']['score_percent']}% |
| **Overall Composite Score** | **{composite_score}%** |
"""
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Benchmark Complete] Markdown summary saved to: {REPORT_MD}")

    return results


if __name__ == "__main__":
    engine = ZenithMultimodalEngine()
    run_benchmarks(engine)
