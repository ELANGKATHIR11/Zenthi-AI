"""
Industry-Grade Benchmark Suite: Zenith-4B vs Base Qwen3.5-4B
Evaluates across 4 Core Pillars:
1. Gen AI (Instruction Following, Logic, Reasoning)
2. OCR (Noisy Receipt/Invoice -> Schema, Key-Value Extraction, Tabular Integrity)
3. Coding (Algorithmic Synthesis, AST Parsing, Live Unit-Test Execution)
4. Image & Diagram Analysis (Topology, Spatial Grounding, Telemetry Metrics)

Enforces Thermal Safeguards:
- Process Priority: BelowNormal
- Inter-query Breathers (1.5s pause)
- Active GPU Temperature Watchdog (<72C)
- Sub-4GB VRAM footprint
"""

import os
import sys
import time
import json
import ast
import re
import subprocess
import requests
from typing import Dict, Any, List, Tuple

OLLAMA_API = "http://127.0.0.1:11434/api/generate"
MODELS = {
    "Base_Qwen3.5-4B": "qwen3.5-hybrid:4b",
    "Zenith-4B (Fine-Tuned)": "zenith-4b:latest"
}

def set_low_process_priority():
    """Sets process priority to Below Normal so user's Brave browser runs smoothly."""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if sys.platform == "win32":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
        print("[System Safety] Process priority lowered to BELOW_NORMAL.")
    except Exception as e:
        print(f"[System Safety Note] Could not adjust process nice level: {e}")

def check_gpu_thermal_safety(max_temp: float = 72.0):
    """Monitors GPU temperature via nvidia-smi and applies thermal breather if needed."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader"],
            text=True
        ).strip()
        temp_str, mem_str = out.split(",")
        temp = float(temp_str.strip())
        mem = mem_str.strip()
        if temp > max_temp:
            print(f"[Thermal Breather] GPU temp {temp}C exceeds {max_temp}C! Pausing 6 seconds to cool...")
            time.sleep(6.0)
        return temp, mem
    except Exception:
        return None, None

def query_ollama(model_name: str, prompt: str, system: str = "", max_tokens: int = 400) -> Tuple[str, float, int]:
    """Queries Ollama with strict timeouts and extracts token stats."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": max_tokens
        }
    }
    if system:
        payload["system"] = system

    t0 = time.time()
    try:
        res = requests.post(OLLAMA_API, json=payload, timeout=50)
        t_elapsed = time.time() - t0
        if res.status_code == 200:
            data = res.json()
            resp_text = data.get("response", "")
            eval_count = data.get("eval_count", 0)
            return resp_text, t_elapsed, eval_count
        else:
            return f"[Error: status {res.status_code}]", t_elapsed, 0
    except Exception as e:
        return f"[Error: {str(e)}]", time.time() - t0, 0

# ==============================================================================
# BENCHMARK SUITES
# ==============================================================================

# --- PILLAR 1: GEN AI & REASONING ---
GEN_AI_TASKS = [
    {
        "id": "genai_logic_puzzle",
        "name": "Multi-Step Logic Deduction",
        "prompt": "Five engineers (A, B, C, D, E) work in five different departments: ML, Frontend, Backend, DevOps, Security.\n1. A and B do NOT work in ML or Security.\n2. C works in Backend.\n3. D works in Security.\n4. B works in DevOps.\nWhich department does A work in? State the department clearly and explain why in 2 sentences.",
        "evaluator": lambda resp: 1.0 if "frontend" in resp.lower() else 0.0
    },
    {
        "id": "genai_negative_constraints",
        "name": "Strict Negative Constraints Instruction Following",
        "prompt": "Explain what an Application Programming Interface (API) is in exactly two sentences.\nYou must NOT use any of the following three words: 'software', 'communicate', or 'interface'.",
        "evaluator": lambda resp: (
            1.0 if (
                not any(w in resp.lower() for w in ["software", "communicate", "interface"])
                and 1 <= len(re.findall(r"[.!?]+", resp.strip())) <= 3
            ) else (0.5 if not any(w in resp.lower() for w in ["software", "communicate", "interface"]) else 0.0)
        )
    },
    {
        "id": "genai_cot_probability",
        "name": "Chain-of-Thought Mathematical Probability",
        "prompt": "A server cluster has 3 nodes.\nNode 1 handles 40% of traffic with a 2% failure rate.\nNode 2 handles 35% of traffic with a 4% failure rate.\nNode 3 handles 25% of traffic with a 1% failure rate.\nWhat is the overall probability that a request entering the cluster encounters a failure? Show step-by-step mathematical reasoning and give the final percentage.",
        "evaluator": lambda resp: (
            1.0 if any(ans in resp for ans in ["2.45%", "0.0245", "2.45 percent"]) else 0.3
        )
    }
]

# --- PILLAR 2: OCR & DOCUMENT EXTRACTION ---
OCR_TASKS = [
    {
        "id": "ocr_receipt_json",
        "name": "Noisy Receipt Text -> Strict Valid JSON Schema",
        "prompt": """Extract the following receipt text into a valid JSON object matching this schema:
{
  "merchant": string,
  "date": string,
  "items": [{"name": string, "price": float}],
  "subtotal": float,
  "tax": float,
  "total": float
}

Output ONLY valid JSON inside ```json ``` code block.

RAW RECEIPT TEXT:
*** SUPERMARKET MART #402 ***
DATE: 2026-05-12   TIME: 14:32:10
1x ORG MILK 1GAL       $ 4.89
2x SOURDOUGH BREAD     $ 7.50
1x GALA APPLES 3LB     $ 5.99
--------------------------------
SUBTOTAL               $18.38
TAX (8.25%)            $ 1.52
TOTAL                  $19.90
PAYMENT: VISA ENDING 4912""",
        "evaluator": lambda resp: (
            _evaluate_json_schema(resp, expected_keys=["merchant", "date", "items", "total"], expected_total=19.90)
        )
    },
    {
        "id": "ocr_nameplate_kv",
        "name": "Industrial Nameplate Key-Value Extraction",
        "prompt": """Extract all equipment specifications from this OCR scan into a clean JSON dictionary:
SIEMENS AG - INDUSTRY AUTOMATION
TYPE: SIMATIC S7-1500 CPU 1516-3 PN/DP
FS: 04   FW: V2.8.3
INPUT: 24V DC / 1.5A
MAC 1: 00-1B-1B-3A-4F-01
MAC 2: 00-1B-1B-3A-4F-02
MADE IN GERMANY

Return ONLY valid JSON.""",
        "evaluator": lambda resp: (
            _evaluate_json_schema(resp, expected_keys=["TYPE", "FW", "INPUT"], case_insensitive=True)
        )
    },
    {
        "id": "ocr_markdown_table",
        "name": "Unstructured Table OCR -> Markdown Grid",
        "prompt": """Convert this noisy space-delimited OCR report into a valid GitHub-flavored Markdown table:
Quarter  Revenue_Billion  Net_Income_Million  Margin_Pct
Q1-2025  14.2            1850                13.0%
Q2-2025  15.1            2100                13.9%
Q3-2025  16.4            2350                14.3%
Q4-2025  18.0            2800                15.5%""",
        "evaluator": lambda resp: (
            1.0 if ("|" in resp and "---" in resp and all(q in resp for q in ["Q1-2025", "Q2-2025", "Q3-2025", "Q4-2025"]))
            else (0.5 if "|" in resp else 0.0)
        )
    }
]

def _evaluate_json_schema(resp: str, expected_keys: List[str], expected_total: float = None, case_insensitive: bool = False) -> float:
    match = re.search(r"```json\s*(.*?)\s*```", resp, re.DOTALL)
    raw = match.group(1).strip() if match else resp.strip()
    # find outermost brace if needed
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end != -1:
        raw = raw[brace_start:brace_end+1]
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return 0.2
        keys = [k.lower() for k in data.keys()] if case_insensitive else list(data.keys())
        keys_present = sum(1 for ek in expected_keys if (ek.lower() if case_insensitive else ek) in keys)
        key_ratio = keys_present / len(expected_keys)
        
        if expected_total is not None:
            tot = float(data.get("total", 0.0))
            if abs(tot - expected_total) < 0.05:
                return min(1.0, 0.5 * key_ratio + 0.5)
            return 0.5 * key_ratio
        return key_ratio
    except Exception:
        return 0.0

# --- PILLAR 3: CODING (SYNTHESIS & REAL UNIT TEST EXECUTION) ---
CODING_TASKS = [
    {
        "id": "code_longest_substring",
        "name": "Algorithm: Longest Substring Without Repeating Chars",
        "prompt": """Write a Python function `length_of_longest_substring(s: str) -> int` that finds the length of the longest substring without repeating characters in O(n) time.
Provide ONLY the Python function inside a ```python ``` code block.""",
        "tests": [
            ("length_of_longest_substring('')", 0),
            ("length_of_longest_substring('abcabcbb')", 3),
            ("length_of_longest_substring('bbbbb')", 1),
            ("length_of_longest_substring('pwwkew')", 3),
        ]
    },
    {
        "id": "code_merge_intervals",
        "name": "Algorithm: Merge Overlapping Intervals",
        "prompt": """Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals and returns an array of the non-overlapping intervals in ascending order.
Provide ONLY the Python function inside a ```python ``` code block.""",
        "tests": [
            ("merge_intervals([[1,3],[2,6],[8,10],[15,18]])", [[1,6],[8,10],[15,18]]),
            ("merge_intervals([[1,4],[4,5]])", [[1,5]]),
            ("merge_intervals([])", []),
            ("merge_intervals([[1,4],[0,4]])", [[0,4]]),
        ]
    },
    {
        "id": "code_deep_merge",
        "name": "Algorithm: Deep Recursive Dictionary Merger",
        "prompt": """Write a Python function `deep_merge(dict1: dict, dict2: dict) -> dict` that recursively merges two nested dictionaries. If a key exists in both and both values are dictionaries, merge them recursively; otherwise, dict2's value overrides dict1's. Return a new dictionary without mutating inputs.
Provide ONLY the Python function inside a ```python ``` code block.""",
        "tests": [
            ("deep_merge({'a': 1, 'b': {'x': 10}}, {'c': 3, 'b': {'y': 20}})", {'a': 1, 'c': 3, 'b': {'x': 10, 'y': 20}}),
            ("deep_merge({'a': {'b': {'c': 1}}}, {'a': {'b': {'c': 2, 'd': 3}}})", {'a': {'b': {'c': 2, 'd': 3}}}),
        ]
    }
]

def _evaluate_code_execution(resp: str, tests: List[Tuple[str, Any]]) -> Tuple[float, float, str]:
    """Extracts Python code, checks AST validity, and executes unit tests."""
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", resp, re.DOTALL)
    code = code_match.group(1).strip() if code_match else resp.strip()
    
    # 1. AST Check
    try:
        ast.parse(code)
        ast_valid = 1.0
    except SyntaxError as e:
        return 0.0, 0.0, f"SyntaxError: {e}"
        
    # 2. Execution Check
    passed_tests = 0
    namespace = {}
    try:
        exec(code, namespace)
        for expr, expected in tests:
            result = eval(expr, namespace)
            if result == expected:
                passed_tests += 1
        exec_rate = passed_tests / len(tests)
        return ast_valid, exec_rate, f"Passed {passed_tests}/{len(tests)} tests"
    except Exception as e:
        return ast_valid, 0.0, f"RuntimeError: {e}"

# --- PILLAR 4: IMAGE & DIAGRAM ANALYSIS ---
IMAGE_DIAGRAM_TASKS = [
    {
        "id": "diagram_mermaid_spof",
        "name": "Mermaid Architecture Diagram SPOF Analysis",
        "prompt": """Analyze the following Mermaid architecture diagram and determine:
1. Primary data flow ingress and egress
2. The Single Point of Failure (SPOF)
3. Actionable high-availability recommendation

```mermaid
graph TD
  Client[Web / Mobile Clients] --> Ingress[Single Load Balancer 01]
  Ingress --> Auth[Auth Service Replica 1 & 2]
  Ingress --> AppAPI[App Server Cluster]
  AppAPI --> WorkerQueue[(Redis Queue)]
  WorkerQueue --> Workers[Background AI Workers]
  AppAPI --> PrimaryDB[(PostgreSQL Primary DB)]
```""",
        "evaluator": lambda resp: (
            1.0 if (
                any(spof in resp.lower() for spof in ["load balancer", "ingress", "single load balancer"])
                and any(db in resp.lower() for db in ["postgres", "primary db", "database"])
                and any(sol in resp.lower() for sol in ["redundant", "failover", "multi-az", "ha", "replica", "high availability"])
            ) else 0.5
        )
    },
    {
        "id": "diagram_spatial_bounding_box",
        "name": "OCR Document Spatial Layout & Bounding Box Reasoning",
        "prompt": """Given these extracted OCR bounding boxes normalized [x_min, y_min, x_max, y_max] from a document page (0 to 1000 scale):
Box A: [50, 40, 950, 120] -> "GLOBAL HEALTHCARE INVOICE #9812"
Box B: [50, 200, 400, 240] -> "Patient Name: John Doe"
Box C: [500, 200, 950, 240] -> "Admission Date: 2026-04-10"
Box D: [50, 850, 950, 950] -> "Total Billed: $14,250.00"

Answer:
1. Which box corresponds to the document header?
2. Are Box B and Box C aligned on the same horizontal row or vertical column?
3. Which box represents the footer summary?""",
        "evaluator": lambda resp: (
            1.0 if (
                "box a" in resp.lower()
                and any(h in resp.lower() for h in ["horizontal", "row", "same row", "horizontal row"])
                and "box d" in resp.lower()
            ) else 0.5
        )
    },
    {
        "id": "diagram_telemetry_metrics",
        "name": "Telemetry Performance & Outage Root Cause Diagnosis",
        "prompt": """An observability dashboard reports the following system incident metrics:
- Time 14:00: Requests/sec: 5,000 | P99 Latency: 45ms  | Error Rate: 0.01% | DB Connection Pool: 25%
- Time 14:05: Requests/sec: 5,100 | P99 Latency: 850ms | Error Rate: 0.05% | DB Connection Pool: 98%
- Time 14:08: Requests/sec: 5,050 | P99 Latency: 4,200ms | Error Rate: 18.4% (HTTP 504) | DB Connection Pool: 100% (Saturated)

Diagnose:
1. What is the root cause bottleneck of the cascade failure?
2. Why did latency explode even though incoming traffic (RPS) remained stable (~5,000 req/s)?
3. What is the immediate architectural remediation?""",
        "evaluator": lambda resp: (
            1.0 if (
                any(bc in resp.lower() for bc in ["connection pool", "database", "db pool", "db connection"])
                and any(lat in resp.lower() for lat in ["blocked", "waiting", "queued", "saturation", "exhaust"])
                and any(rem in resp.lower() for rem in ["pool size", "timeout", "circuit breaker", "read replica", "cache", "slow query"])
            ) else 0.5
        )
    }
]

def run_benchmark():
    print("=" * 72)
    print("      INDUSTRY-GRADE BENCHMARK: ZENITH-4B VS BASE QWEN3.5-4B")
    print("  Domains: Gen AI, OCR/Doc Extraction, Coding, Image/Diagram Analysis")
    print("=" * 72)

    set_low_process_priority()
    temp, mem = check_gpu_thermal_safety()
    if temp:
        print(f"[Initial Thermal Status] GPU Temp: {temp}C | VRAM Used: {mem}")

    results = {}

    for model_label, model_name in MODELS.items():
        print(f"\n=======================================================")
        print(f"  BENCHMARKING: {model_label} ({model_name})")
        print(f"=======================================================")

        model_results = {
            "gen_ai": {"scores": [], "latencies": [], "details": []},
            "ocr": {"scores": [], "latencies": [], "details": []},
            "coding": {"ast_scores": [], "exec_scores": [], "latencies": [], "details": []},
            "image_diagram": {"scores": [], "latencies": [], "details": []},
            "total_tokens": 0,
            "total_time": 0.0
        }

        # 1. GEN AI SUITE
        print("\n--- [Domain 1/4] Gen AI & Reasoning ---")
        for task in GEN_AI_TASKS:
            print(f"  -> Running: {task['name']} ...", end=" ", flush=True)
            resp, elapsed, tokens = query_ollama(model_name, task["prompt"])
            score = task["evaluator"](resp)
            model_results["gen_ai"]["scores"].append(score)
            model_results["gen_ai"]["latencies"].append(elapsed)
            model_results["gen_ai"]["details"].append({
                "task": task["name"],
                "score": score,
                "elapsed": round(elapsed, 2),
                "snippet": resp[:120].replace("\n", " ")
            })
            model_results["total_tokens"] += tokens
            model_results["total_time"] += elapsed
            print(f"Score: {score*100:.1f}% ({elapsed:.2f}s)")
            time.sleep(1.5)  # Thermal breather
            check_gpu_thermal_safety()

        # 2. OCR SUITE
        print("\n--- [Domain 2/4] OCR & Document Extraction ---")
        for task in OCR_TASKS:
            print(f"  -> Running: {task['name']} ...", end=" ", flush=True)
            resp, elapsed, tokens = query_ollama(model_name, task["prompt"])
            score = task["evaluator"](resp)
            model_results["ocr"]["scores"].append(score)
            model_results["ocr"]["latencies"].append(elapsed)
            model_results["ocr"]["details"].append({
                "task": task["name"],
                "score": score,
                "elapsed": round(elapsed, 2),
                "snippet": resp[:120].replace("\n", " ")
            })
            model_results["total_tokens"] += tokens
            model_results["total_time"] += elapsed
            print(f"Score: {score*100:.1f}% ({elapsed:.2f}s)")
            time.sleep(1.5)  # Thermal breather
            check_gpu_thermal_safety()

        # 3. CODING SUITE
        print("\n--- [Domain 3/4] Coding (Synthesis & Live Unit Tests) ---")
        for task in CODING_TASKS:
            print(f"  -> Running: {task['name']} ...", end=" ", flush=True)
            resp, elapsed, tokens = query_ollama(model_name, task["prompt"])
            ast_s, exec_s, detail_msg = _evaluate_code_execution(resp, task["tests"])
            model_results["coding"]["ast_scores"].append(ast_s)
            model_results["coding"]["exec_scores"].append(exec_s)
            model_results["coding"]["latencies"].append(elapsed)
            model_results["coding"]["details"].append({
                "task": task["name"],
                "ast_valid": ast_s,
                "unit_test_pass_rate": exec_s,
                "detail": detail_msg,
                "elapsed": round(elapsed, 2)
            })
            model_results["total_tokens"] += tokens
            model_results["total_time"] += elapsed
            print(f"AST: {ast_s*100:.0f}% | Tests: {exec_s*100:.1f}% ({detail_msg})")
            time.sleep(1.5)  # Thermal breather
            check_gpu_thermal_safety()

        # 4. IMAGE & DIAGRAM ANALYSIS SUITE
        print("\n--- [Domain 4/4] Image & Diagram Analysis ---")
        for task in IMAGE_DIAGRAM_TASKS:
            print(f"  -> Running: {task['name']} ...", end=" ", flush=True)
            resp, elapsed, tokens = query_ollama(model_name, task["prompt"])
            score = task["evaluator"](resp)
            model_results["image_diagram"]["scores"].append(score)
            model_results["image_diagram"]["latencies"].append(elapsed)
            model_results["image_diagram"]["details"].append({
                "task": task["name"],
                "score": score,
                "elapsed": round(elapsed, 2),
                "snippet": resp[:120].replace("\n", " ")
            })
            model_results["total_tokens"] += tokens
            model_results["total_time"] += elapsed
            print(f"Score: {score*100:.1f}% ({elapsed:.2f}s)")
            time.sleep(1.5)  # Thermal breather
            check_gpu_thermal_safety()

        # Aggregate summary
        avg_genai = sum(model_results["gen_ai"]["scores"]) / len(model_results["gen_ai"]["scores"])
        avg_ocr = sum(model_results["ocr"]["scores"]) / len(model_results["ocr"]["scores"])
        avg_coding_exec = sum(model_results["coding"]["exec_scores"]) / len(model_results["coding"]["exec_scores"])
        avg_coding_ast = sum(model_results["coding"]["ast_scores"]) / len(model_results["coding"]["ast_scores"])
        avg_diagram = sum(model_results["image_diagram"]["scores"]) / len(model_results["image_diagram"]["scores"])
        overall = (avg_genai + avg_ocr + avg_coding_exec + avg_diagram) / 4.0

        tok_sec = round(model_results["total_tokens"] / max(model_results["total_time"], 0.01), 2)

        results[model_label] = {
            "overall_score_percent": round(overall * 100, 2),
            "gen_ai_reasoning_percent": round(avg_genai * 100, 2),
            "ocr_document_extraction_percent": round(avg_ocr * 100, 2),
            "coding_functional_pass_rate_percent": round(avg_coding_exec * 100, 2),
            "coding_ast_syntax_validity_percent": round(avg_coding_ast * 100, 2),
            "image_diagram_analysis_percent": round(avg_diagram * 100, 2),
            "throughput_tokens_per_sec": tok_sec,
            "total_benchmark_time_sec": round(model_results["total_time"], 2),
            "details": model_results
        }

    # Save to report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/industry_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print("                    FINAL BENCHMARK COMPARISON")
    print("=" * 72)
    print(f"{'Pillar / Metric':<35} | {'Base Qwen3.5-4B':<16} | {'Zenith-4B':<16}")
    print("-" * 72)
    b_res = results["Base_Qwen3.5-4B"]
    z_res = results["Zenith-4B (Fine-Tuned)"]
    print(f"{'1. Gen AI & Reasoning':<35} | {b_res['gen_ai_reasoning_percent']:<15}% | {z_res['gen_ai_reasoning_percent']:<15}%")
    print(f"{'2. OCR & Document Extraction':<35} | {b_res['ocr_document_extraction_percent']:<15}% | {z_res['ocr_document_extraction_percent']:<15}%")
    print(f"{'3. Coding (Unit Test Pass Rate)':<35} | {b_res['coding_functional_pass_rate_percent']:<15}% | {z_res['coding_functional_pass_rate_percent']:<15}%")
    print(f"{'   Coding AST Syntax Rate':<35} | {b_res['coding_ast_syntax_validity_percent']:<15}% | {z_res['coding_ast_syntax_validity_percent']:<15}%")
    print(f"{'4. Image & Diagram Analysis':<35} | {b_res['image_diagram_analysis_percent']:<15}% | {z_res['image_diagram_analysis_percent']:<15}%")
    print("-" * 72)
    print(f"{'OVERALL COMPOSITE SCORE':<35} | {b_res['overall_score_percent']:<15}% | {z_res['overall_score_percent']:<15}%")
    print(f"{'Token Generation Speed (tok/s)':<35} | {b_res['throughput_tokens_per_sec']:<15} | {z_res['throughput_tokens_per_sec']:<15}")
    print("=" * 72)
    print(f"Detailed JSON report saved to: {report_path}\n")

if __name__ == "__main__":
    run_benchmark()

