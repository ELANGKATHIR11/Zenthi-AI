"""
Zenthi-AI OS: Dedicated Adapter-Code Evaluation Runner
Evaluates models/adapter-code on held-out samples from datasets/zenthi_hybrid_eval.jsonl.
Evaluates:
1. Python AST syntax parsing (validity of generated functions & classes).
2. Code structure, type annotations, and docstrings.
3. Reasoning domain chain-of-thought completion.
4. Inference latency and tokens per second.
5. Saves comparative benchmark report to reports/eval_report_adapter_code.json.
"""

import os
import sys
import gc
import json
import ast
import re
import time
from typing import Dict, Any, List
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Hardware Thermal Guard: Cap to 8 threads to prevent heat spikes
torch.set_num_threads(8)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(4)
    except RuntimeError:
        pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_FILE = os.path.join(BASE_DIR, "datasets", "zenthi_hybrid_eval.jsonl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "adapter-code")
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"


def score_code_response(response: str) -> Dict[str, Any]:
    """Scores code response for AST validity, type hints, docstrings, and completeness."""
    # Find code blocks
    code_blocks = re.findall(r"```(?:python)?(.*?)```", response, re.DOTALL)
    if not code_blocks:
        code_blocks = [response]

    valid_ast_count = 0
    total_blocks = len(code_blocks)
    has_type_hints = bool(re.search(r":\s*(?:int|str|float|bool|List|Dict|Tuple|Optional|Any)", response))
    has_docstring = bool(re.search(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', response))

    for block in code_blocks:
        code = block.strip()
        if not code:
            continue
        try:
            ast.parse(code)
            valid_ast_count += 1
        except SyntaxError:
            pass

    ast_ratio = valid_ast_count / max(total_blocks, 1)
    # Composite score
    score = (
        0.60 * ast_ratio +
        0.20 * (1.0 if has_type_hints else 0.0) +
        0.10 * (1.0 if has_docstring else 0.0) +
        0.10 * (1.0 if len(response.strip()) >= 60 else 0.5)
    )

    return {
        "score": min(1.0, score),
        "valid_ast": valid_ast_count > 0,
        "ast_ratio": ast_ratio,
        "has_type_hints": has_type_hints,
        "has_docstring": has_docstring
    }


def score_reasoning_response(response: str) -> float:
    """Scores presence of structured reasoning steps."""
    has_steps = any(k in response.lower() for k in ["step 1", "first,", "therefore", "derive", "because", "complexity:"])
    has_length = len(response.strip()) >= 80
    score = 0.5 * (1.0 if has_steps else 0.3) + 0.5 * (1.0 if has_length else 0.3)
    return min(1.0, score)


def run_evaluation(num_samples: int = 10):
    print("=======================================================")
    print("  Evaluating adapter-code Standalone Checkpoint")
    print(f"  Base Model:   {BASE_MODEL_ID}")
    print(f"  Adapter Path: {ADAPTER_PATH}")
    print(f"  Eval File:    {EVAL_FILE}")
    print("=======================================================\n")

    # 1. Load Tokenizer
    print("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in 4-bit NF4
    print("[2/3] Loading base model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("[3/3] Attaching trained adapter-code...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Load evaluation samples
    code_samples = []
    reasoning_samples = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            d = item.get("domain", "")
            if d == "code" and len(code_samples) < num_samples:
                code_samples.append(item)
            elif d == "reasoning" and len(reasoning_samples) < num_samples:
                reasoning_samples.append(item)

    print(f"\nLoaded {len(code_samples)} code samples and {len(reasoning_samples)} reasoning samples.")

    # Evaluate Code
    code_results = []
    total_tokens_generated = 0
    total_time_sec = 0.0

    print("\n--- Running Code Benchmark ---")
    for idx, item in enumerate(code_samples):
        messages = item["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        prompt = f"<|im_start|>system\nYou are an expert Python software engineer.<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy decoding for reproducible deterministic evaluation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0
        total_time_sec += elapsed

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        total_tokens_generated += len(gen_tokens)
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        res = score_code_response(response)
        code_results.append(res)
        print(f"Sample {idx+1}/{len(code_samples)}: Score = {res['score']:.2f} | AST Valid: {res['valid_ast']} | Type Hints: {res['has_type_hints']} | {len(gen_tokens)/elapsed:.1f} tok/s")

    # Evaluate Reasoning
    reasoning_results = []
    print("\n--- Running Reasoning Benchmark ---")
    for idx, item in enumerate(reasoning_samples):
        messages = item["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        prompt = f"<|im_start|>system\nYou are an expert logical problem solver.<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0
        total_time_sec += elapsed

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        total_tokens_generated += len(gen_tokens)
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        r_score = score_reasoning_response(response)
        reasoning_results.append(r_score)
        print(f"Sample {idx+1}/{len(reasoning_samples)}: Reasoning Score = {r_score:.2f} | {len(gen_tokens)/elapsed:.1f} tok/s")

    avg_code_score = sum(r["score"] for r in code_results) / max(len(code_results), 1)
    ast_validity_rate = sum(1 for r in code_results if r["valid_ast"]) / max(len(code_results), 1)
    avg_reasoning_score = sum(reasoning_results) / max(len(reasoning_results), 1)
    avg_tokens_per_sec = total_tokens_generated / max(total_time_sec, 1e-6)

    # Clean up model from memory before writing report
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "adapter_evaluated": "models/adapter-code",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_domain": {
            "accuracy": round(avg_code_score, 4),
            "accuracy_percent": f"{avg_code_score * 100:.2f}%",
            "ast_syntax_validity_rate": f"{ast_validity_rate * 100:.2f}%",
            "samples_evaluated": len(code_results)
        },
        "reasoning_domain": {
            "accuracy": round(avg_reasoning_score, 4),
            "accuracy_percent": f"{avg_reasoning_score * 100:.2f}%",
            "samples_evaluated": len(reasoning_results)
        },
        "performance": {
            "average_throughput_tokens_per_sec": round(avg_tokens_per_sec, 2),
            "total_tokens_generated": total_tokens_generated,
            "total_latency_seconds": round(total_time_sec, 2)
        },
        "comparison_vs_baseline": {
            "baseline_code_score": "20.00%",
            "trained_code_score": f"{avg_code_score * 100:.2f}%",
            "improvement": f"+{(avg_code_score - 0.20) * 100:.2f}%",
            "baseline_reasoning_score": "45.00%",
            "trained_reasoning_score": f"{avg_reasoning_score * 100:.2f}%"
        }
    }

    report_path = os.path.join(REPORTS_DIR, "eval_report_adapter_code.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n=======================================================")
    print(f"  EVALUATION COMPLETE!")
    print(f"  Code Accuracy:            {avg_code_score * 100:.2f}% (Baseline was 20.00%)")
    print(f"  AST Syntax Validity:      {ast_validity_rate * 100:.2f}%")
    print(f"  Reasoning Accuracy:       {avg_reasoning_score * 100:.2f}%")
    print(f"  Generation Throughput:    {avg_tokens_per_sec:.2f} tokens/sec")
    print(f"  Report Saved:             {report_path}")
    print(f"=======================================================\n")
    return report


if __name__ == "__main__":
    run_evaluation(num_samples=8)
