import torch
import os
import sys
import json
import time
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import ZenthiAIOrchestrator

def generate_benchmark_queries() -> Dict[str, List[str]]:
    """Programmatically generates 100 unique test queries for each intent category."""
    
    # 1. CODE queries (100 total)
    code_subjects = [
        "python", "javascript", "typescript", "c++", "java", "html", "css", 
        "sql query", "react component", "fastapi route", "mongodb aggregation", 
        "binary search tree", "bubble sort", "quicksort algorithm", "linked list",
        "regex pattern", "rest api", "express middleware", "dockerfile", "yaml config"
    ]
    code_templates = [
        "Write a {} function to {}",
        "How do I debug this {} error: {}",
        "Create a {} script that {}",
        "Explain the {} implementation of {}",
        "What is the best way to optimize {} code for {}"
    ]
    code_actions = [
        ("reverse a string", "index error"),
        ("find prime numbers", "null pointer exception"),
        ("fetch data from an API", "type error"),
        ("parse a JSON string", "syntax error"),
        ("implement a stack", "key error")
    ]
    code_queries = []
    for subj in code_subjects:
        for temp in code_templates:
            action = code_actions[len(code_queries) % len(code_actions)]
            code_queries.append(temp.format(subj, action[0] if "{}" in temp else action[1]))
            if len(code_queries) >= 100:
                break
        if len(code_queries) >= 100:
            break

    # 2. RAG queries (100 total)
    rag_contexts = [
        "the document", "the uploaded PDF", "my notes", "the reference paper",
        "the user manual", "the text file", "my notebook", "the context", "the PDF report", "this uploaded file"
    ]
    rag_questions = [
        "What is the main conclusion of {}?",
        "Can you summarize chapter 3 of {}?",
        "According to {}, what are the key findings?",
        "What does {} say about machine learning?",
        "Based on the text in {}, explain the launch timeline.",
        "Find the key metrics listed in {} Page 4.",
        "Compare the statistics in {} with standard benchmarks.",
        "Who is the author of {}?",
        "Is there a mention of security in {}?",
        "Summarize the limitations discussed in {}."
    ]
    rag_queries = []
    for ctx in rag_contexts:
        for q in rag_questions:
            rag_queries.append(q.format(ctx))
            
    # 3. SEARCH queries (100 total)
    search_topics = [
        "weather", "news", "stock price", "sports score", "latest release",
        "current exchange rate", "recent launch", "today's price of", "live updates", "recent events"
    ]
    search_entities = [
        "Apple AAPL", "SpaceX Falcon", "New York City", "Bitcoin BTC", "Gold",
        "Python 3.13", "Microsoft MSFT", "FIFA World Cup", "Tesla TSLA", "OpenAI GPT-5"
    ]
    search_templates = [
        "What is the {} of {} today?",
        "Find the {} for {} right now.",
        "Give me the {} regarding {} news.",
        "Show me the {} of {} for this week.",
        "What are the {} developments for {}?"
    ]
    search_queries = []
    for topic in search_topics:
        for entity in search_entities:
            temp = search_templates[len(search_queries) % len(search_templates)]
            search_queries.append(temp.format(topic, entity))

    # 4. VISION queries (100 total)
    vision_objects = [
        "this image", "the chart", "the diagram", "this screenshot", "the graph",
        "the photo", "the sketch", "this picture", "the visual layout", "the table image"
    ]
    vision_prompts = [
        "Describe what is depicted in {}.",
        "Analyze the data shown in {}.",
        "What elements are present in {}?",
        "Can you transcribe the text/OCR in {}?",
        "What is the relationship between nodes in {}?",
        "Explain the workflow shown in the diagram of {}.",
        "Identify any anomalies inside {}.",
        "What is the color palette used in {}?",
        "What are the labels on the x-axis of {}?",
        "Can you read the table contents from {}?"
    ]
    vision_queries = []
    for obj in vision_objects:
        for prompt in vision_prompts:
            vision_queries.append(prompt.format(obj))

    # 5. KNOWLEDGE queries (100 total)
    knowledge_topics = [
        "the French Revolution", "photosynthesis", "quantum mechanics", "artificial intelligence",
        "the solar system", "world history", "economics", "organic chemistry", "philosophy", "music theory"
    ]
    knowledge_questions = [
        "Explain the concept of {} in simple terms.",
        "What are the main stages of {}?",
        "Who were the key figures in the history of {}?",
        "Give me a detailed report on the significance of {}.",
        "What are the differences between classical theories and modern concepts of {}?",
        "Write an essay about the ethical implications of {} in society.",
        "How does {} affect global ecosystems?",
        "Summarize the key principles of {}.",
        "What is the mathematical formulation behind {}?",
        "Explain how {} evolved over the 20th century."
    ]
    knowledge_queries = []
    for topic in knowledge_topics:
        for q in knowledge_questions:
            knowledge_queries.append(q.format(topic))

    return {
        "CODE": code_queries[:100],
        "RAG":   rag_queries[:100],
        "SEARCH": search_queries[:100],
        "VISION": vision_queries[:100],
        "KNOWLEDGE": knowledge_queries[:100]
    }

def run_evaluation():
    print("=== Zenthi-AI OS Router Accuracy Benchmark ===")
    print("Initializing Orchestrator and Router Agent...")
    
    orchestrator = ZenthiAIOrchestrator()
    dataset = generate_benchmark_queries()
    
    total_queries = 0
    correct_routes = 0
    
    # Store metrics per class
    metrics = {
        "CODE": {"correct": 0, "total": 100},
        "RAG": {"correct": 0, "total": 100},
        "SEARCH": {"correct": 0, "total": 100},
        "VISION": {"correct": 0, "total": 100},
        "KNOWLEDGE": {"correct": 0, "total": 100}
    }
    
    start_time = time.time()
    
    print("\nRunning semantic routing classification evaluation on 500 test cases...")
    
    for expected_intent, queries in dataset.items():
        print(f"Evaluating {expected_intent} category...")
        for q in queries:
            # We pass has_images=True for vision queries to represent frontend visual payload presence
            has_images = (expected_intent == "VISION")
            
            # Run router only to speed up evaluation (avoiding full LLM execution times)
            routing_res = orchestrator.router.route(q, has_images=has_images)
            predicted_intent = routing_res["intent"]
            
            # Map predictions to expected labels
            is_correct = (predicted_intent == expected_intent)
            
            if is_correct:
                metrics[expected_intent]["correct"] += 1
                correct_routes += 1
            total_queries += 1
            
    end_time = time.time()
    elapsed = end_time - start_time
    
    overall_accuracy = (correct_routes / total_queries) * 100
    
    print("\n=== BENCHMARK CLASSIFICATION REPORT ===")
    print(f"Total Test Cases Evaluated : {total_queries}")
    print(f"Total Time Taken           : {elapsed:.2f} seconds")
    print(f"Average Latency per Query  : {(elapsed/total_queries)*1000:.2f} ms")
    print(f"Overall Routing Accuracy   : {overall_accuracy:.2f}%\n")
    
    print("| Intent Category | Correct Routes | Total Cases | Accuracy (%) |")
    print("|-----------------|----------------|-------------|--------------|")
    for category, stats in metrics.items():
        acc = (stats["correct"] / stats["total"]) * 100
        print(f"| {category:<15} | {stats['correct']:<14} | {stats['total']:<11} | {acc:<12.2f}% |")
        
    # Generate report artifact
    report_content = f"""# Router Classification Accuracy Report

This automated test report details the semantic routing accuracy of **Zenthi-AI OS** across **500 evaluation test queries** (100 queries per intent category).

## Benchmark Performance Summary

- **Total Queries Evaluated**: {total_queries}
- **Evaluation Time**: {elapsed:.2f} seconds
- **Average Latency**: {(elapsed/total_queries)*1000:.2f} ms
- **Overall Routing Accuracy**: {overall_accuracy:.2f}%

## Category Performance Breakdown

| Intent Category | Correct / Total | Category Accuracy | Target Expert Model |
| :--- | :--- | :--- | :--- |
| **CODE** | {metrics["CODE"]["correct"]} / 100 | {(metrics["CODE"]["correct"]/100)*100:.2f}% | `qwen2.5-coder:3b` |
| **RAG** | {metrics["RAG"]["correct"]} / 100 | {(metrics["RAG"]["correct"]/100)*100:.2f}% | `qwen2.5:1.5b-instruct` |
| **SEARCH** | {metrics["SEARCH"]["correct"]} / 100 | {(metrics["SEARCH"]["correct"]/100)*100:.2f}% | `qwen2.5:1.5b-instruct` |
| **VISION** | {metrics["VISION"]["correct"]} / 100 | {(metrics["VISION"]["correct"]/100)*100:.2f}% | `riven/smolvlm:latest` |
| **KNOWLEDGE** | {metrics["KNOWLEDGE"]["correct"]} / 100 | {(metrics["KNOWLEDGE"]["correct"]/100)*100:.2f}% | `qwen2.5:1.5b-instruct` |

*Report generated at run_evaluation execution completion.*
"""
    
    # Save report inside evaluation folder
    os.makedirs("F:\\Zenith-AI\\reports", exist_ok=True)
    with open("F:\\Zenith-AI\\reports\\router_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\nSaved markdown report to: F:\\Zenith-AI\\reports\\router_evaluation_report.md")

if __name__ == "__main__":
    run_evaluation()
