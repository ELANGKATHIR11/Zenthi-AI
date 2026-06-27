import torch
import os
import sys
import base64
import requests
import json
import time

API_BASE = "http://localhost:8000"

def generate_base64_image():
    # 1x1 Red Dot PNG
    red_dot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    return red_dot_b64

def check_health():
    print("\n[TEST 1] Checking backend health status...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print("Status Code:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print("Health Check Failed:", e)
        return False

def test_code_generation():
    print("\n[TEST 2] Testing Code Generation (Coding Expert)...")
    payload = {
        "query": "Write a python function to compute the factorial of a number recursively.",
        "mode": "AUTO"
    }
    try:
        response = requests.post(f"{API_BASE}/chat", json=payload)
        print("Status Code:", response.status_code)
        res_data = response.json()
        print("Mode Assigned:", res_data.get("mode"))
        print("Workflow Steps:", res_data.get("workflow_steps"))
        print("Response Snippet:\n", res_data.get("response")[:300], "...")
    except Exception as e:
        print("Code Generation Test Failed:", e)

def test_document_understanding():
    print("\n[TEST 3] Testing Document Upload and Ingestion (RAG Expert)...")
    doc_path = "F:\\Zenith-AI\\sample_test_doc.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("Zenthi-AI OS launch date is scheduled for September 15, 2026. The coordinator is Dr. Elena.")
    
    try:
        # Ingest
        with open(doc_path, "rb") as f:
            files = {"file": (os.path.basename(doc_path), f, "text/plain")}
            up_res = requests.post(f"{API_BASE}/upload", files=files)
            print("Upload Status Code:", up_res.status_code)
            print("Upload Response:", up_res.json())
            
        # Wait a bit for indexing
        time.sleep(2)
        
        # Query
        payload = {
            "query": "When is the launch date of Zenthi-AI OS and who is the coordinator?",
            "mode": "RAG"
        }
        chat_res = requests.post(f"{API_BASE}/chat", json=payload)
        print("Chat Query Status Code:", chat_res.status_code)
        res_data = chat_res.json()
        print("Mode Assigned:", res_data.get("mode"))
        print("Citations:", res_data.get("citations"))
        print("Workflow Steps:", res_data.get("workflow_steps"))
        print("Response:\n", res_data.get("response"))
    except Exception as e:
        print("Document Ingestion/Query Test Failed:", e)
    finally:
        if os.path.exists(doc_path):
            os.remove(doc_path)

def test_web_search():
    print("\n[TEST 4] Testing Web Search Integration (Search Expert)...")
    payload = {
        "query": "What are the latest news details about space exploration today?",
        "mode": "SEARCH"
    }
    try:
        response = requests.post(f"{API_BASE}/chat", json=payload)
        print("Status Code:", response.status_code)
        res_data = response.json()
        print("Mode Assigned:", res_data.get("mode"))
        print("Citations:", res_data.get("citations"))
        print("Workflow Steps:", res_data.get("workflow_steps"))
        print("Response Snippet:\n", res_data.get("response")[:400], "...")
    except Exception as e:
        print("Web Search Test Failed:", e)

def test_image_understanding():
    print("\n[TEST 5] Testing Image Understanding (Vision Expert)...")
    img_b64 = generate_base64_image()
    payload = {
        "query": "Describe what you see in this image.",
        "mode": "AUTO",
        "images": [img_b64]
    }
    try:
        response = requests.post(f"{API_BASE}/chat", json=payload)
        print("Status Code:", response.status_code)
        res_data = response.json()
        print("Mode Assigned:", res_data.get("mode"))
        print("Workflow Steps:", res_data.get("workflow_steps"))
        print("Response:\n", res_data.get("response"))
    except Exception as e:
        print("Vision Test Failed:", e)

def main():
    print("=== STARTING END-TO-END SYSTEM TESTS ===")
    if not check_health():
        print("Backend server is not running or degraded. Aborting tests.")
        return
        
    test_code_generation()
    test_document_understanding()
    test_web_search()
    test_image_understanding()
    print("\n=== SYSTEM TESTS COMPLETED ===")

if __name__ == "__main__":
    main()
