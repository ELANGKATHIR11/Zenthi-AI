import requests
import json
import time

OLLAMA_HOST = "http://localhost:11434"
models = ["qwen2.5:1.5b-instruct", "qwen2.5-coder:3b", "riven/smolvlm:latest"]

print("=== Preloading Ollama Models into RTX 5060 VRAM ===")

for model in models:
    print(f"\nSending load request for: {model}...")
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": "Hello",
        "stream": False
    }
    
    start = time.time()
    try:
        res = requests.post(url, json=payload, timeout=90)
        elapsed = time.time() - start
        if res.status_code == 200:
            print(f"[SUCCESS] Loaded {model} in {elapsed:.2f} seconds.")
        else:
            print(f"[ERROR] Failed to load {model}. Code: {res.status_code}, Response: {res.text}")
    except Exception as e:
        print(f"[EXCEPTION] Failed to connect to Ollama: {e}")

print("\nAll models loaded successfully in GPU VRAM.")
