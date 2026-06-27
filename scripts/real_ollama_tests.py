import requests
import json
import time

OLLAMA_HOST = "http://localhost:11434"

def test_model(model_name: str, prompt: str, images: list = None):
    print(f"\n" + "="*60)
    print(f"TESTING MODEL: {model_name}")
    print(f"PROMPT: '{prompt}'")
    print("="*60)
    
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    if images:
        payload["images"] = images
        
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            reply = data.get("response", "").strip()
            print(f"Generated response in {elapsed:.2f} seconds:")
            print("-" * 60)
            print(reply)
            print("-" * 60)
        else:
            print(f"Ollama returned error status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Failed to query Ollama API: {e}")

def main():
    print("Starting Live Real Ollama Query Tests...")
    
    # 1. Test Knowledge Expert
    test_model(
        model_name="qwen2.5:1.5b-instruct",
        prompt="Explain the difference between weather and climate in two simple sentences."
    )
    
    # 2. Test Coding Expert
    test_model(
        model_name="qwen2.5-coder:3b",
        prompt="Write a Python class for a BankAccount with deposit and withdraw methods. Keep it minimal."
    )
    
    # 3. Test Vision Expert (using 1x1 red PNG)
    red_dot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    test_model(
        model_name="riven/smolvlm:latest",
        prompt="What is the color of this image?",
        images=[red_dot_b64]
    )

if __name__ == "__main__":
    main()
