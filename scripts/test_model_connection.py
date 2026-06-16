import requests
import json

def test_ollama_connection():
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen2.5:0.5b-instruct",
        "messages": [
            {"role": "user", "content": "Explain who you are in one sentence."}
        ],
        "stream": False
    }
    
    print("Connecting to local Ollama model qwen2.5:0.5b-instruct...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            reply = result.get("message", {}).get("content", "")
            print("\n[SUCCESS] Connected successfully!")
            print(f"Response: {reply}\n")
            return True
        else:
            print(f"\n[ERROR] Ollama returned status code: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Ollama. Is it running?")
        print(f"Details: {e}\n")
        return False

if __name__ == "__main__":
    test_ollama_connection()
