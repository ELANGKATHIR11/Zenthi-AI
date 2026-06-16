import os
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b-instruct"
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "synthetic_data.jsonl"))

# System Prompt detailing Zenthi-AI personality
SYSTEM_PROMPT = (
    "I am Zenthi-AI, a lightweight educational AI assistant developed as a custom Small Language Model project. "
    "I am educational, helpful, accurate, concise, student-friendly, and programming-aware."
)

def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error querying Ollama: {e}")
    return ""

def generate_identity_data():
    print("Generating identity examples...")
    prompts = [
        "Write 10 variations of questions a user might ask about your name, developer, and purpose, and provide the exact answers as Zenthi-AI. Format as JSON list of objects with 'instruction' and 'response'. Keep answers concise, and mention you are developed as a college final year project.",
        "Generate 10 conversational exchanges where you explain your features (RAG, web search, memory) and your educational purpose. Format as JSON list of objects with 'instruction' and 'response'."
    ]
    
    data = []
    for p in prompts:
        resp = query_ollama(p)
        # Parse JSON from block if exists
        try:
            if "```json" in resp:
                json_str = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                json_str = resp.split("```")[1].split("```")[0].strip()
            else:
                json_str = resp
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                data.extend(parsed)
        except Exception as e:
            print(f"Failed to parse identity response: {e}. Raw: {resp[:200]}...")
    return data

def generate_educational_data():
    print("Generating educational examples...")
    prompts = [
        "Generate 10 student questions about basic programming concepts (recursion, OOP, variables, Big O) and detailed student-friendly explanations. Format as JSON list of objects with 'instruction' and 'response'.",
        "Generate 10 math or logic reasoning questions with step-by-step explanations. Format as JSON list of objects with 'instruction' and 'response'."
    ]
    
    data = []
    for p in prompts:
        resp = query_ollama(p)
        try:
            if "```json" in resp:
                json_str = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                json_str = resp.split("```")[1].split("```")[0].strip()
            else:
                json_str = resp
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                data.extend(parsed)
        except Exception as e:
            print(f"Failed to parse educational response: {e}. Raw: {resp[:200]}...")
    return data

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    identity = generate_identity_data()
    edu = generate_educational_data()
    
    all_synthetic = identity + edu
    
    if not all_synthetic:
        # Fallback dataset if generator fails or model returns non-json
        print("Using fallback synthetic data.")
        all_synthetic = [
            {
                "instruction": "Who are you?",
                "response": "I am Zenthi-AI, a lightweight educational AI assistant developed as a custom Small Language Model project."
            },
            {
                "instruction": "What is your project name?",
                "response": "My project name is Zenthi-AI, designed as a custom college final-year project."
            },
            {
                "instruction": "What can you do?",
                "response": "I support conversational chat, local document RAG, live web search, and session-based conversation memory."
            },
            {
                "instruction": "Explain recursion in programming.",
                "response": "Recursion is a programming technique where a function calls itself to solve a smaller instance of the same problem."
            }
        ]
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_synthetic:
            # Format in Qwen message format
            msg_format = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item.get("instruction")},
                    {"role": "assistant", "content": item.get("response")}
                ]
            }
            f.write(json.dumps(msg_format) + "\n")
            
    print(f"[SUCCESS] Synthetic data generation finished. Saved {len(all_synthetic)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
