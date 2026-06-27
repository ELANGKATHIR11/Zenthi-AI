import os
import requests
from typing import List, Dict, Any, Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host.rstrip("/")
        
    def is_healthy(self) -> bool:
        """Check if Ollama server is running and healthy."""
        try:
            response = requests.get(f"{self.host}/", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def list_local_models(self) -> List[str]:
        """Fetch list of models available locally in Ollama."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"[OllamaClient ERROR] Failed to list models: {e}")
        return []

    def chat(self, model: str, messages: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a chat query to Ollama's chat endpoint."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if options:
            payload["options"] = options
            
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Ollama returned status {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Failed to connect to Ollama: {str(e)}"}

    def generate(self, model: str, prompt: str, system: Optional[str] = None, images: Optional[List[str]] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a completion query to Ollama's generate endpoint."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images
        if options:
            payload["options"] = options
            
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Ollama returned status {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Failed to connect to Ollama: {str(e)}"}

if __name__ == "__main__":
    client = OllamaClient()
    print("Ollama running:", client.is_healthy())
    print("Available models:", client.list_local_models())
