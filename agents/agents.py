import os
import sys
import torch
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.vector_store import LocalRAG
from search.searxng_client import SearXNGClient
from memory.history import SessionMemory
from agents.ollama_client import OllamaClient

class RetrievalAgent:
    def __init__(self, rag: Optional[LocalRAG] = None):
        self.rag = rag or LocalRAG()
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query document store to retrieve context."""
        try:
            return self.rag.query(query, top_k=top_k)
        except Exception as e:
            print(f"[RetrievalAgent ERROR] Ingestion/Query failed: {e}")
            return []

class SearchAgent:
    def __init__(self, search_client: Optional[SearXNGClient] = None):
        self.search_client = search_client or SearXNGClient()
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query SearXNG client to retrieve web context."""
        try:
            return self.search_client.search(query, top_k=top_k)
        except Exception as e:
            print(f"[SearchAgent ERROR] Search failed: {e}")
            return []
            
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        return self.search_client.format_context(results)

class MemoryAgent:
    def __init__(self, memory: Optional[SessionMemory] = None):
        self.memory = memory or SessionMemory(max_turns=10)
        
    def get_context(self, session_id: str, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.memory.get_messages_for_prompt(session_id, system_prompt=system_prompt)
        
    def add_message(self, session_id: str, role: str, content: str):
        self.memory.add_message(session_id, role, content)
        
    def clear(self, session_id: str):
        self.memory.clear_session(session_id)

class ExecutionAgent:
    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        
    def execute(self, model: str, messages: List[Dict[str, Any]]) -> str:
        """Call Ollama's chat API on the target expert model."""
        try:
            res = self.client.chat(model=model, messages=messages)
            if "message" in res and "content" in res["message"]:
                return res["message"]["content"].strip()
            elif "error" in res:
                return f"[Execution Error] {res['error']}"
            else:
                return f"[Execution Error] Unknown response format: {res}"
        except Exception as e:
            return f"[Execution Error] Failed to contact model {model}: {e}"

class ValidationAgent:
    def validate(self, response: str) -> Dict[str, Any]:
        """Perform validation checks on model outputs."""
        if not response or len(response.strip()) == 0:
            return {"valid": False, "reason": "Empty response received."}
            
        # Avoid common model leakages/artifacts
        leakage_patterns = [
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"\[SYSTEM\]",
            r"\[/SYSTEM\]"
        ]
        for pattern in leakage_patterns:
            if pattern in response:
                return {"valid": False, "reason": f"System format leak detected: '{pattern}'"}
                
        return {"valid": True}
