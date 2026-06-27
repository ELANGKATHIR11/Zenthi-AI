import os
import re
import json
from typing import Dict, Any, Optional
from agents.ollama_client import OllamaClient

class RouterAgent:
    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self.model = "qwen2.5:1.5b-instruct"
        
        # Simple heuristics as fallbacks
        self.code_patterns = [
            r"\b(code|python|react|javascript|html|css|api|database|sql|function|class|bug|debug|refactor|compile)\b",
            r"```",
            r"\b(node|express|mongodb|fastapi|django|flask)\b"
        ]
        self.search_patterns = [
            r"\b(weather|news|today|latest|current|recent|price|stock|score|yesterday|tomorrow|live|currently)\b"
        ]
        self.rag_patterns = [
            r"\b(document|pdf|file|upload|indexed|notebook|paper|text|context|reference)\b"
        ]

    def _heuristic_classify(self, query: str, has_images: bool = False) -> str:
        if has_images:
            return "VISION"
        
        query_lower = query.lower()
        has_code = any(re.search(pat, query_lower) for pat in self.code_patterns)
        has_search = any(re.search(pat, query_lower) for pat in self.search_patterns)
        has_rag = any(re.search(pat, query_lower) for pat in self.rag_patterns)
        
        if has_code:
            return "CODE"
        if has_rag:
            return "RAG"
        if has_search:
            return "SEARCH"
        return "KNOWLEDGE"

    def route(self, query: str, has_images: bool = False) -> Dict[str, Any]:
        """Classifies query using Ollama LLM, falling back to heuristics if needed."""
        if has_images:
            return {
                "intent": "VISION",
                "expert_model": "riven/smolvlm:latest",
                "reasoning": "Input contains image content."
            }

        # Build prompt for Ollama Router
        system_prompt = (
            "You are the Router Agent for Zenthi-AI OS. Your job is to classify the user's intent into one of the following categories:\n"
            "- VISION: Prompt is about analyzing images or visual components.\n"
            "- CODE: Programming questions, code requests, debugging, software architecture, syntax help, math/logic calculations.\n"
            "- RAG: Questions asking to retrieve from files, documents, papers, uploads, or specific context.\n"
            "- SEARCH: Questions requiring real-time web search or current information (weather, news, scores, dates).\n"
            "- COMPLEX: Multi-step requests that require planning and coordination of multiple actions.\n"
            "- KNOWLEDGE: General conversation, summaries, reports, writing assignments, history, general questions.\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "intent": "KNOWLEDGE|CODE|RAG|SEARCH|COMPLEX",\n'
            '  "reasoning": "brief explanation"\n'
            "}"
        )
        
        try:
            res = self.client.generate(
                model=self.model,
                prompt=f"Classify this query: {query}",
                system=system_prompt,
                options={"temperature": 0.0}  # Deterministic routing
            )
            
            response_text = res.get("response", "").strip()
            # Try parsing JSON from the response text
            # Find json boundaries if there is wrapping text
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                intent = data.get("intent", "KNOWLEDGE").upper()
                reasoning = data.get("reasoning", "")
            else:
                intent = self._heuristic_classify(query)
                reasoning = "LLM response was not valid JSON; fallback to heuristics."
        except Exception as e:
            intent = self._heuristic_classify(query)
            reasoning = f"Exception in LLM routing: {str(e)}; fallback to heuristics."
            
        # Map intent to expert model
        model_mapping = {
            "VISION": "riven/smolvlm:latest",
            "CODE": "qwen2.5-coder:3b",
            "KNOWLEDGE": "qwen2.5:1.5b-instruct",
            "RAG": "qwen2.5:1.5b-instruct",
            "SEARCH": "qwen2.5:1.5b-instruct",
            "COMPLEX": "qwen2.5:1.5b-instruct"
        }
        
        # Safe default fallback
        if intent not in model_mapping:
            intent = "KNOWLEDGE"
            
        return {
            "intent": intent,
            "expert_model": model_mapping[intent],
            "reasoning": reasoning
        }

if __name__ == "__main__":
    router = RouterAgent()
    test_queries = [
        "Write a python quicksort function.",
        "What is the weather in New York today?",
        "Explain the main theme of the document I uploaded.",
        "Who was Abraham Lincoln?",
        "Can you fix this bug in my React code?"
    ]
    for q in test_queries:
        print(f"Query: {q}")
        print(f"Result: {router.route(q)}\n")
