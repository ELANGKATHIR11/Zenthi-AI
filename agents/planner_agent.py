import os
import json
import re
from typing import Dict, Any, List, Optional
from agents.ollama_client import OllamaClient

class PlannerAgent:
    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self.model = os.environ.get("ZENTHI_MODEL", "zenthi-ai:latest")

    def plan(self, query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Generates a plan containing sequential steps to solve a complex query."""
        system_prompt = (
            "You are the Planner Agent for Zenthi-AI OS. Your job is to break down a complex, multi-step query into a sequential list of steps.\n"
            "Each step must use one of these tools/actions:\n"
            "- SEARCH: Web search for current info\n"
            "- RAG: Retrieve context from documents\n"
            "- MEMORY: Read past conversational memory\n"
            "- CODE: Execute code task or calculation\n"
            "- EXECUTE: Run final expert model synthesis\n\n"
            "Format your output ONLY as a JSON object:\n"
            "{\n"
            '  "steps": [\n'
            '    {"step": 1, "tool": "SEARCH|RAG|MEMORY|CODE|EXECUTE", "instruction": "description of what to do"}\n'
            '  ],\n'
            '  "reasoning": "why this plan was chosen"\n'
            "}"
        )
        
        if not self.client.is_healthy():
            # Fast-path heuristic decomposition when Ollama is offline
            steps = []
            step_idx = 1
            q_lower = query.lower()
            if any(w in q_lower for w in ["pdf", "doc", "file", "upload"]):
                steps.append({"step": step_idx, "tool": "RAG", "instruction": f"Retrieve document context for: {query}"})
                step_idx += 1
            if any(w in q_lower for w in ["search", "web", "latest", "news", "current"]):
                steps.append({"step": step_idx, "tool": "SEARCH", "instruction": f"Search web context for: {query}"})
                step_idx += 1
            if any(w in q_lower for w in ["code", "script", "python", "function", "calculate"]):
                steps.append({"step": step_idx, "tool": "CODE", "instruction": f"Code synthesis for: {query}"})
                step_idx += 1
            steps.append({"step": step_idx, "tool": "EXECUTE", "instruction": f"Synthesize final solution for: {query}"})
            return {
                "steps": steps,
                "reasoning": "Native fast-path heuristic task decomposition."
            }

        user_prompt = f"Query: {query}\n"
        if conversation_context:
            user_prompt += f"Context:\n{conversation_context}"
            
        try:
            res = self.client.generate(
                model=self.model,
                prompt=user_prompt,
                system=system_prompt,
                options={"temperature": 0.1}
            )
            
            response_text = res.get("response", "").strip()
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            return {"error": f"Planner failed: {str(e)}"}
            
        # Fallback plan if JSON fails
        return {
            "steps": [
                {"step": 1, "tool": "EXECUTE", "instruction": f"Synthesize response directly for: {query}"}
            ],
            "reasoning": "Fallback default plan due to planner parsing failure."
        }

if __name__ == "__main__":
    planner = PlannerAgent()
    q = "Read the uploaded PDF about machine learning, search the web for the latest PyTorch release, and write a training script."
    print("Plan for:", q)
    print(json.dumps(planner.plan(q), indent=2))
