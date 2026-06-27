import os
import sys
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ollama_client import OllamaClient
from agents.router_agent import RouterAgent
from agents.planner_agent import PlannerAgent
from agents.agents import RetrievalAgent, SearchAgent, MemoryAgent, ExecutionAgent, ValidationAgent

class ZenthiAIOrchestrator:
    def __init__(self):
        self.ollama = OllamaClient()
        self.router = RouterAgent(self.ollama)
        self.planner = PlannerAgent(self.ollama)
        self.retrieval = RetrievalAgent()
        self.search = SearchAgent()
        self.memory = MemoryAgent()
        self.execution = ExecutionAgent(self.ollama)
        self.validation = ValidationAgent()
        
    def process_request(self, query: str, session_id: str, mode: str = "AUTO", images: Optional[List[str]] = None) -> Dict[str, Any]:
        workflow_steps = []
        citations = []
        context = ""
        
        # 1. Validate Input
        workflow_steps.append("1. Validating input query...")
        if not query.strip() and not images:
            return {
                "response": "Error: Input query is empty.",
                "session_id": session_id,
                "mode": mode,
                "citations": [],
                "workflow_steps": ["Input validation failed."]
            }
            
        has_images = bool(images)
        
        # 2. Detect Intent & Select Expert
        workflow_steps.append("2. Analyzing query intent and selecting expert model...")
        routing_info = self.router.route(query, has_images=has_images)
        detected_intent = routing_info["intent"]
        expert_model = routing_info["expert_model"]
        
        workflow_steps.append(f"-> Detected intent: {detected_intent}")
        workflow_steps.append(f"-> Target expert model: {expert_model}")
        
        # Override mode if manually specified
        run_mode = detected_intent
        if mode != "AUTO":
            run_mode = mode
            
        # 3. Determine Required Tools & Steps (Planner for COMPLEX tasks)
        if run_mode == "COMPLEX":
            workflow_steps.append("3. Complex task detected. Planner Agent generating execution plan...")
            plan_info = self.planner.plan(query)
            workflow_steps.append(f"-> Plan generated: {plan_info.get('reasoning', 'No reasoning provided')}")
            # Execute planner steps
            for step in plan_info.get("steps", []):
                tool = step.get("tool")
                instruction = step.get("instruction")
                workflow_steps.append(f"   [Step {step.get('step')}] Running {tool}: {instruction}")
                
                if tool == "RAG":
                    rag_docs = self.retrieval.retrieve(query, top_k=2)
                    if rag_docs:
                        context += "\n=== RAG Context ===\n"
                        for doc in rag_docs:
                            context += f"- [{doc['source']}]: {doc['text']}\n"
                            if f"Doc: {doc['source']}" not in citations:
                                citations.append(f"Doc: {doc['source']}")
                elif tool == "SEARCH":
                    search_res = self.search.search(query, top_k=2)
                    if search_res:
                        context += "\n=== Live Web Search Context ===\n"
                        context += self.search.format_results(search_res)
                        for r in search_res:
                            citations.append(f"Web: {r['title']} ({r['url']})")
        else:
            workflow_steps.append("3. Processing single-turn workflow tools...")
            
            # 4. Memory Integration
            workflow_steps.append("4. Checking conversational memory for relevant history...")
            # Handled below by loading full session history
            
            # 5. Retrieval Integration
            if run_mode in ["RAG", "HYBRID"] or detected_intent == "RAG":
                workflow_steps.append("5. Retrieval Agent querying ChromaDB document store...")
                rag_docs = self.retrieval.retrieve(query, top_k=3)
                if rag_docs:
                    context += "\n=== Local Document Context ===\n"
                    for doc in rag_docs:
                        context += f"- [{doc['source']}]: {doc['text']}\n"
                        citations.append(f"Local Doc: {doc['source']}")
                else:
                    workflow_steps.append("   (No matching local documents found)")
            else:
                workflow_steps.append("5. RAG not required for this request.")
                
            # 6. Web Search Integration
            if run_mode in ["SEARCH", "HYBRID"] or detected_intent == "SEARCH":
                workflow_steps.append("6. Search Agent querying SearXNG for current information...")
                search_res = self.search.search(query, top_k=3)
                if search_res:
                    context += "\n=== Live Web Search Context ===\n"
                    context += self.search.format_results(search_res)
                    for r in search_res:
                        citations.append(f"Web: {r['title']} ({r['url']})")
                else:
                    workflow_steps.append("   (No web search results returned)")
            else:
                workflow_steps.append("6. Web search not required for this request.")
                
        # 7. Format Complete Prompt & Select model
        workflow_steps.append(f"7. Compiling context and history for Expert Model: {expert_model}...")
        
        # Build prompt content
        user_message_content = query
        if context:
            user_message_content = (
                f"Context:\n{context}\n\n"
                f"User Query: {query}\n\n"
                f"Instructions: Answer using the provided context as supporting reference. Maintain the Zenthi-AI OS identity."
            )
            
        # Get memory history
        system_prompt = (
            "I am Zenthi-AI OS, a production-grade Agentic Multi-Model AI Operating System. "
            "I deliver accurate, secure, maintainable, and production-ready solutions by coordinating specialized AI capabilities."
        )
        
        # Fetch history and add current user message
        history = self.memory.get_context(session_id, system_prompt=system_prompt)
        
        # Ollama /api/chat messages format
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add the new user prompt
        user_msg = {"role": "user", "content": user_message_content}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)
        
        # Save user message to memory
        self.memory.add_message(session_id, "user", query)
        
        # 8. Execute Expert Model
        workflow_steps.append(f"8. Dispatched execution to Expert: {expert_model}...")
        response_content = self.execution.execute(expert_model, messages)
        
        # 9. Validate Response Output
        workflow_steps.append("9. Validation Agent running quality check on output...")
        validation_res = self.validation.validate(response_content)
        if not validation_res["valid"]:
            workflow_steps.append(f"-> Validation warning: {validation_res['reason']}")
            # Run reflection / cleanup if formatting issues found
            response_content = response_content.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            
        # 10. Save assistant reply to memory & Return
        workflow_steps.append("10. Response finalized. Returning to workspace.")
        self.memory.add_message(session_id, "assistant", response_content)
        
        return {
            "response": response_content,
            "session_id": session_id,
            "mode": run_mode,
            "citations": list(set(citations)),
            "workflow_steps": workflow_steps
        }
