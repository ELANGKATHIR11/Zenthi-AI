import torch
import os
import sys
import time

from agents.orchestrator import ZenthiAIOrchestrator

def main():
    print("Initializing Orchestrator...")
    try:
        orchestrator = ZenthiAIOrchestrator()
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("Testing connection to Ollama...")
    healthy = orchestrator.ollama.is_healthy()
    print(f"Ollama Healthy: {healthy}")
    
    try:
        models = orchestrator.ollama.list_local_models()
        print("Available models:", models)
    except Exception as e:
        print(f"Failed to list models: {e}")
        
    # Test query
    query = "Write a python function to print hello world"
    print(f"\nProcessing query: '{query}'")
    try:
        res = orchestrator.process_request(query, session_id="test_session")
        print("\nResponse:")
        print(res.get("response"))
        print("\nWorkflow Steps:")
        for step in res.get("workflow_steps", []):
            print(f" - {step}")
    except Exception as e:
        print(f"Failed to process request: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
