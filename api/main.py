import torch
import os
os.environ["BNB_CUDA_VERSION"] = "129" # Apply bitsandbytes Windows matching override

import sys
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import ZenthiAIOrchestrator

app = FastAPI(title="Zenthi-AI Backend API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Orchestrator
print("Initializing Zenthi-AI OS Multi-Agent Orchestrator...")
orchestrator = ZenthiAIOrchestrator()
print("Orchestrator initialized.")

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: Optional[str] = "AUTO" # AUTO, DIRECT, RAG, SEARCH, HYBRID
    images: Optional[List[str]] = None # Base64 encoded images list

class ChatResponse(BaseModel):
    response: str
    session_id: str
    mode: str
    citations: List[str]
    workflow_steps: List[str]

@app.get("/health")
def health_check():
    ollama_healthy = orchestrator.ollama.is_healthy()
    return {
        "status": "healthy" if ollama_healthy else "degraded",
        "ollama_connected": ollama_healthy,
        "engine": "Zenthi-AI Multi-Agent OS"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest to ChromaDB via the Orchestrator's Retrieval Agent
        success = orchestrator.retrieval.rag.ingest_document(filepath)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to parse and index document content.")
            
        return {"status": "success", "filename": file.filename, "detail": "File indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_web_endpoint(q: str):
    results = orchestrator.search.search(q)
    return {"query": q, "results": results}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    query = req.query
    mode = req.mode or "AUTO"
    images = req.images
    
    try:
        result = orchestrator.process_request(
            query=query,
            session_id=session_id,
            mode=mode,
            images=images
        )
        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            mode=result["mode"],
            citations=result["citations"],
            workflow_steps=result["workflow_steps"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Orchestration Error: {str(e)}")

# Mount static frontend files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
