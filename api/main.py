import os
import sys
import uuid
import shutil
import anyio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import ZenthiAIOrchestrator
from api.security import (
    validate_safe_upload_path,
    sanitize_user_input,
    SecurityValidationError,
    MAX_FILE_SIZE_BYTES
)
from zenith_engine.runtime_manager import HardwareProfile

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
        "status": "healthy",
        "engine": "Zenthi-AI Multi-Agent OS",
        "native_engine_ready": True,
        "ollama_connected": ollama_healthy
    }

@app.get("/metrics")
def get_metrics():
    hw = HardwareProfile()
    return {
        "status": "active",
        "hardware": hw.get_memory_status()
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        safe_filepath = validate_safe_upload_path(file.filename, UPLOAD_DIR)
        
        # Read file with size limit guard
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
            )
            
        with open(safe_filepath, "wb") as buffer:
            buffer.write(content)
        
        # Ingest to Qdrant vector database via the Orchestrator's Retrieval Agent
        success = await anyio.to_thread.run_sync(orchestrator.retrieval.rag.ingest_document, safe_filepath)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to parse and index document content.")
            
        return {
            "status": "success",
            "filename": os.path.basename(safe_filepath),
            "detail": "File indexed in Qdrant successfully."
        }
    except SecurityValidationError as sec_err:
        raise HTTPException(status_code=400, detail=str(sec_err))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_web_endpoint(q: str):
    clean_query = sanitize_user_input(q)
    results = await anyio.to_thread.run_sync(orchestrator.search.search, clean_query)
    return {"query": clean_query, "results": results}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    query = sanitize_user_input(req.query)
    mode = req.mode or "AUTO"
    images = req.images
    
    try:
        result = await anyio.to_thread.run_sync(
            lambda: orchestrator.process_request(
                query=query,
                session_id=session_id,
                mode=mode,
                images=images
            )
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
