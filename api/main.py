import os
os.environ["BNB_CUDA_VERSION"] = "129" # Apply bitsandbytes Windows matching override

import sys
import uuid
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.vector_store import LocalRAG
from search.searxng_client import SearXNGClient
from memory.history import SessionMemory
from api.router import QueryRouter

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

# Initialize components
rag = LocalRAG()
search_client = SearXNGClient()
memory = SessionMemory(max_turns=10)
router = QueryRouter()

# Load native local 4-bit model
LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct-4bit"))
print(f"Loading local 4-bit tokenizer and model from: {LOCAL_MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully.")

# System prompt for Zenthi-AI
SYSTEM_PROMPT = (
    "I am Zenthi-AI, a lightweight educational AI assistant developed as a custom Small Language Model project. "
    "I am educational, helpful, accurate, concise, student-friendly, and programming-aware."
)

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: Optional[str] = "AUTO" # AUTO, DIRECT, RAG, SEARCH, HYBRID

class ChatResponse(BaseModel):
    response: str
    session_id: str
    mode: str
    citations: List[str]

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "Zenthi-AI (Ollama)"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest to ChromaDB
        success = rag.ingest_document(filepath)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to parse and index document content.")
            
        return {"status": "success", "filename": file.filename, "detail": "File indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_web_endpoint(q: str):
    results = search_client.search(q)
    return {"query": q, "results": results}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    query = req.query
    
    # Determine routing mode
    mode = req.mode
    if mode == "AUTO":
        mode = router.route(query)
        
    citations = []
    context = ""
    
    # Gather Context
    if mode == "RAG" or mode == "HYBRID":
        rag_results = rag.query(query, top_k=3)
        if rag_results:
            context += "\n=== Local Document Context ===\n"
            for r in rag_results:
                context += f"- [{r['source']}]: {r['text']}\n"
                if r['source'] not in citations:
                    citations.append(f"Local Doc: {r['source']}")
                    
    if mode == "SEARCH" or mode == "HYBRID":
        web_results = search_client.search(query, top_k=3)
        if web_results:
            context += "\n=== Live Web Search Context ===\n"
            context += search_client.format_context(web_results)
            for w in web_results:
                if w['url'] not in citations:
                    citations.append(f"Web: {w['title']} ({w['url']})")

    # Retrieve history
    history = memory.get_messages_for_prompt(session_id, system_prompt=SYSTEM_PROMPT)
    
    # Construct prompt with context if available
    user_message = query
    if context:
        user_message = (
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            f"Instructions: Answer the query accurately using the context provided if relevant, otherwise reply using your standard knowledge."
        )
        
    # Append user turn to memory
    memory.add_message(session_id, "user", user_message)
    
    # Re-retrieve complete history including new user message
    messages = memory.get_session(session_id)
    
    # Request response natively using the loaded Hugging Face model
    try:
        # Format conversation with chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Prepare inputs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Generate tokens
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Extract answer content
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        ai_reply = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        ai_reply = f"[ERROR] Native inference failed: {e}"

    # Save assistant reply to memory
    memory.add_message(session_id, "assistant", ai_reply)
    
    return ChatResponse(
        response=ai_reply,
        session_id=session_id,
        mode=mode,
        citations=citations
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
