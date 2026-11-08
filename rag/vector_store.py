import os
import uuid
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

RAG_DIR = os.path.abspath(os.path.dirname(__file__))
QDRANT_PATH = os.path.join(RAG_DIR, "qdrant_db")
COLLECTION_NAME = "zenthi_ai_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384

class LocalRAG:
    _instance = None
    _client = None
    _model = None
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalRAG, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = QDRANT_PATH):
        if LocalRAG._client is not None:
            self.client = LocalRAG._client
            self.tokenizer = LocalRAG._tokenizer
            self.model = LocalRAG._model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            return
            
        os.makedirs(db_path, exist_ok=True)
        self.client = QdrantClient(path=db_path)
        LocalRAG._client = self.client
        
        # Initialize collection if not present
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' created at {db_path}.")
        else:
            print(f"[Qdrant] Connected to existing collection '{COLLECTION_NAME}'.")
            
        # Initialize GPU-accelerated local embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Qdrant] Loading embedding model '{EMBEDDING_MODEL_NAME}' on {self.device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(self.device)
        self.model.eval()
        
        LocalRAG._tokenizer = self.tokenizer
        LocalRAG._model = self.model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized 384-d dense embeddings using PyTorch CUDA."""
        if not texts:
            return []
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().tolist()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            return splitter.split_text(text)
        except Exception:
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap
                if start >= len(text):
                    break
            return chunks

    def ingest_document(self, filepath: str) -> bool:
        print(f"[Qdrant] Ingesting file: {filepath}")
        if not os.path.exists(filepath):
            print(f"[Qdrant ERROR] File does not exist: {filepath}")
            return False
            
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        
        content = ""
        if ext in [".txt", ".md"]:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(filepath)
                text_list = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text_list.append(t)
                content = "\n".join(text_list)
            except Exception as e:
                print(f"[Qdrant WARNING] pypdf extraction failed: {e}")
                return False
        else:
            print(f"[Qdrant] Unsupported file format: {ext}. Supported: .txt, .md, .pdf")
            return False
            
        if not content.strip():
            print("[Qdrant WARNING] No readable text found in document.")
            return False
            
        chunks = self.chunk_text(content)
        if not chunks:
            return False
            
        vectors = self.embed_texts(chunks)
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": chunk,
                    "source": filename
                }
            )
            for chunk, vec in zip(chunks, vectors)
        ]
        
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"[Qdrant SUCCESS] Successfully indexed {len(points)} chunks from '{filename}'.")
        return True

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vectors = self.embed_texts([query_text])
        if not query_vectors:
            return []
            
        query_vec = query_vectors[0]
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k
        )
        
        retrieved_docs = []
        for point in results.points:
            payload = point.payload or {}
            retrieved_docs.append({
                "text": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "score": point.score
            })
            
        return retrieved_docs

if __name__ == "__main__":
    rag = LocalRAG()
    test_file = os.path.join(RAG_DIR, "test_document.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Zenthi-AI is an Agentic Multi-Model AI Operating System.\n"
                "It uses Qdrant Vector Database for local document storage and semantic search.\n"
                "Qdrant provides high-speed vector retrieval with cosine distance.")
                
    rag.ingest_document(test_file)
    res = rag.query("What vector database does Zenthi-AI use?")
    print("\nQuery Results:")
    for r in res:
        print(f"- [{r['source']}] (Score: {r['score']:.4f}): {r['text']}")
        
    try:
        os.remove(test_file)
    except:
        pass
