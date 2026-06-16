import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

RAG_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(RAG_DIR, "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class LocalRAG:
    def __init__(self):
        os.makedirs(DB_PATH, exist_ok=True)
        # Set up persistent Chroma client
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        # Load local sentence transformers embedding function
        print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="zenthi_ai_knowledge",
            embedding_function=self.embedding_function
        )
        print("ChromaDB persistent collection initialized.")

    def chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def ingest_document(self, filepath):
        print(f"Ingesting file: {filepath}")
        if not os.path.exists(filepath):
            print("File does not exist.")
            return False
            
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        
        content = ""
        if ext == ".txt":
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        elif ext == ".md":
            with open(filepath, 'r', encoding='utf-8') as f:
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
            except ImportError:
                # Fallback if pypdf is not installed
                print("[WARNING] pypdf not installed. Please run `pip install pypdf` for PDF parsing. Extracting raw lines if TXT encoded.")
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except:
                    pass
        else:
            print("Unsupported file format. Supported: .txt, .md, .pdf")
            return False
            
        if not content.strip():
            print("No text content found to index.")
            return False
            
        # Chunk text
        chunks = self.chunk_text(content)
        print(f"Generated {len(chunks)} chunks.")
        
        # Ingest to ChromaDB
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": filename} for _ in chunks]
        
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[SUCCESS] Successfully indexed {len(chunks)} chunks from {filename}.")
        return True

    def query(self, query_text, top_k=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}]*len(docs)
            for doc, meta in zip(docs, metas):
                retrieved_docs.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown")
                })
        return retrieved_docs

if __name__ == "__main__":
    # Self-test
    rag = LocalRAG()
    test_file = os.path.join(RAG_DIR, "test_document.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Zenthi-AI is a specialized educational Small Language Model fine-tuned for college final-year projects.\n"
                "It has an integrated ChromaDB RAG layer to access local text, markdown, and PDF files.\n"
                "The system supports SearXNG search engine integrations for real-time live queries.")
                
    rag.ingest_document(test_file)
    res = rag.query("What is Zenthi-AI RAG layer?")
    print("Query Results:")
    for r in res:
        print(f"- [{r['source']}]: {r['text']}")
    
    # Cleanup test file
    try:
        os.remove(test_file)
    except:
        pass
