import os
from huggingface_hub import HfApi

def main():
    # Attempt to read from environment variable first, then fallback to default cache token
    token = os.getenv("HF_TOKEN")
    repo_id = "KATHIR2006/zenthi-ai"
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-merged", "README.md"))
    
    if not os.path.exists(file_path):
        print(f"[ERROR] README.md not found at {file_path}")
        return
        
    print(f"Uploading README.md to '{repo_id}' on Hugging Face Hub...")
    try:
        # HfApi automatically uses cached credentials if token is None
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("[SUCCESS] README.md successfully uploaded with Apache 2.0 license card metadata!")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")

if __name__ == "__main__":
    main()
