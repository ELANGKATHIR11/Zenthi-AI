import os
from huggingface_hub import HfApi

def main():
    token = os.getenv("HF_TOKEN")
    repo_id = "KATHIR2006/zenthi-ai"
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LICENSE"))
    
    if not os.path.exists(file_path):
        print(f"[ERROR] LICENSE not found at {file_path}")
        return
        
    print(f"Uploading LICENSE to '{repo_id}' on Hugging Face Hub...")
    try:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="LICENSE",
            repo_id=repo_id,
            repo_type="model"
        )
        print("[SUCCESS] LICENSE file successfully uploaded to Hugging Face model repository!")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")

if __name__ == "__main__":
    main()
