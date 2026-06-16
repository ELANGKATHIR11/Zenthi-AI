import os
from huggingface_hub import HfApi

# Initialize the API client
api = HfApi()

# Define repository parameters
repo_id = "KATHIR2006/zenthi-ai"
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-merged"))

def main():
    if not os.path.exists(folder_path):
        print(f"[ERROR] Merged model folder not found at {folder_path}. Make sure to run training/export_gguf.py first.")
        return
        
    print(f"Creating model repository '{repo_id}' on Hugging Face Hub...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("Repository verified/created.")
    except Exception as e:
        print(f"Repository creation warning (may already exist): {e}")
        
    print(f"Uploading files from local folder '{folder_path}' to HF Hub...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"\n[SUCCESS] Model successfully uploaded! Accessible at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        print("Verify that you have logged in using 'huggingface-cli login' and have Write tokens.")

if __name__ == "__main__":
    main()
