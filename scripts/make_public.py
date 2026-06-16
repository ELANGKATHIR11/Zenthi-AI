import os
from huggingface_hub import HfApi

def main():
    token = os.getenv("HF_TOKEN")
    repo_id = "KATHIR2006/zenthi-ai"
    
    print(f"Ensuring Hugging Face repository '{repo_id}' is set to PUBLIC...")
    api = HfApi(token=token)
    try:
        api.update_repo_settings(
            repo_id=repo_id,
            private=False,
            repo_type="model"
        )
        print("[SUCCESS] Repository settings updated! The repository is now PUBLIC.")
    except Exception as e:
        print(f"[ERROR] Failed to update settings: {e}")

if __name__ == "__main__":
    main()
