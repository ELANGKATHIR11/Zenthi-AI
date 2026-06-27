import os
import sys
from huggingface_hub import HfApi, login

def main():
    print("=== Zenthi-AI OS Model Uploader ===")
    
    # Target folders and files
    model_dir = "F:/Zenith-AI/models/Zenthi-AI-merged"
    repo_id = "KATHIR2006/zenthi-ai"
    
    if not os.path.exists(model_dir):
        print(f"[ERROR] Model directory does not exist: {model_dir}")
        print("Please check if the model has been successfully fine-tuned and merged.")
        sys.exit(1)
        
    print(f"Model path verified: {model_dir}")
    print(f"Target Hugging Face Repository: {repo_id}")
    
    # Prompt user for write token
    print("\nTo upload to Hugging Face, you need an Access Token with 'Write' permissions.")
    print("You can get one from: https://huggingface.co/settings/tokens")
    token = input("Enter your Hugging Face Access Token: ").strip()
    
    if not token:
        print("[ERROR] Token cannot be empty. Aborting upload.")
        sys.exit(1)
        
    # Login
    try:
        login(token=token)
        print("[SUCCESS] Logged in to Hugging Face Hub successfully.")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        sys.exit(1)
        
    # Initialize HfApi
    api = HfApi()
    
    # Create repo if not exists
    print(f"\nCreating/Verifying repository: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"[SUCCESS] Repository {repo_id} is ready.")
    except Exception as e:
        print(f"[ERROR] Failed to verify/create repository: {e}")
        sys.exit(1)
        
    # Upload folder contents
    print(f"\nUploading model files from {model_dir} to Hugging Face Hub...")
    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print("\n" + "="*50)
        print("[SUCCESS] Model files uploaded successfully!")
        print(f"Your model is now public at: https://huggingface.co/KATHIR2006/zenthi-ai")
        print("="*50)
    except Exception as e:
        print(f"[ERROR] Model upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
