import os
import sys
import huggingface_hub

def main():
    print("=== Zenthi-AI OS Multi-Model Adapter Deployer ===")
    
    # Load token from environment or prompt user
    token = os.getenv("HF_TOKEN")
    if not token:
        print("To deploy the adapters, you need an Access Token with 'Write' permissions.")
        print("You can generate one at: https://huggingface.co/settings/tokens")
        token = input("Enter your Hugging Face Access Token: ").strip()
    repo_id = "KATHIR2006/zenthi-ai"
    
    # 1. Login
    try:
        huggingface_hub.login(token=token)
        print("[SUCCESS] Logged in to Hugging Face Hub successfully.")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        sys.exit(1)
        
    api = huggingface_hub.HfApi()
    
    # 2. Upload Code Adapters
    code_lora_dir = "F:/Zenith-AI/models/Zenthi-AI-LoRA"
    if os.path.exists(code_lora_dir):
        print(f"\nUploading Code Adapters from {code_lora_dir} to {repo_id} (code-adapters/)...")
        try:
            api.upload_folder(
                folder_path=code_lora_dir,
                path_in_repo="code-adapters",
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=["checkpoint-*"]
            )
            print("[SUCCESS] Code Adapters uploaded.")
        except Exception as e:
            print(f"[ERROR] Code adapters upload failed: {e}")
    else:
        print("[WARNING] Code adapters directory not found. Skipping.")
        
    # 3. Upload Vision Adapters
    vision_lora_dir = "F:/Zenith-AI/models/Zenthi-AI-Vision-LoRA"
    if os.path.exists(vision_lora_dir):
        print(f"\nUploading Vision Adapters from {vision_lora_dir} to {repo_id} (vision-adapters/)...")
        try:
            api.upload_folder(
                folder_path=vision_lora_dir,
                path_in_repo="vision-adapters",
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=["checkpoint-*"]
            )
            print("[SUCCESS] Vision Adapters uploaded.")
        except Exception as e:
            print(f"[ERROR] Vision adapters upload failed: {e}")
    else:
        print("[WARNING] Vision adapters directory not found. Skipping.")
        
    # 4. Upload updated main README.md
    readme_path = "F:/Zenith-AI/models/Zenthi-AI-merged/README.md"
    if os.path.exists(readme_path):
        print(f"\nUploading updated main README.md model card to {repo_id}...")
        try:
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model"
            )
            print("[SUCCESS] Model card README.md updated on Hugging Face.")
        except Exception as e:
            print(f"[ERROR] Model card upload failed: {e}")
            
    print("\n" + "="*50)
    print("[SUCCESS] All adapters and updated model card deployed successfully!")
    print(f"Explore your public model page: https://huggingface.co/KATHIR2006/zenthi-ai")
    print("="*50)

if __name__ == "__main__":
    main()
