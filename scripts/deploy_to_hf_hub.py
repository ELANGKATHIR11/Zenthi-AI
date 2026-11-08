#!/usr/bin/env python3
"""
Zenith-4B Hugging Face Model Hub Deployment Script
Uploads model weights, mmproj, configs, tokenizers, digital signatures, and benchmarks
to the public Hugging Face model repository KATHIR2006/Zenith-4B.
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, get_token

HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()
REPO_ID = "KATHIR2006/Zenith-4B"
REPO_TYPE = "model"
MODEL_DIR = Path("models/Zenith-4B").resolve()

def deploy():
    print(f"[*] Initializing Hugging Face deployment...")
    print(f"[*] Target Repository: {REPO_ID} (type: {REPO_TYPE})")
    print(f"[*] Source Directory:  {MODEL_DIR}")
    
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")
        
    api = HfApi(token=HF_TOKEN)
    
    # Ensure public repo exists
    print(f"[*] Ensuring public model repository exists on Hugging Face...")
    repo_url = api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        exist_ok=True
    )
    print(f"[+] Repository URL: {repo_url}")
    
    commit_message = "Release Zenith-4B: Foundation weights, mmproj, benchmarks, and Qwen 3.5 attribution"
    commit_description = (
        "Zenith-4B is a pure fine-tuned multimodal vision-language foundation model.\n"
        "Created by Kathir (Zenith AI).\n"
        "Special credit and acknowledgment to Qwen 3.5-4B (Alibaba Cloud) for the foundational architecture.\n"
        "Features 100% coding unit test pass rate and +48.61% composite benchmark improvement."
    )
    
    print(f"[*] Uploading folder contents to {REPO_ID} (Git LFS enabled for safetensors & gguf)...")
    start_time = time.time()
    
    commit_info = api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=commit_message,
        commit_description=commit_description,
        token=HF_TOKEN
    )
    
    elapsed = time.time() - start_time
    print(f"[+] Deployment complete in {elapsed:.2f} seconds!")
    print(f"[+] Commit URL: {commit_info}")
    print(f"[+] Public Model Hub: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    deploy()
