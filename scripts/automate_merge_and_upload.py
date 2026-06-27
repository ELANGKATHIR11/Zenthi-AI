import os
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.export_gguf import merge_and_export
from scripts.upload_to_hf import main as upload_main

LORA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-LoRA"))
CHECK_FILE = os.path.join(LORA_DIR, "adapter_config.json")

def wait_and_deploy():
    print("Automated Deployer: Monitoring training progress...")
    
    # We poll every 20 seconds. We check if the training output folder and adapter_config.json exists,
    # meaning the Trainer successfully finished and saved the adapter weights.
    max_checks = 240 # Check for 80 minutes max
    checks = 0
    while not os.path.exists(CHECK_FILE):
        time.sleep(20)
        checks += 1
        if checks % 3 == 0:
            print(f"Automated Deployer: Still waiting for training checkpoint... (elapsed: {checks * 20}s)")
        if checks >= max_checks:
            print("[ERROR] Timeout waiting for training to complete.")
            return
            
    print("\n[SUCCESS] Training finished! Starting merge pipeline...")
    # Add a small buffer to ensure file handles are released
    time.sleep(5)
    
    try:
        # Run merge weights
        merge_and_export()
        print("Merge complete!")
    except Exception as e:
        print(f"[ERROR] Weight merge failed: {e}")
        return

    print("Starting Hugging Face upload pipeline...")
    try:
        # Run upload to HF Hub
        upload_main()
        print("[SUCCESS] Automated deployment complete!")
    except Exception as e:
        print(f"[ERROR] Upload to HF failed: {e}")

if __name__ == "__main__":
    wait_and_deploy()
