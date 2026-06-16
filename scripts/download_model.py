import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct"))

def download_model():
    print(f"Starting native download of model and tokenizer: {MODEL_NAME}...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(SAVE_DIR)
    
    # Download model weights
    print("Downloading model weights (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu", # Download weights using CPU to avoid CUDA resource conflicts during download
        trust_remote_code=True
    )
    model.save_pretrained(SAVE_DIR)
    print(f"[SUCCESS] Native model saved locally at: {SAVE_DIR}")

if __name__ == "__main__":
    download_model()
