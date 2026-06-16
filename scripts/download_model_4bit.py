import os
os.environ["BNB_CUDA_VERSION"] = "129"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct-4bit"))

def download_and_quantize_4bit():
    print(f"Starting native 4-bit download/quantization of: {MODEL_NAME}...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Download and save tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(SAVE_DIR)
    
    # Configure 4-bit bitsandbytes quantization
    print("Configuring 4-bit bitsandbytes parameters...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model directly in 4-bit (uses bitsandbytes internally to quantize weights on the fly during load)
    print("Downloading and loading weights directly in 4-bit (saving VRAM and disk space)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Save the quantized model locally
    print(f"Saving quantized 4-bit model to: {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    print(f"[SUCCESS] Native 4-bit model saved locally at: {SAVE_DIR}")

if __name__ == "__main__":
    download_and_quantize_4bit()
