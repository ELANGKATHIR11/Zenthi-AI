import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-LoRA"))
MERGED_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-merged"))

def merge_and_export():
    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu", # Merge on CPU to avoid CUDA memory issues
        trust_remote_code=True
    )
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if not os.path.exists(LORA_ADAPTER_DIR):
        print(f"[WARNING] LoRA adapter path {LORA_ADAPTER_DIR} not found. Skipping merge and saving base model directly for development setup.")
        model_to_save = base_model
    else:
        print(f"Loading LoRA adapters from: {LORA_ADAPTER_DIR}")
        peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
        print("Merging LoRA adapters into base model...")
        model_to_save = peft_model.merge_and_unload()
        
    print(f"Saving merged model to: {MERGED_OUTPUT_DIR}")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    model_to_save.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    print(f"[SUCCESS] Model merged and saved successfully at {MERGED_OUTPUT_DIR}!")
    
    print("\n--- GGUF Quantization Instructions ---")
    print("To convert this merged model into GGUF format:")
    print("1. Clone llama.cpp repository:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("2. Install requirements:")
    print("   pip install -r llama.cpp/requirements.txt")
    print("3. Convert to GGUF format:")
    print("   python llama.cpp/convert_hf_to_gguf.py F:\\Zenith-AI\\models\\Zenthi-AI-merged --outfile F:\\Zenith-AI\\models\\Zenthi-AI.gguf")
    print("4. Quantize to Q4_K_M (or similar quantization levels):")
    print("   llama-quantize F:\\Zenith-AI\\models\\Zenthi-AI.gguf F:\\Zenith-AI\\models\\Zenthi-AI-Q4_K_M.gguf Q4_K_M")

if __name__ == "__main__":
    merge_and_export()
