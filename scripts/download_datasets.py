import os
import sys
import json
from datasets import load_dataset

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

def ensure_dirs():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    print(f"Dataset directory ensured at: {DATASETS_DIR}")

def download_alpaca():
    print("\n--- Downloading Stanford Alpaca ---")
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        filepath = os.path.join(DATASETS_DIR, "alpaca.json")
        dataset.to_json(filepath)
        print(f"[SUCCESS] Saved Alpaca to {filepath} ({len(dataset)} items)")
    except Exception as e:
        print(f"[ERROR] Failed to download Alpaca: {e}")

def download_dolly():
    print("\n--- Downloading Databricks Dolly 15K ---")
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        filepath = os.path.join(DATASETS_DIR, "dolly_15k.jsonl")
        dataset.to_json(filepath)
        print(f"[SUCCESS] Saved Dolly 15K to {filepath} ({len(dataset)} items)")
    except Exception as e:
        print(f"[ERROR] Failed to download Dolly 15K: {e}")

def download_openhermes():
    print("\n--- Downloading OpenHermes 2.5 (Subset) ---")
    try:
        print("Streaming/loading OpenHermes 2.5...")
        dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
        # Since the dataset is extremely large (1M+ rows), we take a subset of the first 15,000 items
        subset = dataset.select(range(min(15000, len(dataset))))
        filepath = os.path.join(DATASETS_DIR, "openhermes_subset.json")
        subset.to_json(filepath)
        print(f"[SUCCESS] Saved OpenHermes subset to {filepath} ({len(subset)} items)")
    except Exception as e:
        print(f"[ERROR] Failed to download OpenHermes: {e}")

def download_ultrachat():
    print("\n--- Downloading UltraChat (Subset) ---")
    try:
        print("Loading HuggingFaceH4/ultrachat_200k...")
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        subset = dataset.select(range(min(15000, len(dataset))))
        filepath = os.path.join(DATASETS_DIR, "ultrachat_subset.json")
        subset.to_json(filepath)
        print(f"[SUCCESS] Saved UltraChat subset to {filepath} ({len(subset)} items)")
    except Exception as e:
        print(f"[ERROR] Failed to download UltraChat: {e}")

def download_sharegpt():
    print("\n--- Downloading ShareGPT (Subset) ---")
    try:
        print("Loading anon8231489123/ShareGPT_Vicuna_unfiltered...")
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train")
        subset = dataset.select(range(min(10000, len(dataset))))
        filepath = os.path.join(DATASETS_DIR, "sharegpt_subset.json")
        subset.to_json(filepath)
        print(f"[SUCCESS] Saved ShareGPT subset to {filepath} ({len(subset)} items)")
    except Exception as e:
        print(f"[ERROR] Failed to download ShareGPT: {e}")

def main():
    ensure_dirs()
    if not os.path.exists(os.path.join(DATASETS_DIR, "alpaca.json")):
        download_alpaca()
    else:
        print("Alpaca dataset already exists, skipping.")
        
    if not os.path.exists(os.path.join(DATASETS_DIR, "dolly_15k.jsonl")):
        download_dolly()
    else:
        print("Dolly 15K dataset already exists, skipping.")
        
    if not os.path.exists(os.path.join(DATASETS_DIR, "openhermes_subset.json")):
        download_openhermes()
    else:
        print("OpenHermes subset dataset already exists, skipping.")
        
    if not os.path.exists(os.path.join(DATASETS_DIR, "ultrachat_subset.json")):
        download_ultrachat()
    else:
        print("UltraChat subset dataset already exists, skipping.")
        
    if not os.path.exists(os.path.join(DATASETS_DIR, "sharegpt_subset.json")):
        download_sharegpt()
    else:
        print("ShareGPT subset dataset already exists, skipping.")
    print("\n--- Dataset download task finished ---")

if __name__ == "__main__":
    main()
