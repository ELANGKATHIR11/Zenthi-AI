"""
Zenthi-AI Dataset Ingestion Pipeline
Downloads and prepares the 3 curated open datasets requested for Zenithi:
1. Python Codes 25K (Python code generation and programming instructions)
2. OpenAssistant OASST1 (English, high-quality, reviewed instruction pairs)
3. CORD (Consolidated Receipt Dataset for Document & Table OCR)
"""

import os
import sys
import json
import time
import requests
from typing import Dict, Any, List
from datasets import load_dataset
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
CORD_IMAGES_DIR = os.path.join(DATASETS_DIR, "cord_images")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(CORD_IMAGES_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# 1. Python Codes 25K
# -------------------------------------------------------------------------
def download_python_codes_25k():
    output_path = os.path.join(DATASETS_DIR, "python_codes_25k.jsonl")
    url = "https://huggingface.co/datasets/flytech/python-codes-25k/resolve/main/python-codes-25k.jsonl"

    print("\n=======================================================")
    print(" [1/3] Downloading: Python Codes 25K (~25.4 MB)")
    print(f" URL: {url}")
    print(f" Target: {output_path}")
    print("=======================================================")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 20 * 1024 * 1024:
        print(f"[SKIP] python_codes_25k.jsonl already exists ({os.path.getsize(output_path)/1024/1024:.2f} MB).")
        return output_path

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0
    start_time = time.time()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100 if total_size else 0
                elapsed = time.time() - start_time
                speed = (downloaded / 1024 / 1024) / elapsed if elapsed > 0 else 0
                print(f"\rDownloading: {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB ({percent:.1f}%) @ {speed:.2f} MB/s", end="", flush=True)

    print(f"\n[SUCCESS] Saved Python Codes 25K to {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
    return output_path


# -------------------------------------------------------------------------
# 2. OpenAssistant OASST1 (English Curated)
# -------------------------------------------------------------------------
def download_and_filter_oasst1(target_samples: int = 3000):
    output_path = os.path.join(DATASETS_DIR, "oasst1_curated.jsonl")

    print("\n=======================================================")
    print(f" [2/3] Downloading & Curating: OpenAssistant OASST1 ({target_samples} samples)")
    print(f" Target: {output_path}")
    print("=======================================================")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1 * 1024 * 1024:
        print(f"[SKIP] oasst1_curated.jsonl already exists ({os.path.getsize(output_path)/1024/1024:.2f} MB).")
        return output_path

    print("Loading OpenAssistant/oasst1 train split...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    print(f"Loaded {len(ds)} raw messages from OASST1. Indexing conversation trees...")

    # Map message_id to message for tree lookup
    msg_by_id = {}
    for item in ds:
        msg_by_id[item["message_id"]] = item

    # Filter assistant responses in English, reviewed, not deleted, rank == 0 (best)
    curated_pairs = []
    for item in ds:
        if (
            item.get("role") == "assistant"
            and item.get("lang") == "en"
            and not item.get("deleted", False)
            and item.get("review_result", False)
            and item.get("rank") in (0, None)
        ):
            parent_id = item.get("parent_id")
            if parent_id and parent_id in msg_by_id:
                parent = msg_by_id[parent_id]
                prompt_text = parent.get("text", "").strip()
                resp_text = item.get("text", "").strip()

                if len(prompt_text) > 10 and len(resp_text) > 20:
                    curated_pairs.append({
                        "id": f"oasst1_{item['message_id'][:12]}",
                        "instruction": prompt_text,
                        "response": resp_text
                    })
                    if len(curated_pairs) >= target_samples:
                        break

    print(f"Extracted {len(curated_pairs)} high-quality English instruction pairs.")

    with open(output_path, "w", encoding="utf-8") as f:
        for p in curated_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved curated OASST1 to {output_path} ({len(curated_pairs)} pairs, {os.path.getsize(output_path)/1024/1024:.2f} MB)")
    return output_path


# -------------------------------------------------------------------------
# 3. CORD (Consolidated Receipt Dataset for Post-OCR Parsing)
# -------------------------------------------------------------------------
def download_and_process_cord(num_images: int = 150):
    output_jsonl = os.path.join(DATASETS_DIR, "cord_train.jsonl")

    print("\n=======================================================")
    print(f" [3/3] Downloading & Processing CORD Dataset ({num_images} images)")
    print(f" Images Directory: {CORD_IMAGES_DIR}")
    print(f" JSONL Annotations: {output_jsonl}")
    print("=======================================================")

    if os.path.exists(output_jsonl) and os.path.getsize(output_jsonl) > 100 * 1024:
        print(f"[SKIP] cord_train.jsonl already exists ({os.path.getsize(output_jsonl)/1024:.1f} KB).")
        return output_jsonl

    print(f"Loading {num_images} samples from naver-clova-ix/cord-v2...")
    ds = load_dataset("naver-clova-ix/cord-v2", split=f"train[:{num_images}]")

    records = []
    print(f"Processing and resizing {len(ds)} receipt images to 768px max budget...")
    for idx, item in enumerate(ds):
        img: Image.Image = item["image"]
        gt_raw = item.get("ground_truth", "{}")

        # Parse ground truth JSON
        try:
            gt_data = json.loads(gt_raw)
        except Exception:
            gt_data = {"raw": gt_raw}

        # Resize image keeping aspect ratio (max dimension 768px)
        max_dim = 768
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_filename = f"cord_receipt_{idx:04d}.png"
        img_path = os.path.join(CORD_IMAGES_DIR, img_filename)
        img.save(img_path, format="PNG", optimize=True)

        # Extract structured text tokens if available
        valid_lines = gt_data.get("valid_line", [])
        extracted_lines = []
        for line in valid_lines:
            words = line.get("words", [])
            line_text = " ".join(w.get("text", "") for w in words if w.get("text"))
            if line_text:
                extracted_lines.append(line_text)

        full_transcription = "\n".join(extracted_lines) if extracted_lines else str(gt_data)

        records.append({
            "id": f"cord_{idx:04d}",
            "image_path": os.path.relpath(img_path, BASE_DIR).replace("\\", "/"),
            "image_size": img.size,
            "transcription": full_transcription,
            "ground_truth": gt_data
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved {len(records)} CORD images and annotations to {output_jsonl}")
    return output_jsonl


def main():
    print("=" * 60)
    print(" Zenthi-AI: Curated Dataset Download & Preparation Task")
    print(" 1. Python Codes 25K (Coding / Logic)")
    print(" 2. OpenAssistant OASST1 (General Instruction / Reasoning)")
    print(" 3. CORD (Receipt & Document OCR / Table Extraction)")
    print("=" * 60)

    # 1. Python Codes 25K
    download_python_codes_25k()

    # 2. OpenAssistant OASST1
    download_and_filter_oasst1(target_samples=3000)

    # 3. CORD Dataset
    download_and_process_cord(num_images=150)

    print("\n" + "=" * 60)
    print(" [COMPLETE] All 3 Zenithi datasets downloaded and processed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

