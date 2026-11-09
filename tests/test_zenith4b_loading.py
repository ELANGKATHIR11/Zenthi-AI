"""
Tests for Zenith-4B Model Loading and Tokenizer Verification.
"""

import os
import json
import pytest
from transformers import AutoTokenizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Zenith-4B")

def test_model_directory_structure():
    assert os.path.isdir(MODEL_PATH), f"Model directory not found at {MODEL_PATH}"
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "mmproj-zenith-4b.gguf"
    ]
    for rf in required_files:
        fp = os.path.join(MODEL_PATH, rf)
        assert os.path.exists(fp), f"Missing required model artifact: {rf}"

def test_tokenizer_loading_and_special_tokens():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    assert tokenizer.vocab_size == 151643 or len(tokenizer) >= 151643
    assert tokenizer.eos_token_id == 151645 or tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.pad_token_id == 151643 or tokenizer.pad_token == "<|endoftext|>"
    
    # Test ChatML encoding
    test_text = "<|im_start|>user\nHello Zenith-4B<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(test_text)
    assert len(encoded["input_ids"]) > 0

def test_safetensors_index_validity():
    idx_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    assert "weight_map" in idx
    assert len(idx["weight_map"]) > 200
    assert "model.embed_tokens.weight" in idx["weight_map"]
    assert "model.norm.weight" in idx["weight_map"]
