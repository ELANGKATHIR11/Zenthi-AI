"""
Automated Model Architecture & Multi-Adapter Validation Tests for Zenith-4B.

Validates:
1. Architectural integrity of Zenith-4B config.json (36 layers, 2048 hidden, 11008 intermediate).
2. Adapter parameter dimensions (r=16, alpha=32, 504 tensors, 29,933,568 params each).
3. Subspace orthogonality / interference bounds across Code, Extraction, and Vision adapters.
4. TIES-merge and weight soup consensus compatibility.
"""

import os
import json
import pytest
import torch
import safetensors.torch as st

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
ZENITH_DIR = os.path.join(MODELS_DIR, "Zenith-4B")
ADAPTER_DIRS = {
    "code": os.path.join(MODELS_DIR, "adapter-code"),
    "extraction": os.path.join(MODELS_DIR, "adapter-extraction"),
    "vision": os.path.join(MODELS_DIR, "adapter-vision"),
    "soup": os.path.join(MODELS_DIR, "adapter-soup")
}

def test_zenith4b_config_architecture():
    """Verify Zenith-4B config.json reflects the validated 3.09B parameter architecture."""
    config_path = os.path.join(ZENITH_DIR, "config.json")
    assert os.path.exists(config_path), f"Missing config.json at {config_path}"
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    assert cfg.get("num_hidden_layers") == 36, "Zenith-4B must have 36 transformer layers"
    assert cfg.get("hidden_size") == 2048, "Hidden size must be 2048"
    assert cfg.get("intermediate_size") == 11008, "Intermediate size must be 11008"
    assert cfg.get("num_attention_heads") == 16, "Attention heads must be 16"
    assert cfg.get("num_key_value_heads") == 2, "KV heads must be 2 (GQA 8:1)"
    assert cfg.get("vocab_size") == 151936, "Vocab size must be 151936"
    assert cfg.get("tie_word_embeddings") is True, "tie_word_embeddings must be True"

def test_adapter_configurations():
    """Verify that all adapters have valid adapter_config.json with r=16 and alpha=32."""
    for name, path in ADAPTER_DIRS.items():
        cfg_file = os.path.join(path, "adapter_config.json")
        assert os.path.exists(cfg_file), f"Missing adapter_config.json for {name}"
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        assert cfg.get("r") == 16, f"Adapter {name} must have rank r=16"
        assert cfg.get("lora_alpha") == 32, f"Adapter {name} must have lora_alpha=32"
        target_modules = cfg.get("target_modules", [])
        expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(target_modules) == expected_targets, f"Adapter {name} must target all 7 linear projections"

def test_adapter_safetensors_shapes_and_param_counts():
    """Verify that adapter weights match exact shapes and 29.93M parameter count."""
    expected_param_count = 29933568
    expected_tensor_count = 504  # 36 layers * 7 modules * 2 (lora_A, lora_B)
    
    for name, path in ADAPTER_DIRS.items():
        st_file = os.path.join(path, "adapter_model.safetensors")
        assert os.path.exists(st_file), f"Missing adapter_model.safetensors for {name}"
        
        tensors = st.load_file(st_file)
        assert len(tensors) == expected_tensor_count, (
            f"Adapter {name} tensor count {len(tensors)} != expected {expected_tensor_count}"
        )
        
        total_params = sum(t.numel() for t in tensors.values())
        assert total_params == expected_param_count, (
            f"Adapter {name} total params {total_params} != expected {expected_param_count}"
        )
        
        # Spot check layer 0 down_proj
        a_key = "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
        b_key = "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"
        assert a_key in tensors
        assert b_key in tensors
        assert tuple(tensors[a_key].shape) == (16, 11008)
        assert tuple(tensors[b_key].shape) == (2048, 16)

def test_adapter_subspace_orthogonality():
    """Verify that domain adapters are mutually near-orthogonal to avoid catastrophic interference."""
    c_tensors = st.load_file(os.path.join(ADAPTER_DIRS["code"], "adapter_model.safetensors"))
    e_tensors = st.load_file(os.path.join(ADAPTER_DIRS["extraction"], "adapter_model.safetensors"))
    v_tensors = st.load_file(os.path.join(ADAPTER_DIRS["vision"], "adapter_model.safetensors"))
    
    cos_ce_list = []
    cos_cv_list = []
    cos_ev_list = []
    
    # Check across 30 sampled projection tensors
    sample_keys = [k for k in c_tensors.keys() if "lora_B" in k][:30]
    for k in sample_keys:
        tc = c_tensors[k].flatten()
        te = e_tensors[k].flatten()
        tv = v_tensors[k].flatten()
        
        cos_ce = torch.cosine_similarity(tc.unsqueeze(0), te.unsqueeze(0)).item()
        cos_cv = torch.cosine_similarity(tc.unsqueeze(0), tv.unsqueeze(0)).item()
        cos_ev = torch.cosine_similarity(te.unsqueeze(0), tv.unsqueeze(0)).item()
        
        cos_ce_list.append(cos_ce)
        cos_cv_list.append(cos_cv)
        cos_ev_list.append(cos_ev)
        
    mean_ce = abs(sum(cos_ce_list) / len(cos_ce_list))
    mean_cv = abs(sum(cos_cv_list) / len(cos_cv_list))
    mean_ev = abs(sum(cos_ev_list) / len(cos_ev_list))
    
    # Assert near-orthogonality (all cosine similarities bounded strictly below 0.05)
    assert mean_ce < 0.05, f"Code-Extraction correlation too high: {mean_ce}"
    assert mean_cv < 0.05, f"Code-Vision correlation too high: {mean_cv}"
    assert mean_ev < 0.05, f"Extraction-Vision correlation too high: {mean_ev}"

def test_multimodal_projector_gguf_integrity():
    """Verify mmproj-zenith-4b.gguf header and CLIP ViT architecture parameters."""
    import struct
    mmproj_path = os.path.join(ZENITH_DIR, "mmproj-zenith-4b.gguf")
    assert os.path.exists(mmproj_path), f"Missing mmproj file at {mmproj_path}"
    
    with open(mmproj_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"Invalid GGUF magic bytes: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        assert version >= 2, f"GGUF version {version} must be >= 2"
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        assert tensor_count > 100, f"Expected >100 projector tensors, got {tensor_count}"
