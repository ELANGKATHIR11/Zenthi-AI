"""
Tests for Hardware-Aware Runtime Management on Windows 11 / RTX 5060.
"""

import os
import torch
import pytest
from zenith_engine.runtime_manager import HardwareProfile, RuntimeModelManager

def test_hardware_profile_detection():
    hw = HardwareProfile(max_vram_gb=6.5, max_cpu_threads=4)
    status = hw.get_memory_status()
    
    assert "ram_total_gb" in status
    assert status["ram_total_gb"] > 4.0
    assert "cuda_available" in status
    
    if status["cuda_available"]:
        assert "vram_total_gb" in status
        assert status["vram_total_gb"] > 0
        assert "vram_headroom_mb" in status

def test_thread_limiting():
    HardwareProfile(max_cpu_threads=4)
    # PyTorch get_num_threads
    assert torch.get_num_threads() <= 4

def test_single_model_manager_lifecycle():
    hw = HardwareProfile()
    mgr = RuntimeModelManager(hw)
    
    # Verify singleton
    mgr2 = RuntimeModelManager()
    assert mgr is mgr2
    
    # Test safe unload when no model loaded
    mgr.unload_active_model()
    assert mgr.active_model is None
    assert mgr.active_tokenizer is None
