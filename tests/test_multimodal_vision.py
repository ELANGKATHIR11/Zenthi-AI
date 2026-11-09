"""
Tests for Zenith Multimodal Vision Processing and Grounding.
"""

import os
import io
import base64
import pytest
from PIL import Image

from zenith_engine.multimodal_inference import ZenithMultimodalEngine

@pytest.fixture
def engine():
    # Initialize engine without loading heavy weights for fast testing
    return ZenithMultimodalEngine(enable_native_loading=False)

def test_process_image_input_pil(engine):
    img = Image.new("RGB", (640, 480), color="blue")
    meta = engine.process_image_input(img)
    assert meta["width"] == 640
    assert meta["height"] == 480
    assert meta["aspect_ratio"] == round(640 / 480, 3)

def test_process_image_input_base64(engine):
    img = Image.new("RGB", (200, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    meta = engine.process_image_input(b64_str)
    assert meta["width"] == 200
    assert meta["height"] == 100
    assert meta["aspect_ratio"] == 2.0

def test_chatml_prompt_generation(engine):
    messages = [
        {"role": "user", "content": "Analyze this architecture"},
        {"role": "assistant", "content": "I see three services"},
        {"role": "user", "content": "Where is the failure point?"}
    ]
    prompt = engine.build_chatml_prompt(messages)
    assert "<|im_start|>system\n" in prompt
    assert "<|im_start|>user\nAnalyze this architecture<|im_end|>\n" in prompt
    assert "<|im_start|>assistant\nI see three services<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
