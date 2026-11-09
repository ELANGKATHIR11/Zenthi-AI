"""
POLAR-FLASH Optimization Bridge for Zenith-4B & Zenthi-AI OS.

Connects Zenith foundation model layers to POLAR-FLASH v1.1.0:
- Mathematical Polar Coordinate Quantization
- Memory-Mapped Streaming Decompression (.pflash format)
- Effective bit-per-weight (bpw) compression accounting
- Zero-copy weight retrieval
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure project root in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from polarflash_core.formats.pflash_format import PFlashReader, PFlashWriter
from polarflash_quant.polar.polar_quant import RecursivePolarQuant, PolarQuant2D

logger = logging.getLogger("PolarFlashBridge")


class PolarFlashBridge:
    """Manages POLAR-FLASH compressed weight storage and runtime streaming."""

    def __init__(self, default_pflash_path: Optional[str] = None):
        self.default_pflash_path = default_pflash_path or os.path.join(
            BASE_DIR, "models", "sample_zenith_polar.pflash"
        )

    def inspect_archive(self, pflash_path: Optional[str] = None) -> Dict[str, Any]:
        """Reads and analyzes metadata, catalog, and compression metrics of a .pflash archive."""
        target_path = pflash_path or self.default_pflash_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f".pflash file not found: {target_path}")

        file_size_bytes = os.path.getsize(target_path)
        with PFlashReader(target_path) as reader:
            metadata = reader.metadata or {}
            catalog = reader.tensor_catalog or {}
            
            total_elements = 0
            total_payload_bytes = 0
            tensor_details = []
            
            for name, meta in catalog.items():
                shape = meta.get("shape", [])
                numel = int(np.prod(shape)) if shape else 0
                payload_bytes = meta.get("size", 0)
                q_meta = meta.get("quant_meta", {})
                
                total_elements += numel
                total_payload_bytes += payload_bytes
                
                tensor_details.append({
                    "name": name,
                    "shape": shape,
                    "num_elements": numel,
                    "payload_bytes": payload_bytes,
                    "dtype": meta.get("dtype", "unknown"),
                    "effective_bpw": q_meta.get("effective_nominal_bpw", 4.0),
                    "snr_db": round(q_meta.get("snr_db", 0.0), 2)
                })

        fp16_equivalent_bytes = total_elements * 2
        fp32_equivalent_bytes = total_elements * 4
        compression_ratio_fp16 = round(fp16_equivalent_bytes / max(file_size_bytes, 1), 2)
        compression_ratio_fp32 = round(fp32_equivalent_bytes / max(file_size_bytes, 1), 2)
        effective_bpw = round((file_size_bytes * 8) / max(total_elements, 1), 3)

        return {
            "archive_path": target_path,
            "archive_size_mb": round(file_size_bytes / (1024 ** 2), 2),
            "tensor_count": len(catalog),
            "total_weight_elements": total_elements,
            "effective_bpw": effective_bpw,
            "compression_ratio_vs_fp16": compression_ratio_fp16,
            "compression_ratio_vs_fp32": compression_ratio_fp32,
            "source_model": metadata.get("source_model", "Zenith-4B"),
            "backend": metadata.get("backend", "polar"),
            "tensors": tensor_details
        }

    def stream_tensor(self, tensor_name: str, pflash_path: Optional[str] = None) -> bytes:
        """Retrieves raw compressed bytes for a specific tensor using memory mapping."""
        target_path = pflash_path or self.default_pflash_path
        with PFlashReader(target_path) as reader:
            if tensor_name not in reader.tensor_catalog:
                raise KeyError(f"Tensor '{tensor_name}' not in .pflash catalog")
            return reader.get_tensor_bytes(tensor_name)

    def benchmark_streaming_throughput(
        self,
        pflash_path: Optional[str] = None,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Profiles memory-mapped disk retrieval throughput in MB/sec."""
        target_path = pflash_path or self.default_pflash_path
        info = self.inspect_archive(target_path)
        tensor_names = [t["name"] for t in info["tensors"]]
        
        total_bytes_read = 0
        t0 = time.perf_counter()
        
        for _ in range(iterations):
            with PFlashReader(target_path) as reader:
                for name in tensor_names:
                    data = reader.get_tensor_bytes(name)
                    total_bytes_read += len(data)
                    
        elapsed = time.perf_counter() - t0
        mb_read = total_bytes_read / (1024 ** 2)
        throughput_mb_s = round(mb_read / max(elapsed, 0.0001), 2)
        
        return {
            "archive": os.path.basename(target_path),
            "iterations": iterations,
            "total_mb_read": round(mb_read, 2),
            "elapsed_seconds": round(elapsed, 4),
            "throughput_mb_per_sec": throughput_mb_s,
            "effective_bpw": info["effective_bpw"],
            "compression_ratio_fp16": info["compression_ratio_vs_fp16"]
        }
