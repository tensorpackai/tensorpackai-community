"""
Hardware detection utilities for TensorPackAI.
Maps raw device names to canonical accelerator profiles (T4, A10G, A100, H100, B200).
"""

from typing import Dict, Any

GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "B200": {
        "architecture": "Blackwell",
        "tdp_watts": 1000,
        "idle_watts": 200,
        "f_max_mhz": 2100,
        "f_mem_mhz": 1140,
        "f_min_mhz": 600,
        "vram_gb": 192,
    },
    "H100": {
        "architecture": "Hopper",
        "tdp_watts": 700,
        "idle_watts": 150,
        "f_max_mhz": 1980,
        "f_mem_mhz": 1095,
        "f_min_mhz": 540,
        "vram_gb": 80,
    },
    "A100": {
        "architecture": "Ampere",
        "tdp_watts": 400,
        "idle_watts": 110,
        "f_max_mhz": 1410,
        "f_mem_mhz": 930,
        "f_min_mhz": 420,
        "vram_gb": 80,
    },
    "A10G": {
        "architecture": "Ampere",
        "tdp_watts": 150,
        "idle_watts": 45,
        "f_max_mhz": 1695,
        "f_mem_mhz": 1110,
        "f_min_mhz": 300,
        "vram_gb": 24,
    },
    "T4": {
        "architecture": "Turing",
        "tdp_watts": 70,
        "idle_watts": 25,
        "f_max_mhz": 1590,
        "f_mem_mhz": 1020,
        "f_min_mhz": 300,
        "vram_gb": 16,
    },
}

def resolve_gpu_tier(device_name: str) -> str:
    """Resolves raw vendor string into canonical architecture tier."""
    name_upper = device_name.upper()
    for tier in ["B200", "H100", "A100", "A10G", "T4"]:
        if tier in name_upper:
            return tier
    return "H100"  # Default modern reference tier
