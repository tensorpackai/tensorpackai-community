"""
Simple Rule-Based Static Governor (Community Edition).
Demonstrates threshold-based reactive clock stepping (analogous to Linux ondemand governor).

Note: For production workloads requiring zero-hunting, phase-aligned prefill/decode DVFS
with latency amortization and strict SLA enforcement, see TensorPackAI Enterprise.
"""

import time
import logging
from typing import Optional
from tensorpack import FastNVML
from tensorpack.utils.hardware_detect import resolve_gpu_tier, GPU_PROFILES

logger = logging.getLogger("TensorPack-StaticGov")

class StaticRuleGovernor:
    """
    Rule-based, coarse-threshold reactive frequency governor.
    Down-scales GPU SM core clock when utilization drops below threshold,
    and scales up to f_max when utilization exceeds upper threshold.
    """
    def __init__(self, gpu_id: int = 0, high_threshold: int = 80, low_threshold: int = 30, mock: bool = False):
        self.gpu_id = gpu_id
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.nvml = FastNVML()
        if mock:
            self.nvml.set_mock_mode(True)

        t = self.nvml.get_telemetry(gpu_id)
        self.tier = resolve_gpu_tier(t.device_name)
        self.profile = GPU_PROFILES.get(self.tier, GPU_PROFILES["H100"])
        self.current_clock = self.profile["f_max_mhz"]

    def step(self):
        """Executes a single monitoring and frequency adjustment step."""
        t = self.nvml.get_telemetry(self.gpu_id)
        util = t.gpu_utilization_pct

        if util >= self.high_threshold:
            # High utilization: Boost to peak compute frequency f_max
            target_clock = self.profile["f_max_mhz"]
        elif util < self.low_threshold:
            # Very low utilization: Drop to idle clock f_min
            target_clock = self.profile["f_min_mhz"]
        else:
            # Moderate utilization: Memory-paced frequency f_mem
            target_clock = self.profile["f_mem_mhz"]

        if target_clock != self.current_clock:
            logger.info(f"GPU {self.gpu_id} util={util}%: Stepping clock {self.current_clock}MHz -> {target_clock}MHz")
            self.nvml.set_locked_clocks(self.gpu_id, target_clock, target_clock)
            self.current_clock = target_clock
