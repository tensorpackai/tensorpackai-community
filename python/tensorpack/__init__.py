"""
TensorPackAI - Silicon & Node-Level LLM Runtime Energy Optimizer (Community Edition)
Official Website: https://tensorpackai.com
License: Apache 2.0

Open-source real-time GPU energy telemetry, Joules-per-Token exporter,
and hardware frequency management for vLLM and NVIDIA Triton.
"""

__version__ = "0.1.0"

try:
    from ._tensorpack_csrc import FastNVML, GpuTelemetry, DeviceClockRange
except ImportError:
    # Python fallback emulation if C++ extension has not been compiled yet
    class FastNVML:
        def __init__(self):
            pass
        def is_available(self):
            return False
        def is_mock_mode(self):
            return True
        def get_device_count(self):
            return 1
        def get_driver_version(self):
            return "Emulated (C-Extension not built)"
        def get_telemetry(self, idx):
            class MockTelemetry:
                device_index = idx
                device_name = "Mock Accelerator (Build C-Extension for Hardware NVML)"
                power_mw = 250000
                temperature_c = 50
                sm_clock_mhz = 1095
                mem_clock_mhz = 2619
                memory_used_bytes = 20 * 1024**3
                memory_total_bytes = 80 * 1024**3
                gpu_utilization_pct = 70
                mem_utilization_pct = 85
            return MockTelemetry()
        def get_all_telemetry(self):
            return [self.get_telemetry(0)]
        def set_locked_clocks(self, idx, min_mhz, max_mhz):
            return True
        def reset_locked_clocks(self, idx):
            return True

__all__ = ["FastNVML", "__version__"]
