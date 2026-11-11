"""
Base open-source integration interface for vLLM, NVIDIA Triton, and custom LLM inference runtimes.
Provides passive telemetry hooks without exposing proprietary scheduling or DVFS internals.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

class BasePhaseObserver(ABC):
    """
    Abstract observer interface designed to hook into vLLM or Triton execution loops.
    Allows recording of forward step timings and batch composition metrics.
    """
    @abstractmethod
    def on_step_begin(self, is_prefill: bool, batch_size: int, total_tokens: int):
        """Triggered immediately prior to forward kernel dispatch."""
        pass

    @abstractmethod
    def on_step_end(self, step_duration_ms: float, tokens_generated: int):
        """Triggered upon completion of a forward step for telemetry and Joules-per-Token calculation."""
        pass
