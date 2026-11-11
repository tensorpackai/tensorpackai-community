"""
Prometheus Metrics Exporter for TensorPackAI.
Exposes real hardware telemetry (Watts, Clocks, Temp, Cumulative Joules) from NVML,
and computes measured Joules-per-Token and Tokens-per-Watt by scraping live vLLM
token counters (`vllm:generation_tokens_total`) or receiving in-process observer events.
"""

import re
import time
import threading
import urllib.request
from typing import Dict, List, Optional
import click
from prometheus_client import start_http_server, Gauge, Counter
from tensorpack import FastNVML
from tensorpack.utils.hardware_detect import resolve_gpu_tier

# Hardware Telemetry Gauges & Counters
GPU_POWER = Gauge(
    "tensorpack_gpu_power_watts",
    "Instantaneous GPU electrical power dissipation in Watts",
    ["gpu", "model", "tier"],
)
GPU_ENERGY_TOTAL = Counter(
    "tensorpack_gpu_energy_joules_total",
    "Cumulative electrical energy consumed by the GPU in Joules (time-integrated power)",
    ["gpu", "tier"],
)
GPU_TEMP = Gauge(
    "tensorpack_gpu_temperature_celsius",
    "GPU core temperature in Celsius",
    ["gpu", "tier"],
)
GPU_SM_CLOCK = Gauge(
    "tensorpack_gpu_sm_clock_mhz",
    "Current SM operating frequency in MHz",
    ["gpu", "tier"],
)
GPU_MEM_CLOCK = Gauge(
    "tensorpack_gpu_mem_clock_mhz",
    "Current Memory operating frequency in MHz",
    ["gpu", "tier"],
)
GPU_UTIL = Gauge(
    "tensorpack_gpu_utilization_ratio",
    "GPU compute utilization ratio (0.0 to 1.0)",
    ["gpu", "tier"],
)

# Measured Token-Efficiency Metrics (populated when token stream is active)
GENERATION_TOKENS_TOTAL = Counter(
    "tensorpack_generation_tokens_total",
    "Total generated tokens observed from vLLM metrics or runtime hooks",
)
JOULES_PER_TOKEN = Gauge(
    "tensorpack_joules_per_token",
    "Measured electrical energy consumed per generated output token (Joules/Token) over the active window",
)
CUMULATIVE_JOULES_PER_TOKEN = Gauge(
    "tensorpack_cumulative_joules_per_token",
    "Cumulative electrical energy per generated output token (Joules/Token) since exporter start",
)
TOKENS_PER_WATT = Gauge(
    "tensorpack_tokens_per_watt",
    "Measured token throughput per Watt of active cluster power (Tokens/s per Watt)",
)

_VLLM_GEN_TOKEN_RE = re.compile(
    r"^vllm:generation_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
    re.MULTILINE,
)


def fetch_vllm_generation_tokens(metrics_urls: List[str], timeout: float = 1.0) -> Optional[float]:
    """Scrapes `vllm:generation_tokens_total` across one or more vLLM Prometheus endpoints."""
    if not metrics_urls:
        return None
    total = 0.0
    any_success = False
    for url in metrics_urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            matches = _VLLM_GEN_TOKEN_RE.findall(body)
            if matches:
                total += sum(float(m) for m in matches)
                any_success = True
        except Exception:
            continue
    return total if any_success else None


class PrometheusTelemetryCollector:
    """
    Integrates real NVML power readings over monotonic time intervals (dE = P * dt)
    and pairs them with actual token generation counts from vLLM or in-process hooks.
    """

    def __init__(
        self,
        sample_interval_s: float = 0.5,
        vllm_metrics_urls: Optional[List[str]] = None,
        mock: bool = False,
    ):
        self.interval = sample_interval_s
        self.vllm_metrics_urls = vllm_metrics_urls or []
        self.nvml = FastNVML()
        if mock:
            self.nvml.set_mock_mode(True)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Monotonic integration state
        self._last_ts: Optional[float] = None
        self._last_power_w: Dict[int, float] = {}
        self._cumulative_energy_j: float = 0.0
        self._window_energy_j: float = 0.0

        # Token tracking state
        self._inproc_tokens_total: float = 0.0
        self._last_scraped_tokens: Optional[float] = None
        self._cumulative_tokens: float = 0.0
        self._window_tokens: float = 0.0

    def record_tokens(self, count: int):
        """Thread-safe method for in-process observers (BasePhaseObserver) to report generated tokens."""
        if count <= 0:
            return
        with self._lock:
            self._inproc_tokens_total += count
            self._cumulative_tokens += count
            self._window_tokens += count
            GENERATION_TOKENS_TOTAL.inc(count)

    def start(self):
        self._running = True
        self._last_ts = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def sample_once(self):
        """Performs a single telemetry scrape and energy integration step."""
        now = time.monotonic()
        dt = (now - self._last_ts) if self._last_ts is not None else 0.0
        self._last_ts = now

        telemetries = self.nvml.get_all_telemetry()
        step_energy_j = 0.0

        for t in telemetries:
            gpu_idx = t.device_index
            gpu_str = str(gpu_idx)
            tier = resolve_gpu_tier(t.device_name)
            power_w = t.power_mw / 1000.0

            GPU_POWER.labels(gpu=gpu_str, model=t.device_name, tier=tier).set(power_w)
            GPU_TEMP.labels(gpu=gpu_str, tier=tier).set(t.temperature_c)
            GPU_SM_CLOCK.labels(gpu=gpu_str, tier=tier).set(t.sm_clock_mhz)
            GPU_MEM_CLOCK.labels(gpu=gpu_str, tier=tier).set(t.mem_clock_mhz)
            GPU_UTIL.labels(gpu=gpu_str, tier=tier).set(t.gpu_utilization_pct / 100.0)

            if dt > 0.0:
                # Trapezoidal integration between previous and current power reading
                prev_p = self._last_power_w.get(gpu_idx, power_w)
                gpu_de = 0.5 * (prev_p + power_w) * dt
                GPU_ENERGY_TOTAL.labels(gpu=gpu_str, tier=tier).inc(gpu_de)
                step_energy_j += gpu_de

            self._last_power_w[gpu_idx] = power_w

        # Scrape external vLLM metrics if configured
        scraped_total = fetch_vllm_generation_tokens(self.vllm_metrics_urls)
        with self._lock:
            self._cumulative_energy_j += step_energy_j
            self._window_energy_j += step_energy_j

            if scraped_total is not None:
                if self._last_scraped_tokens is not None:
                    delta_tok = max(0.0, scraped_total - self._last_scraped_tokens)
                    if delta_tok > 0:
                        self._cumulative_tokens += delta_tok
                        self._window_tokens += delta_tok
                        GENERATION_TOKENS_TOTAL.inc(delta_tok)
                self._last_scraped_tokens = scraped_total

            # Update efficiency gauges strictly from real measured deltas
            if self._window_tokens > 0 and self._window_energy_j > 0:
                j_per_tok = self._window_energy_j / self._window_tokens
                tok_per_w = self._window_tokens / self._window_energy_j
                JOULES_PER_TOKEN.set(j_per_tok)
                TOKENS_PER_WATT.set(tok_per_w)
                # Reset rolling window once measured
                self._window_energy_j = 0.0
                self._window_tokens = 0.0
            elif dt > 0.0 and self._window_tokens == 0:
                # No tokens generated in this interval -> 0 throughput per watt
                TOKENS_PER_WATT.set(0.0)

            if self._cumulative_tokens > 0 and self._cumulative_energy_j > 0:
                CUMULATIVE_JOULES_PER_TOKEN.set(
                    self._cumulative_energy_j / self._cumulative_tokens
                )

    def _loop(self):
        while self._running:
            try:
                self.sample_once()
            except Exception:
                pass
            time.sleep(self.interval)


@click.command()
@click.option("--port", default=9100, type=int, help="Prometheus exporter HTTP server port (default: 9100).")
@click.option("--interval", default=0.5, type=float, help="Sampling telemetry interval in seconds.")
@click.option(
    "--vllm-url",
    multiple=True,
    default=["http://127.0.0.1:8000/metrics"],
    help="vLLM Prometheus /metrics URL(s) to scrape `vllm:generation_tokens_total` from.",
)
@click.option("--mock", is_flag=True, help="Run in mock/emulation mode for environments without NVIDIA GPUs.")
def main(port: int, interval: float, vllm_url: tuple, mock: bool):
    """Launch TensorPackAI Prometheus Metrics Exporter."""
    urls = list(vllm_url)
    click.echo(f"Starting TensorPackAI Prometheus Exporter on port {port} (Interval: {interval}s)...")
    click.echo(f"Scraping vLLM token counters from: {urls}")
    collector = PrometheusTelemetryCollector(
        sample_interval_s=interval,
        vllm_metrics_urls=urls,
        mock=mock,
    )
    start_http_server(port)
    collector.start()
    click.echo(f"Metrics exposed at http://0.0.0.0:{port}/metrics")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping exporter.")
        collector.stop()


if __name__ == "__main__":
    main()
