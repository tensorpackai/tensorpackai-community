"""
tensorpack-tune: Developer CLI tool for manual GPU frequency pinning, telemetry monitoring, and energy profiling.
"""

import sys
import click
from tabulate import tabulate
from tensorpack import FastNVML
from tensorpack.utils.hardware_detect import resolve_gpu_tier, GPU_PROFILES

@click.group()
def main():
    """TensorPackAI Silicon & Node-Level GPU Frequency Tuning CLI."""
    pass

@main.command()
@click.option("--mock", is_flag=True, help="Force mock/emulation mode for testing without physical GPUs.")
def status(mock):
    """Display real-time GPU power dissipation, temperature, clock setpoints, VRAM, and compute utilization."""
    nvml = FastNVML()
    if mock:
        nvml.set_mock_mode(True)

    telemetry_list = nvml.get_all_telemetry()
    table_data = []

    for t in telemetry_list:
        tier = resolve_gpu_tier(t.device_name)
        power_w = t.power_mw / 1000.0
        vram_used_gb = t.memory_used_bytes / (1024**3)
        vram_total_gb = t.memory_total_bytes / (1024**3)

        table_data.append([
            t.device_index,
            t.device_name,
            tier,
            f"{power_w:.1f} W",
            f"{t.temperature_c} °C",
            f"{t.sm_clock_mhz} MHz",
            f"{t.mem_clock_mhz} MHz",
            f"{vram_used_gb:.1f} / {vram_total_gb:.1f} GB",
            f"{t.gpu_utilization_pct}%",
        ])

    headers = ["GPU", "Device Name", "Tier", "Power", "Temp", "SM Clock", "Mem Clock", "VRAM Used/Total", "SM Util"]
    click.echo(f"\nTensorPackAI GPU Telemetry (Driver: {nvml.get_driver_version()}):")
    click.echo(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    click.echo("")

@main.command()
@click.option("--gpu", required=True, type=str, help="GPU index (e.g., 0) or 'all'.")
@click.option("--clock", required=True, type=int, help="Target SM clock frequency in MHz.")
def pin(gpu, clock):
    """Pin GPU core clock frequency to a static value (Requires CAP_SYS_ADMIN or root)."""
    nvml = FastNVML()
    count = nvml.get_device_count()

    indices = list(range(count)) if gpu.lower() == "all" else [int(gpu)]

    for idx in indices:
        click.echo(f"Locking GPU {idx} core clock to {clock} MHz...")
        success = nvml.set_locked_clocks(idx, clock, clock)
        if success:
            click.echo(f"  [SUCCESS] GPU {idx} locked at {clock} MHz.")
        else:
            click.echo(f"  [ERROR] Failed to lock GPU {idx}. Ensure CAP_SYS_ADMIN or root privileges.", err=True)

@main.command()
@click.option("--gpu", default="all", type=str, help="GPU index (e.g., 0) or 'all'.")
def reset(gpu):
    """Reset GPU locked clocks to default dynamic vendor curves."""
    nvml = FastNVML()
    count = nvml.get_device_count()

    indices = list(range(count)) if gpu.lower() == "all" else [int(gpu)]

    for idx in indices:
        click.echo(f"Resetting GPU {idx} clocks to default...")
        success = nvml.reset_locked_clocks(idx)
        if success:
            click.echo(f"  [SUCCESS] GPU {idx} clocks restored to default dynamic curve.")
        else:
            click.echo(f"  [ERROR] Failed to reset GPU {idx}.", err=True)

@main.command()
@click.option("--gpu", default=0, type=int, help="Target GPU index.")
def profile(gpu):
    """Inspect recommended phase-aware frequency profile for the detected GPU architecture."""
    nvml = FastNVML()
    t = nvml.get_telemetry(gpu)
    tier = resolve_gpu_tier(t.device_name)
    prof = GPU_PROFILES.get(tier, GPU_PROFILES["H100"])

    click.echo(f"\n=======================================================")
    click.echo(f" TensorPackAI Profile for GPU {gpu}: {t.device_name} ({tier})")
    click.echo(f"=======================================================")
    click.echo(f" • Architecture:                 {prof['architecture']}")
    click.echo(f" • TDP / Baseline Idle:          {prof['tdp_watts']} W / {prof['idle_watts']} W")
    click.echo(f" • Prefill Boost Clock (f_max):  {prof['f_max_mhz']} MHz  (Peak GEMM Compute Bound)")
    click.echo(f" • Decode Throttle Clock (f_mem):{prof['f_mem_mhz']} MHz  (Memory-Paced GEMV Bound)")
    click.echo(f" • Idle Power-Gated Clock (f_min):{prof['f_min_mhz']} MHz (Deep Energy Gating)")
    click.echo(f" • Estimated Decode Dynamic Drop:~72.9% power reduction (P_dyn ∝ V^2 * f)")
    click.echo(f"=======================================================\n")

if __name__ == "__main__":
    main()
