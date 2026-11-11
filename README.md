<div align="center">

# TensorPackAI (Community Edition)
### Silicon & Node-Level LLM Runtime Energy Optimizer for vLLM & NVIDIA Triton

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/c%2B%2B-17-green.svg)](https://isocpp.org/)
[![Prometheus](https://img.shields.io/badge/metrics-Prometheus-orange.svg)](https://prometheus.io/)
[![Hardware Support](https://img.shields.io/badge/GPUs-NVIDIA%20B200%20%7C%20H100%20%7C%20A100%20%7C%20A10G%20%7C%20T4-76B900.svg)](https://www.nvidia.com)

**Real-time GPU power telemetry, Joules-per-Token exporter, and silicon frequency management designed specifically for generative foundation model serving.**

[Official Website](https://tensorpackai.com) • [Documentation](https://tensorpackai.com/docs) • [Enterprise Edition](#-tensorpackai-enterprise) • [Bug Tracker](https://github.com/tensorpackai/tensorpackai-community/issues)

</div>

---

## 📖 Overview

Modern Large Language Model (LLM) serving frameworks such as **vLLM** and **NVIDIA Triton Inference Server** operate GPUs at maximum core clock frequencies ($f_{\max}$) throughout both **prefill** (compute-bound) and **decode** (memory-bound) phases.

Because dynamic CMOS power dissipation scales quadratically with supply voltage and linearly with operating frequency ($P_{\text{dyn}} \propto V^2 f$), sustaining peak clock frequencies during memory-stalled autoregressive decoding wastes immense electrical power without meaningful latency gains.

**TensorPackAI Community Edition** provides the essential developer and telemetry layer:
1. **Zero-Overhead C++17 NVML Engine:** Direct dynamic loading (`dlopen`) of `libnvidia-ml.so` to eliminate Python ctypes latency.
2. **Prometheus Metrics Exporter:** Real-time metrics including `tensorpack_joules_per_token` and `tensorpack_tokens_per_watt` for Grafana dashboards.
3. **Developer CLI (`tensorpack-tune`):** Command-line inspection, static clock pinning, and phase-aware profile recommendations.
4. **Static Rule Governor:** Threshold-based dynamic clock scaling for quick energy optimization experiments.

---

## ⚡ Architecture

```
                    +------------------------------------+
                    |  vLLM / Triton Serving Engine      |
                    +------------------------------------+
                                      |
                           (Token Generation Stream)
                                      v
                    +------------------------------------+
                    |  TensorPackAI Community Exporter   |
                    |  • Real-Time Energy / Token Metric |
                    |  • Tokens per Watt Efficiency      |
                    +------------------------------------+
                                      |
                      +---------------+---------------+
                      |                               |
                      v                               v
         +------------------------+      +------------------------+
         | Prometheus / Grafana   |      | `tensorpack-tune` CLI  |
         | (:9100/metrics)        |      | (Manual Clock Locking) |
         +------------------------+      +------------------------+
                      |                               |
                      +---------------+---------------+
                                      |
                                      v
                    +------------------------------------+
                    | Zero-Overhead C++17 Fast NVML Core |
                    +------------------------------------+
                                      |
                                      v
                    +------------------------------------+
                    | NVIDIA GPU Silicon (B200/H100/A100)|
                    +------------------------------------+
```

---

## 🚀 Quickstart

### 1. Installation

Install via pip directly from the repository:

```bash
git clone https://github.com/tensorpackai/tensorpackai-community.git
cd tensorpackai-community
pip install .
```

*Requirements:* Linux x86_64 or aarch64, NVIDIA GPU Driver 525+, Python 3.8+.

### 2. Live GPU Telemetry & Inspection

Inspect real-time GPU power draw, temperatures, operating clocks, and VRAM occupancy:

```bash
tensorpack-tune status
```

**Output Example:**
```text
TensorPackAI GPU Telemetry (Driver: 550.54.14):
╒═══════╤══════════════════════════════════╤════════╤═════════╤════════╤════════════╤═════════════╤════════════════╤════════╕
│   GPU │ Device Name                      │ Tier   │ Power   │ Temp   │ SM Clock   │ Mem Clock   │ VRAM Used/Total│ SM Util│
╞═══════╪══════════════════════════════════╪════════╪═════════╪════════╪════════════╪═════════════╪════════════════╪════════╡
│     0 │ NVIDIA H100 80GB HBM3            │ H100   │ 285.0 W │ 52 °C  │ 1095 MHz   │ 2619 MHz    │ 40.0 / 80.0 GB │ 75%    │
│     1 │ NVIDIA H100 80GB HBM3            │ H100   │ 290.0 W │ 53 °C  │ 1095 MHz   │ 2619 MHz    │ 40.0 / 80.0 GB │ 75%    │
╘═══════╧══════════════════════════════════╧════════╧═════════╧════════╧════════════╧═════════════╧════════════════╧════════╛
```

### 3. Recommend Architecture Profiles

Inspect recommended phase-aware operating clocks for your detected accelerator:

```bash
tensorpack-tune profile --gpu 0
```

```text
=======================================================
 TensorPackAI Profile for GPU 0: NVIDIA H100 80GB HBM3 (H100)
=======================================================
 • Architecture:                 Hopper
 • TDP / Baseline Idle:          700 W / 150 W
 • Prefill Boost Clock (f_max):  1980 MHz  (Peak GEMM Compute Bound)
 • Decode Throttle Clock (f_mem):1095 MHz  (Memory-Paced GEMV Bound)
 • Idle Power-Gated Clock (f_min):540 MHz  (Deep Energy Gating)
 • Estimated Decode Dynamic Drop:~72.9% power reduction (P_dyn ∝ V^2 * f)
=======================================================
```

### 4. Lock Operating Clocks (CLI Tuning)

Lock GPU core clock to memory-paced decode frequency (e.g., 1095 MHz on H100) to measure power drops:

```bash
# Pin GPU 0 core frequency (Requires CAP_SYS_ADMIN or sudo)
sudo tensorpack-tune pin --gpu 0 --clock 1095

# Reset back to default vendor curve
sudo tensorpack-tune reset --gpu all
```

### 5. Launch Prometheus Telemetry Exporter

Start the Prometheus exporter service to scrape real-time hardware energy and vLLM token metrics into Grafana:

```bash
tensorpack-exporter --port 9100 --interval 0.5 --vllm-url http://localhost:8000/metrics
```

The metrics endpoint will be available at `http://localhost:9100/metrics`.

---

## 📊 Prometheus Metrics Reference

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `tensorpack_gpu_energy_joules_total` | Counter | Cumulative electrical energy consumed by the GPU since exporter start ($\text{Joules}$). |
| `tensorpack_generation_tokens_total` | Counter | Cumulative generated tokens observed via vLLM `/metrics` or runtime hooks. |
| `tensorpack_joules_per_token` | Gauge | Windowed electrical energy consumed per generated token ($\Delta E / \Delta N_{\text{tok}}$). |
| `tensorpack_cumulative_joules_per_token` | Gauge | Cumulative electrical energy per generated token since start ($E_{\text{total}} / N_{\text{tok,total}}$). |
| `tensorpack_tokens_per_watt` | Gauge | Windowed throughput efficiency ($\text{Tokens/s} / \text{Watt}$, equivalent to $\text{Tokens/Joule}$). |
| `tensorpack_gpu_power_watts` | Gauge | Instantaneous GPU electrical power dissipation in Watts. |
| `tensorpack_gpu_sm_clock_mhz` | Gauge | Streaming Multiprocessor (SM) operating frequency in MHz. |
| `tensorpack_gpu_mem_clock_mhz` | Gauge | GPU Memory operating frequency in MHz. |
| `tensorpack_gpu_temperature_celsius` | Gauge | GPU silicon core temperature in Celsius. |
| `tensorpack_gpu_utilization_ratio` | Gauge | GPU compute utilization (0.0 to 1.0). |

---

## 🏢 TensorPackAI Enterprise

While the Community edition provides essential hardware telemetry, Prometheus energy integration, and manual frequency locking, **TensorPackAI Enterprise** delivers fully autonomous, micro-batch phase-aware runtime co-design:

* **Autonomous Phase-Aware Silicon Governor:**
  - Distinguishes **prefill** from **decode** phases at micro-batch execution boundaries without modifying model weights or application code.
  - Amortizes driver invocation latencies ($\Delta_{\text{driver}}^{\text{amortized}} < 0.05\text{ ms/token}$) with zero pipeline blocking.
  - Eliminates frequency hunting and protects server VRMs from thermo-mechanical stress during continuous batching.
* **Cluster-Level Multi-Objective Batch Orchestrator:**
  - Optimizes request-to-GPU placement across heterogeneous accelerator pools in background execution threads with **Zero Dispatch Latency**.
  - **Guarantees P90 Inter-Token Latency (ITL) < 20 ms** while reducing net cluster energy by **22.3% – 58.1%** (achieving up to **2.73 Tokens/Joule**, a **2.39× efficiency gain**).

To request an Enterprise trial or benchmark evaluation for your datacenter:
👉 Visit [tensorpackai.com](https://tensorpackai.com) or contact `oguzhan.sereflisan@gmail.com`.

---

## 🔍 Search & Discovery Keywords

`vllm energy optimization` • `vllm power management` • `vllm dvfs` • `triton inference server energy` • `llm inference power consumption` • `joules per token` • `tokens per watt` • `gpu energy telemetry` • `prometheus nvml exporter` • `nvidia h100 power capping` • `nvidia a100 clock locking` • `nvidia b200 power efficiency` • `sustainable ai inference` • `green computing llm` • `cloud gpu cost reduction` • `prefill decode phase separation` • `datacenter carbon footprint reduction`

---

## 📄 License

TensorPackAI Community Edition is licensed under the Apache License, Version 2.0 (`SPDX-License-Identifier: Apache-2.0`).  
Enterprise runtime governors and cluster orchestration engines are proprietary and protected by pending patent applications.
