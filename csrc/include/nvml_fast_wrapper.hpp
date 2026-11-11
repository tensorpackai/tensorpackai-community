#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace tensorpack {

/**
 * Real-time GPU telemetry structure capturing power, thermal, clock, and memory utilization.
 */
struct GpuTelemetry {
    uint32_t device_index;
    std::string device_name;
    uint32_t power_mw;              // Instantaneous power dissipation (milliwatts)
    uint32_t temperature_c;         // Core temperature (Celsius)
    uint32_t sm_clock_mhz;          // Streaming Multiprocessor (SM) clock frequency (MHz)
    uint32_t mem_clock_mhz;         // Memory controller clock frequency (MHz)
    uint64_t memory_used_bytes;     // Allocated VRAM footprint (bytes)
    uint64_t memory_total_bytes;    // Total onboard physical VRAM (bytes)
    uint32_t gpu_utilization_pct;   // SM compute utilization percentage (0-100)
    uint32_t mem_utilization_pct;   // Memory bus utilization percentage (0-100)
};

/**
 * Valid operating clock bounds for the device.
 */
struct DeviceClockRange {
    uint32_t min_clock_mhz;
    uint32_t max_clock_mhz;
};

/**
 * Fast NVML dynamic loading wrapper for low-overhead runtime telemetry and clock control.
 */
class FastNVML {
public:
    FastNVML();
    ~FastNVML();

    // System Information
    bool isAvailable() const;
    uint32_t getDeviceCount() const;
    std::string getDriverVersion() const;

    // Telemetry Collection
    GpuTelemetry getTelemetry(uint32_t device_index);
    std::vector<GpuTelemetry> getAllTelemetry();

    // Frequency and Power Control
    DeviceClockRange getClockRange(uint32_t device_index);
    bool setLockedClocks(uint32_t device_index, uint32_t min_mhz, uint32_t max_mhz);
    bool resetLockedClocks(uint32_t device_index);

    // Mock Mode (For development, non-GPU instances, and CI/CD pipelines)
    void setMockMode(bool enable);
    bool isMockMode() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace tensorpack
