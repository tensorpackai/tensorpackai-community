#include "nvml_fast_wrapper.hpp"
#include <iostream>
#include <sstream>
#include <cstring>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace tensorpack {

// NVML C API function types and definitions (Isolated dynamic linking without header dependency)
typedef int nvmlReturn_t;
#define NVML_SUCCESS 0
#define NVML_TEMPERATURE_GPU 0
#define NVML_CLOCK_SM 1
#define NVML_CLOCK_MEM 2

struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};

struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

typedef void* nvmlDevice_t;

class FastNVML::Impl {
public:
    bool initialized = false;
    bool mock_mode = false;
    void* lib_handle = nullptr;

    // NVML Function Pointers
    nvmlReturn_t (*p_nvmlInit_v2)() = nullptr;
    nvmlReturn_t (*p_nvmlShutdown)() = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetCount_v2)(unsigned int*) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t*) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int*) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetClockInfo)(nvmlDevice_t, int, unsigned int*) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceSetGpuLockedClocks)(nvmlDevice_t, unsigned int, unsigned int) = nullptr;
    nvmlReturn_t (*p_nvmlDeviceResetGpuLockedClocks)(nvmlDevice_t) = nullptr;
    nvmlReturn_t (*p_nvmlSystemGetDriverVersion)(char*, unsigned int) = nullptr;

    Impl() {
#if defined(__linux__)
        lib_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!lib_handle) {
            lib_handle = dlopen("libnvidia-ml.so", RTLD_NOW);
        }

        if (lib_handle) {
            p_nvmlInit_v2 = (nvmlReturn_t(*)())dlsym(lib_handle, "nvmlInit_v2");
            p_nvmlShutdown = (nvmlReturn_t(*)())dlsym(lib_handle, "nvmlShutdown");
            p_nvmlDeviceGetCount_v2 = (nvmlReturn_t(*)(unsigned int*))dlsym(lib_handle, "nvmlDeviceGetCount_v2");
            p_nvmlDeviceGetHandleByIndex_v2 = (nvmlReturn_t(*)(unsigned int, nvmlDevice_t*))dlsym(lib_handle, "nvmlDeviceGetHandleByIndex_v2");
            p_nvmlDeviceGetName = (nvmlReturn_t(*)(nvmlDevice_t, char*, unsigned int))dlsym(lib_handle, "nvmlDeviceGetName");
            p_nvmlDeviceGetPowerUsage = (nvmlReturn_t(*)(nvmlDevice_t, unsigned int*))dlsym(lib_handle, "nvmlDeviceGetPowerUsage");
            p_nvmlDeviceGetTemperature = (nvmlReturn_t(*)(nvmlDevice_t, int, unsigned int*))dlsym(lib_handle, "nvmlDeviceGetTemperature");
            p_nvmlDeviceGetClockInfo = (nvmlReturn_t(*)(nvmlDevice_t, int, unsigned int*))dlsym(lib_handle, "nvmlDeviceGetClockInfo");
            p_nvmlDeviceGetMemoryInfo = (nvmlReturn_t(*)(nvmlDevice_t, nvmlMemory_t*))dlsym(lib_handle, "nvmlDeviceGetMemoryInfo");
            p_nvmlDeviceGetUtilizationRates = (nvmlReturn_t(*)(nvmlDevice_t, nvmlUtilization_t*))dlsym(lib_handle, "nvmlDeviceGetUtilizationRates");
            p_nvmlDeviceSetGpuLockedClocks = (nvmlReturn_t(*)(nvmlDevice_t, unsigned int, unsigned int))dlsym(lib_handle, "nvmlDeviceSetGpuLockedClocks");
            p_nvmlDeviceResetGpuLockedClocks = (nvmlReturn_t(*)(nvmlDevice_t))dlsym(lib_handle, "nvmlDeviceResetGpuLockedClocks");
            p_nvmlSystemGetDriverVersion = (nvmlReturn_t(*)(char*, unsigned int))dlsym(lib_handle, "nvmlSystemGetDriverVersion");

            if (p_nvmlInit_v2 && p_nvmlInit_v2() == NVML_SUCCESS) {
                initialized = true;
                return;
            }
        }
#endif
        // Fallback to mock mode if hardware or dynamic library is unavailable
        mock_mode = true;
        initialized = true;
    }

    ~Impl() {
#if defined(__linux__)
        if (initialized && !mock_mode && p_nvmlShutdown) {
            p_nvmlShutdown();
        }
        if (lib_handle) {
            dlclose(lib_handle);
        }
#endif
    }
};

FastNVML::FastNVML() : pImpl(std::make_unique<Impl>()) {}
FastNVML::~FastNVML() = default;

bool FastNVML::isAvailable() const {
    return pImpl->initialized;
}

bool FastNVML::isMockMode() const {
    return pImpl->mock_mode;
}

void FastNVML::setMockMode(bool enable) {
    pImpl->mock_mode = enable;
}

uint32_t FastNVML::getDeviceCount() const {
    if (pImpl->mock_mode) {
        return 4; // Mock cluster: 4x H100
    }
#if defined(__linux__)
    if (pImpl->p_nvmlDeviceGetCount_v2) {
        unsigned int count = 0;
        if (pImpl->p_nvmlDeviceGetCount_v2(&count) == NVML_SUCCESS) {
            return count;
        }
    }
#endif
    return 0;
}

std::string FastNVML::getDriverVersion() const {
    if (pImpl->mock_mode) {
        return "550.54.14 (Mock / Synthetic Engine)";
    }
#if defined(__linux__)
    if (pImpl->p_nvmlSystemGetDriverVersion) {
        char ver[80] = {0};
        if (pImpl->p_nvmlSystemGetDriverVersion(ver, sizeof(ver)) == NVML_SUCCESS) {
            return std::string(ver);
        }
    }
#endif
    return "Unavailable";
}

GpuTelemetry FastNVML::getTelemetry(uint32_t device_index) {
    GpuTelemetry t{};
    t.device_index = device_index;

    if (pImpl->mock_mode) {
        t.device_name = "NVIDIA H100 80GB HBM3 (Emulated)";
        t.power_mw = 285000 + (device_index * 5000); // ~285W
        t.temperature_c = 52 + (device_index % 5);
        t.sm_clock_mhz = 1095; // Decode memory-paced frequency
        t.mem_clock_mhz = 2619;
        t.memory_total_bytes = 85899345920ULL; // 80 GB
        t.memory_used_bytes = 42949672960ULL;  // 40 GB
        t.gpu_utilization_pct = 75;
        t.mem_utilization_pct = 88;
        return t;
    }

#if defined(__linux__)
    if (pImpl->p_nvmlDeviceGetHandleByIndex_v2) {
        nvmlDevice_t dev = nullptr;
        if (pImpl->p_nvmlDeviceGetHandleByIndex_v2(device_index, &dev) == NVML_SUCCESS) {
            char name[64] = {0};
            if (pImpl->p_nvmlDeviceGetName && pImpl->p_nvmlDeviceGetName(dev, name, sizeof(name)) == NVML_SUCCESS) {
                t.device_name = std::string(name);
            }
            if (pImpl->p_nvmlDeviceGetPowerUsage) {
                unsigned int p = 0;
                pImpl->p_nvmlDeviceGetPowerUsage(dev, &p);
                t.power_mw = p;
            }
            if (pImpl->p_nvmlDeviceGetTemperature) {
                unsigned int temp = 0;
                pImpl->p_nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);
                t.temperature_c = temp;
            }
            if (pImpl->p_nvmlDeviceGetClockInfo) {
                unsigned int sm_clk = 0, mem_clk = 0;
                pImpl->p_nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_clk);
                pImpl->p_nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &mem_clk);
                t.sm_clock_mhz = sm_clk;
                t.mem_clock_mhz = mem_clk;
            }
            if (pImpl->p_nvmlDeviceGetMemoryInfo) {
                nvmlMemory_t mem{};
                if (pImpl->p_nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
                    t.memory_used_bytes = mem.used;
                    t.memory_total_bytes = mem.total;
                }
            }
            if (pImpl->p_nvmlDeviceGetUtilizationRates) {
                nvmlUtilization_t util{};
                if (pImpl->p_nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
                    t.gpu_utilization_pct = util.gpu;
                    t.mem_utilization_pct = util.memory;
                }
            }
        }
    }
#endif
    return t;
}

std::vector<GpuTelemetry> FastNVML::getAllTelemetry() {
    uint32_t count = getDeviceCount();
    std::vector<GpuTelemetry> result;
    result.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        result.push_back(getTelemetry(i));
    }
    return result;
}

DeviceClockRange FastNVML::getClockRange(uint32_t device_index) {
    // Standard H100 operational clock range (MHz)
    return DeviceClockRange{540, 1980};
}

bool FastNVML::setLockedClocks(uint32_t device_index, uint32_t min_mhz, uint32_t max_mhz) {
    if (pImpl->mock_mode) {
        return true;
    }
#if defined(__linux__)
    if (pImpl->p_nvmlDeviceGetHandleByIndex_v2 && pImpl->p_nvmlDeviceSetGpuLockedClocks) {
        nvmlDevice_t dev = nullptr;
        if (pImpl->p_nvmlDeviceGetHandleByIndex_v2(device_index, &dev) == NVML_SUCCESS) {
            return (pImpl->p_nvmlDeviceSetGpuLockedClocks(dev, min_mhz, max_mhz) == NVML_SUCCESS);
        }
    }
#endif
    return false;
}

bool FastNVML::resetLockedClocks(uint32_t device_index) {
    if (pImpl->mock_mode) {
        return true;
    }
#if defined(__linux__)
    if (pImpl->p_nvmlDeviceGetHandleByIndex_v2 && pImpl->p_nvmlDeviceResetGpuLockedClocks) {
        nvmlDevice_t dev = nullptr;
        if (pImpl->p_nvmlDeviceGetHandleByIndex_v2(device_index, &dev) == NVML_SUCCESS) {
            return (pImpl->p_nvmlDeviceResetGpuLockedClocks(dev) == NVML_SUCCESS);
        }
    }
#endif
    return false;
}

} // namespace tensorpack
