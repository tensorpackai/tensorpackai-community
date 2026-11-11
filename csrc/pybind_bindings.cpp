#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nvml_fast_wrapper.hpp"

namespace py = pybind11;
using namespace tensorpack;

PYBIND11_MODULE(_tensorpack_csrc, m) {
    m.doc() = "TensorPackAI High-Performance Silicon & NVML Telemetry Engine (C++)";

    py::class_<GpuTelemetry>(m, "GpuTelemetry")
        .def(py::init<>())
        .def_readonly("device_index", &GpuTelemetry::device_index)
        .def_readonly("device_name", &GpuTelemetry::device_name)
        .def_readonly("power_mw", &GpuTelemetry::power_mw)
        .def_readonly("temperature_c", &GpuTelemetry::temperature_c)
        .def_readonly("sm_clock_mhz", &GpuTelemetry::sm_clock_mhz)
        .def_readonly("mem_clock_mhz", &GpuTelemetry::mem_clock_mhz)
        .def_readonly("memory_used_bytes", &GpuTelemetry::memory_used_bytes)
        .def_readonly("memory_total_bytes", &GpuTelemetry::memory_total_bytes)
        .def_readonly("gpu_utilization_pct", &GpuTelemetry::gpu_utilization_pct)
        .def_readonly("mem_utilization_pct", &GpuTelemetry::mem_utilization_pct)
        .def("__repr__", [](const GpuTelemetry &t) {
            return "<GpuTelemetry GPU=" + std::to_string(t.device_index) + 
                   " Power=" + std::to_string(t.power_mw / 1000.0) + "W" +
                   " Temp=" + std::to_string(t.temperature_c) + "C" +
                   " SM_Clock=" + std::to_string(t.sm_clock_mhz) + "MHz" +
                   " Util=" + std::to_string(t.gpu_utilization_pct) + "%>";
        });

    py::class_<DeviceClockRange>(m, "DeviceClockRange")
        .def_readonly("min_clock_mhz", &DeviceClockRange::min_clock_mhz)
        .def_readonly("max_clock_mhz", &DeviceClockRange::max_clock_mhz);

    py::class_<FastNVML>(m, "FastNVML")
        .def(py::init<>())
        .def("is_available", &FastNVML::isAvailable)
        .def("is_mock_mode", &FastNVML::isMockMode)
        .def("set_mock_mode", &FastNVML::setMockMode, py::arg("enable"))
        .def("get_device_count", &FastNVML::getDeviceCount)
        .def("get_driver_version", &FastNVML::getDriverVersion)
        .def("get_telemetry", &FastNVML::getTelemetry, py::arg("device_index"))
        .def("get_all_telemetry", &FastNVML::getAllTelemetry)
        .def("get_clock_range", &FastNVML::getClockRange, py::arg("device_index"))
        .def("set_locked_clocks", &FastNVML::setLockedClocks, py::arg("device_index"), py::arg("min_mhz"), py::arg("max_mhz"))
        .def("reset_locked_clocks", &FastNVML::resetLockedClocks, py::arg("device_index"));
}
