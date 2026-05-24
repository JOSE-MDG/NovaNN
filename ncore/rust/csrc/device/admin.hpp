/**
 * @file admin.hpp
 * @brief Runtime device-backend detection and enumeration.
 *
 * Defines the DeviceKind_t enum shared by all backend-specific allocator
 * modules and exposes a C-FFI function to query the active GPU runtime
 * (CUDA vs. HIP) at run time.
 */

#pragma once

#include <cstdint>

/**
 * @brief Identifies the active GPU compute backend.
 *
 * @var DeviceCUDA NVIDIA CUDA runtime.
 * @var DeviceHIP  AMD ROCm HIP runtime.
 * @var DeviceNull No supported GPU runtime detected.
 */
enum class DeviceKind_t : std::int8_t {
  DeviceCUDA = 0,
  DeviceHIP = 1,
  DeviceNull = 3
};

extern "C" {

/**
 * @brief Query the active device backend.
 *
 * @return DeviceKind_t value indicating the runtime that was detected
 *         during library initialisation.
 */
DeviceKind_t get_device_backend(void);
}
