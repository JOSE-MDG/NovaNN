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
 * @enum DeviceKind_t
 * @brief Identifies which GPU runtime (if any) is available on the
 *        current system.
 *
 * @details
 * Used throughout the csrc module to route memory allocations and
 * data transfers to the correct backend.  The value is determined
 * at runtime by @ref get_device_backend().
 *
 * @see get_device_backend()  Returns the active backend.
 */
enum class DeviceKind_t : std::int8_t {
  DeviceCUDA = 0, ///< NVIDIA CUDA runtime detected.
  DeviceHIP = 1,  ///< AMD ROCm HIP runtime detected.
  DeviceNull = 2  ///< No supported GPU runtime detected.
};

extern "C" {

/**
 * @brief Query the system to determine which GPU runtime is
 *        available.
 *
 * @details
 * Probes CUDA and HIP runtime availability in order.  The first
 * runtime found wins:
 * 1. CUDA — returns @ref DeviceKind_t::DeviceCUDA.
 * 2. HIP  — returns @ref DeviceKind_t::DeviceHIP.
 * 3. Neither — returns @ref DeviceKind_t::DeviceNull.
 *
 * @return The detected GPU backend.
 *
 * @see is_cuda_available()  CUDA runtime probe.
 * @see is_hip_available()   HIP runtime probe.
 */
DeviceKind_t get_device_backend(void);
}
