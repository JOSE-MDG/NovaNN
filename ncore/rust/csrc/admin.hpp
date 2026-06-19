/**
 * @file admin.hpp
 * @brief GPU backend identification for the device-agnostic FFI layer.
 *
 * @details
 * Declares the @ref deviceKind_t enumeration and the runtime probe
 * function @ref getDeviceBackend().  The rest of the csrc module
 * uses these to dispatch memory and copy operations to the correct
 * GPU runtime (CUDA or HIP).
 *
 * The detected backend is determined once at process start by
 * probing CUDA and HIP runtime availability via
 * @ref is_cuda_available() and @ref is_hip_available() from
 * `<ncore/device.h>`.
 *
 * @see admin.cpp       Implementation of the runtime probe.
 * @see ffi.hpp         Top-level FFI dispatch that consumes this enum.
 * @see ncore/device.h  Backend availability query functions.
 */

#pragma once

#include <cstdint>

/**
 * @enum deviceKind_t
 * @brief Identifies which GPU runtime (if any) is available on the
 *        current system.
 *
 * @details
 * Used throughout the csrc module to route memory allocations and
 * data transfers to the correct backend.  The value is determined
 * at runtime by @ref getDeviceBackend().
 *
 * @see getDeviceBackend()  Returns the active backend.
 */
enum class deviceKind_t : std::int8_t {
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
 * 1. CUDA — returns @ref deviceKind_t::DeviceCUDA.
 * 2. HIP  — returns @ref deviceKind_t::DeviceHIP.
 * 3. Neither — returns @ref deviceKind_t::DeviceNull.
 *
 * @return The detected GPU backend.
 *
 * @see is_cuda_available()  CUDA runtime probe.
 * @see is_hip_available()   HIP runtime probe.
 */
deviceKind_t getDeviceBackend();
}
