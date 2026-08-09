/**
 * @file admin.cpp
 * @brief Runtime implementation of GPU backend detection.
 *
 * @details
 * Calls into the ncore C API to probe CUDA and HIP runtime
 * availability.  The result is used by @ref deviceReserve,
 * @ref deviceRelease, @ref deviceResize, and @ref deviceTransfer
 * in @c ffi.cpp to dispatch operations to the correct backend.
 *
 * @see admin.hpp      deviceKind_t enum and function declaration.
 * @see ffi.cpp        FFI dispatch layer that consumes this probe.
 * @see device.h       Provides is_cuda_available() and is_hip_available().
 */

#include "admin.hpp"
#include <ncore/core/device.h>

/**
 * @brief Probe CUDA and HIP runtime availability in priority order.
 *
 * @details
 * Checks CUDA first, then HIP.  The first detected runtime
 * wins.  This function is called by @ref deviceTransfer,
 *  and FFI Rust wrappers (in @c ncore/memory/src)
 * to auto-detect the active backend for data transfers.
 *
 * @return @ref deviceKind_t::DeviceCUDA if CUDA is available,
 *         @ref deviceKind_t::DeviceHIP if HIP is available, or
 *         @ref deviceKind_t::DeviceNull if neither is found.
 *
 * @see is_cuda_available()  CUDA runtime probe.
 * @see is_hip_available()   HIP runtime probe.
 */
deviceKind_t getDeviceBackend() {

  if (is_cuda_available()) {
    return deviceKind_t::DeviceCUDA;
  }
  if (is_hip_available()) {
    return deviceKind_t::DeviceHIP;
  }
  return deviceKind_t::DeviceNull;
}
