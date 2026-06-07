/**
 * @file admin.cpp
 * @brief Runtime implementation of GPU backend detection.
 *
 * @details
 * Calls into the ncore C API to probe CUDA and HIP runtime
 * availability.  The result is used by @ref device_reserve,
 * @ref device_release, @ref device_resize, and @ref device_memcpy
 * in `ffi.cpp` to dispatch operations to the correct backend.
 *
 * @see admin.hpp      DeviceKind_t enum and function declaration.
 * @see ffi.cpp        FFI dispatch layer that consumes this probe.
 * @see ncore/device.h Provides is_cuda_available() and is_hip_available().
 */

#include "admin.hpp"
#include <ncore/device.h>

DeviceKind_t get_device_backend(void) {

  /**
   * @brief Probe CUDA and HIP runtime availability in priority order.
   *
   * @details
   * Checks CUDA first, then HIP.  The first detected runtime
   * wins.  This function is called by @ref device_memcpy and
   * @ref device_memcpy_c to auto-detect the active backend for
   * data transfers.
   *
   * @return @ref DeviceKind_t::DeviceCUDA if CUDA is available,
   *         @ref DeviceKind_t::DeviceHIP if HIP is available, or
   *         @ref DeviceKind_t::DeviceNull if neither is found.
   *
   * @see is_cuda_available()  CUDA runtime probe.
   * @see is_hip_available()   HIP runtime probe.
   */
  if (is_cuda_available()) {
    return DeviceKind_t::DeviceCUDA;
  }
  if (is_hip_available()) {
    return DeviceKind_t::DeviceHIP;
  }
  return DeviceKind_t::DeviceNull;
}
