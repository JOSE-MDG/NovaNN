/**
 * @file admin.cpp
 * @brief Runtime selection of the available GPU backend.
 *
 * Implements the backend probe used by the C-FFI layer to decide whether
 * CUDA, HIP, or no GPU runtime should handle device operations.
 */

#include "admin.hpp"

/**
 * @brief Return the first available device backend.
 *
 * CUDA is preferred when both CUDA and HIP report availability. If neither
 * runtime is available, DeviceNull is returned so callers can fail gracefully.
 *
 * @return DeviceCUDA, DeviceHIP, or DeviceNull according to runtime detection.
 */
DeviceKind_t get_device_backend(void) {
  if (is_cuda_available()) {
    return DeviceKind_t::DeviceCUDA;
  }
  if (is_hip_available()) {
    return DeviceKind_t::DeviceHIP;
  }
  return DeviceKind_t::DeviceNull;
}
