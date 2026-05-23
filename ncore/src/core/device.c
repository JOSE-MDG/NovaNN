/**
 * @file device.c
 * @brief Runtime device backend detection helpers.
 *
 * Implements the public device availability checks declared in device.h.
 * CUDA and HIP probes are delegated to backend-specific detection units,
 * while this file decides which backend is currently usable by the core C
 * layer.
 */

#include <ncore/device.h>

/**
 * @brief Check whether any GPU device backend is available.
 *
 * @param kind Requested backend kind.
 * @param verbose If true, backend probes may print runtime diagnostics.
 * @return true when the requested backend reports an available device.
 */
bool is_device_available(DeviceKind kind, bool verbose) {
  switch (kind) {
  case CUDA_DEVICE:
    return is_cuda_device_available(verbose);
  case HIP_DEVICE:
    return is_hip_device_available(verbose);
  case NULL_DEVICE:
  default:
    return false;
  }
}

/**
 * @brief Check whether CUDA should be selected as the active backend.
 *
 * @return true when CUDA reports an available device.
 */
bool is_cuda_available(void) {
  return is_cuda_device_available(false);
}

/**
 * @brief Check whether HIP should be selected as the active backend.
 *
 * @return true when HIP reports an available device.
 */
bool is_hip_available(void) {
  return is_hip_device_available(false);
}
