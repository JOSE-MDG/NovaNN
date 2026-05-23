/**
 * @file cuda_device.c
 * @brief CUDA runtime device detection.
 *
 * Probes the CUDA Runtime API for available devices and stores the active
 * device id selected by the core device layer.
 */

#include <stdbool.h>
#include <stdio.h>
#include <threads.h>

/** @brief Active CUDA device id, or -1 when no CUDA device is active. */
int active_cuda_device_id = -1;

/** @brief Cached CUDA device availability flag. */
bool cuda_device_available = false;

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>

static once_flag device_flags_once = ONCE_FLAG_INIT;
static mtx_t device_flags_mtx;

/**
 * @brief Initialise the mutex used to protect CUDA device flags.
 */
static void init_device_flags_lock(void) {
  (void)mtx_init(&device_flags_mtx, mtx_plain);
}

/**
 * @brief Probe whether at least one CUDA device is available.
 *
 * @param log If true, print CUDA runtime errors to stdout.
 * @return true when cudaGetDeviceCount reports one or more devices.
 */
bool is_cuda_device_available(bool log) {
  call_once(&device_flags_once, init_device_flags_lock);

  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);

  if (err != cudaSuccess) {
    if (log) {
      printf("%s\n", cudaGetErrorString(err));
    }
    return false;
  }

  if (count == 0) {
    return false;
  }

  mtx_lock(&device_flags_mtx);
  active_cuda_device_id = 0;
  cuda_device_available = true;
  mtx_unlock(&device_flags_mtx);
  return cuda_device_available;
}

/**
 * @brief Probe whether CUDA devices are unavailable.
 *
 * @param log If true, print CUDA runtime errors to stdout.
 * @return true when no CUDA device is available.
 */
bool is_cuda_device_not_available(bool log) {
  return (bool)(!is_cuda_device_available(log));
}

/**
 * @brief Return the selected CUDA device id.
 *
 * @return Active CUDA device id, or -1 when no CUDA device is active.
 */
int get_cuda_device_id(void) { return active_cuda_device_id; }
#else

/**
 * @brief Fallback CUDA availability probe when CUDA headers are unavailable.
 */
bool is_cuda_device_available(bool log) { return false; }

/**
 * @brief Fallback CUDA unavailability probe when CUDA headers are unavailable.
 */
bool is_cuda_device_not_available(bool log) { return true; }

/**
 * @brief Fallback CUDA device id when CUDA headers are unavailable.
 */
int get_cuda_device_id(void) { return -1; }
#endif
