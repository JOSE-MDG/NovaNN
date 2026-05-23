/**
 * @file hip_device.c
 * @brief HIP runtime device detection.
 *
 * Probes the HIP Runtime API for available devices and stores the active
 * device id selected by the core device layer.
 */

#include <stdbool.h>
#include <stdio.h>
#include <threads.h>

/** @brief Active HIP device id, or -1 when no HIP device is active. */
int active_device_id = -1;

/** @brief Cached HIP device availability flag. */
bool device_available = false;

#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

static once_flag device_flags_once = ONCE_FLAG_INIT;
static mtx_t device_flags_mtx;

/**
 * @brief Initialise the mutex used to protect HIP device flags.
 */
static void init_device_flags_lock(void) {
  (void)mtx_init(&device_flags_mtx, mtx_plain);
}

/**
 * @brief Probe whether at least one HIP device is available.
 *
 * @param log If true, print HIP runtime errors to stdout.
 * @return true when hipGetDeviceCount reports one or more devices.
 */
bool is_hip_device_available(bool log) {
  call_once(&device_flags_once, init_device_flags_lock);

  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);

  if (err != hipSuccess) {
    if (log) {
      printf("%s\n", hipGetErrorString(err));
    }
    return false;
  }

  if (count == 0) {
    return false;
  }

  mtx_lock(&device_flags_mtx);
  active_device_id = 0;
  device_available = true;
  mtx_unlock(&device_flags_mtx);
  return device_available;
}

/**
 * @brief Probe whether HIP devices are unavailable.
 *
 * @param log If true, print HIP runtime errors to stdout.
 * @return true when no HIP device is available.
 */
bool is_hip_device_not_available(bool log) {
  return (bool)(!is_hip_device_available(log));
}

/**
 * @brief Return the selected HIP device id.
 *
 * @return Active HIP device id, or -1 when no HIP device is active.
 */
int get_hip_device_id(void) { return active_device_id; }
#else

/**
 * @brief Fallback HIP availability probe when HIP headers are unavailable.
 */
bool is_hip_device_available(bool log) { return false; }

/**
 * @brief Fallback HIP unavailability probe when HIP headers are unavailable.
 */
bool is_hip_device_not_available(bool log) { return true; }

/**
 * @brief Fallback HIP device id when HIP headers are unavailable.
 */
int get_hip_device_id(void) { return -1; }
#endif
