/**
 * @file DetectHipDevice.hpp
 * @brief HIP device availability detection and identification.
 *
 * @details
 * Declares the two public C functions that the core device layer
 * calls to probe the HIP runtime for an available GPU and to
 * retrieve the active device index.  These functions are called
 * from @ref device.c via `extern "C"` linkage.
 *
 * The detection is performed exactly once (call-once semantics)
 * and cached in module-level atomics.  Subsequent queries return
 * the cached result without touching the HIP runtime API.
 *
 * @see DetectHipDevice.cpp  Implementation of the detection logic.
 * @see device.c             Core device layer that calls these.
 * @see DetectHipDeviceInfo.hpp  Device property queries.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Probe the HIP runtime for an available GPU device.
 *
 * @details
 * Queries `hipGetDeviceCount` to determine whether a usable GPU
 * is present.  The first call performs the actual runtime probe
 * and caches the result in module-level atomics; subsequent calls
 * return the cached value.
 *
 * When @p log is `true` and a device is found, delegates to
 * @ref printHipDeviceInfo to display device properties.
 *
 * @param[in] log      If `true`, print detection results to stderr
 *                     on failure, or call @ref printHipDeviceInfo
 *                     on success.
 * @param[in] verbose  If `true`, pass verbose flag to
 *                     @ref printHipDeviceInfo for detailed output.
 *
 * @return `true` when a HIP-capable device is found and
 *         successfully probed.  `false` if no device is available
 *         or the HIP runtime reports an error.
 *
 * @note Thread-safe.  Uses `std::call_once` to guarantee the
 *       probe runs exactly once.  The cached result is stored in
 *       `std::atomic` variables with acquire/release ordering.
 *
 * @see getHipDeviceId()  Returns the device index after detection.
 * @see printHipDeviceInfo()  Prints device properties.
 */
bool isHipDeviceAvailable(bool log, bool verbose);

/**
 * @brief Return the 0-based device index of the detected HIP GPU.
 *
 * @details
 * Returns the value stored by the most recent call to
 * @ref isHipDeviceAvailable.  The device index is `0` when a GPU
 * is found, or `-1` if detection has not yet been performed or no
 * device was found.
 *
 * @return 0-based HIP device index, or `-1` if unavailable.
 *
 * @note The return value is only meaningful after at least one call
 *       to @ref isHipDeviceAvailable has completed.
 *
 * @see isHipDeviceAvailable()
 */
int getHipDeviceId(void);

#ifdef __cplusplus
}
#endif
