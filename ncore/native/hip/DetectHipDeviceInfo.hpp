/**
 * @file DetectHipDeviceInfo.hpp
 * @brief HIP device property queries and formatted output.
 *
 * @details
 * Declares the @ref hipDetectedDeviceProps_t struct that carries
 * the properties of a detected HIP device, and the two public
 * functions that retrieve and display those properties.
 *
 * The properties are cached after the first query; subsequent calls
 * return the cached values without additional HIP runtime API
 * calls.
 *
 * @see DetectHipDeviceInfo.cpp  Implementation of property queries.
 * @see DetectHipDevice.hpp      Device availability detection.
 * @see device.c                 Core device layer that calls these.
 */

#pragma once

#include <ncore/core/status.h>
#include <string>

/**
 * @struct hipDetectedDeviceProps_t
 * @brief Properties of a detected HIP device.
 *
 * @details
 * Populated by @ref getHipDeviceProperties on the first call and
 * cached thereafter.  The @ref isAvailable member indicates whether
 * the struct contains valid data; all other fields are undefined
 * when @ref isAvailable is `false`.
 *
 * Provides an `explicit operator bool()` for convenient
 * availability checking:
 *
 * @code{.cpp}
 * hipDetectedDeviceProps_t props = getHipDeviceProperties(&status);
 * if (props) {
 *     std::cout << props.name << "\n";
 * }
 * @endcode
 */
struct hipDetectedDeviceProps_t {
  bool isAvailable;                ///< `true` if the device was detected.
  std::string name;                ///< Device name (e.g., "AMD Radeon RX 7900 XTX").
  std::string runtimeVersion;      ///< HIP runtime version (e.g., "6.4.54321").
  std::string driverVersion;       ///< HIP driver version (e.g., "6.4.54321").
  std::string totalGlobalMem;      ///< Total device memory (formatted, e.g., "24.0 GiB").
  std::string gcnArchName;         ///< GCN architecture name (e.g., "gfx1100").
  int multiProcessorCount;         ///< Number of compute units (CUs).
  int warpSize;                    ///< Wavefront size in threads.
  int maxThreadsPerBlock;          ///< Maximum threads per block.
  int maxThreadsPerMultiProcessor; ///< Maximum threads per CU.
  explicit operator bool() const noexcept { return isAvailable; }
};

/**
 * @brief Retrieve the properties of the detected HIP device.
 *
 * @details
 * Returns a cached @ref hipDetectedDeviceProps_t populated on the
 * first call.  Subsequent calls return the cached value without
 * additional HIP runtime API calls.
 *
 * @param[out] status  Receives `novaSuccess` on success, or an
 *                     error code with a descriptive message on
 *                     failure.
 *
 * @return Cached device properties.  Check @ref isAvailable to
 *         determine whether the data is valid.
 *
 * @note Thread-safe.  The result is cached in a `static` local
 *       variable initialised exactly once (C++11 guarantee).
 *
 * @see printHipDeviceInfo()  Prints the properties to stdout.
 * @see hipDetectedDeviceProps_t  The returned struct type.
 */
hipDetectedDeviceProps_t getHipDeviceProperties(novaStatus_t *status) noexcept;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print HIP device properties to stdout.
 *
 * @details
 * Queries the HIP runtime for device 0 properties and prints
 * them using ANSI colour codes.  When @p verbose is `false`, a
 * concise two-line summary is printed.  When @p verbose is `true`,
 * a detailed multi-line block is printed.
 *
 * @param[in] verbose  If `true`, print the full property block.
 *                     If `false`, print a concise summary.
 *
 * @return @ref novaStatus_t with the result of the detection.
 *         On success, set to @ref novaSuccess.  On failure, set to
 *         the appropriate error code.
 *
 * @note Does not require a prior call to @ref isHipDeviceAvailable.
 *       Queries the HIP runtime directly.
 *
 * @see getHipDeviceProperties()  Returns the raw property struct.
 * @see print_device_info()  Core device layer wrapper.
 */
novaStatus_t printHipDeviceInfo(bool verbose);

#ifdef __cplusplus
}
#endif
