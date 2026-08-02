/**
 * @file DetectCudaDeviceInfo.hpp
 * @brief CUDA device property queries and formatted output.
 *
 * @details
 * Declares the @ref cudaDetectedDeviceProps_t struct that carries
 * the properties of a detected CUDA device, and the two public
 * functions that retrieve and display those properties.
 *
 * The properties are cached after the first query; subsequent calls
 * return the cached values without additional CUDA runtime API
 * calls.
 *
 * @see DetectCudaDeviceInfo.cpp  Implementation of property queries.
 * @see DetectCudaDevice.hpp      Device availability detection.
 * @see device.c                  Core device layer that calls these.
 */

#pragma once

#include <ncore/core/status.h>
#include <string>

/**
 * @struct cudaDetectedDeviceProps_t
 * @brief Properties of a detected CUDA device.
 *
 * @details
 * Populated by @ref getCudaDeviceProperties on the first call and
 * cached thereafter.  The @ref isAvailable member indicates whether
 * the struct contains valid data; all other fields are undefined
 * when @ref isAvailable is @c false.
 *
 * Provides an @c explicit operator bool() for convenient
 * availability checking:
 *
 * @code{.cpp}
 * cudaDetectedDeviceProps_t props = getCudaDeviceProperties(&status);
 * if (props) {
 *     std::cout << props.name << "\n";
 * }
 * @endcode
 */
struct cudaDetectedDeviceProps_t {
  bool isAvailable;                ///< @c true if the device was detected.
  std::string name;                ///< Device name (e.g., "NVIDIA GeForce RTX 5070").
  std::string runtimeVersion;      ///< CUDA runtime version (e.g., "12.8").
  std::string driverVersion;       ///< CUDA driver version (e.g., "12.8").
  std::string totalGlobalMem;      ///< Total device memory (formatted, e.g., "12.0 GiB").
  std::string comCapability;       ///< Compute capability (e.g., "12.0").
  int multiProcessorCount;         ///< Number of streaming multiprocessors (SMs).
  int warpSize;                    ///< Warp size in threads.
  int maxThreadsPerBlock;          ///< Maximum threads per block.
  int maxThreadsPerMultiProcessor; ///< Maximum threads per SM.
  explicit operator bool() const noexcept { return isAvailable; }
};

/**
 * @brief Retrieve the properties of the detected CUDA device.
 *
 * @details
 * Returns a cached @ref cudaDetectedDeviceProps_t populated on the
 * first call.  Subsequent calls return the cached value without
 * additional CUDA runtime API calls.
 *
 * @param[out] status  Receives @c novaSuccess on success, or an
 *                     error code with a descriptive message on
 *                     failure.
 *
 * @return Cached device properties.  Check @ref isAvailable to
 *         determine whether the data is valid.
 *
 * @note Thread-safe.  The result is cached in a @c static local
 *       variable initialised exactly once (C++11 guarantee).
 *
 * @see printCudaDeviceInfo()  Prints the properties to stdout.
 * @see cudaDetectedDeviceProps_t  The returned struct type.
 */
cudaDetectedDeviceProps_t
getCudaDeviceProperties(novaStatus_t *status) noexcept;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print CUDA device properties to stdout.
 *
 * @details
 * Queries the CUDA runtime for device 0 properties and prints
 * them using ANSI colour codes.  When @p verbose is @c false, a
 * concise two-line summary is printed.  When @p verbose is @c true,
 * a detailed multi-line block is printed.
 *
 * @param[in] verbose  If @c true, print the full property block.
 *                     If @c false, print a concise summary.
 *
 * @return @ref novaStatus_t with the result of the detection.
 *         On success, set to @ref novaSuccess.  On failure, set to
 *         the appropriate error code.
 *
 * @note Does not require a prior call to @ref isCudaDeviceAvailable.
 *       Queries the CUDA runtime directly.
 *
 * @see getCudaDeviceProperties()  Returns the raw property struct.
 * @see print_device_info()  Core device layer wrapper.
 */
novaStatus_t printCudaDeviceInfo(bool verbose);

#ifdef __cplusplus
}
#endif
