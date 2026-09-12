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

#include <array>
#include <cstddef>
#include <cstdint>
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
 * when @ref isAvailable is @c false.
 *
 * Provides an @c explicit operator bool() for convenient
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
  bool isAvailable;           ///< @c true if the device was detected.
  std::string name;           ///< Device name (e.g., "AMD Radeon RX 7900 XTX").
  std::string runtimeVersion; ///< HIP runtime version (e.g., "6.4.54321").
  std::string driverVersion;  ///< HIP driver version (e.g., "6.4.54321").
  std::string
      totalGlobalMem; ///< Total device memory (formatted, e.g., "24.0 GiB").
  std::string gcnArchName;         ///< GCN architecture name (e.g., "gfx1100").
  int multiProcessorCount;         ///< Number of compute units (CUs).
  int warpSize;                    ///< Wavefront size in threads.
  int maxThreadsPerBlock;          ///< Maximum threads per block.
  int maxThreadsPerMultiProcessor; ///< Maximum threads per CU.
  int maxBlocksPerMultiProcessor;  ///< Maximum resident blocks per CU.
  int major;                       ///< Reported major version.
  int minor;                       ///< Reported minor version.
  int clockRate;                   ///< CU clock in kHz.
  int memoryClockRate;             ///< DRAM clock in kHz.
  int memoryBusWidth;              ///< DRAM bus width in bits.
  size_t sharedMemPerBlock;        ///< Shared memory per block, bytes.
  size_t sharedMemPerMultiprocessor; ///< Shared memory per CU, bytes.
  int regsPerMultiprocessor;       ///< 32-bit registers per CU.
  std::array<int, 3> maxGridSize;  ///< Max grid extent per axis.
  int l2CacheSize;                 ///< L2 capacity, bytes.
  int persistingL2CacheMaxSize;    ///< Max L2 persisting lines, bytes.
  uint64_t peakFp32Flops;          ///< FP32 peak, flop/s (0 when unknown).
  std::string memBandwidth;        ///< Theoretical bandwidth, formatted.
  explicit operator bool() const noexcept { return isAvailable; }
};

/**
 * @struct hipDetectedDeviceAttrs_t
 * @brief Auxiliary device attributes working around @c hipGetDeviceProperties gaps.
 *
 * @details
 * Carries fields the property query documents as zero-returning (see the
 * @c @bug notes on @c hipGetDeviceProperties), queried via
 * @c hipDeviceGetAttribute instead. Populated by
 * @ref getHipDeviceAttributes on the first call and cached thereafter.
 * Each field degrades to 0 (unknown) on query failure without failing
 * detection: attributes are auxiliary by contract. Clock figures are
 * intentionally absent here: @c hipDeviceProp_t still reports them.
 */
struct hipDetectedDeviceAttrs_t {
  int maxThreadsPerMultiProcessor; ///< Resident threads per CU (0 when unknown).
};

/**
 * @brief Retrieve auxiliary device attributes.
 *
 * @details
 * Returns a cached @ref hipDetectedDeviceAttrs_t populated on the
 * first call. Subsequent calls return the cached value without
 * additional runtime API calls. The status is always success:
 * failed queries degrade individual fields to 0.
 * Query after detection: the device id is captured on the first call,
 * so a later device switch keeps returning the first device attrs
 * until process restart.
 *
 * @param[out] status  Receives @c novaSuccess.
 *
 * @return Cached device attributes (zeros where unknown).
 *
 * @note Thread-safe.  The result is cached in a @c static local
 *       variable initialised exactly once (C++11 guarantee).
 */
hipDetectedDeviceAttrs_t
getHipDeviceAttributes(novaStatus_t *status) noexcept;

/**
 * @brief Retrieve the properties of the detected HIP device.
 *
 * @details
 * Returns a cached @ref hipDetectedDeviceProps_t populated on the
 * first call.  Subsequent calls return the cached value without
 * additional HIP runtime API calls.
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
