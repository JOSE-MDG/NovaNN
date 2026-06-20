/**
 * @file DetectHipDeviceInfo.cpp
 * @brief HIP device property query and formatted output implementation.
 *
 * @details
 * Implements the property retrieval and printing API declared in
 * @ref DetectHipDeviceInfo.hpp.  Queries the HIP runtime for
 * device 0 properties (name, GCN architecture, memory, CUs, warp
 * size, thread limits, driver/runtime versions) and formats them
 * for human-readable output.
 *
 * The file is conditionally compiled behind `NOVA_HAS_HIP` and
 * `__has_include(<hip/hip_runtime_api.h>)`.  When HIP headers are
 * unavailable, stub functions returning error statuses are
 * provided.
 *
 * A `__HIP_PLATFORM_AMD__` macro is defined when neither AMD nor
 * NVIDIA platform macros are set, ensuring clangd and clang-tidy
 * can parse the HIP headers correctly.
 *
 * ## Caching
 *
 * Device properties are queried once and cached in a `static`
 * local variable within @ref initHipDeviceProperties.  Subsequent
 * calls to @ref getHipDeviceProperties and @ref
 * printHipDeviceInfo return the cached values without additional
 * HIP runtime API calls.
 *
 * ## Internal Helpers
 *
 * - @ref formatMemory — Converts byte counts to human-readable
 *   strings (GiB / MiB / bytes).
 * - @ref formatHipVersion — Converts the HIP integer version
 *   encoding to a "major.minor.patch" string.
 * - @ref initHipDeviceProperties — Performs the actual HIP
 *   runtime queries and populates the cached struct.
 *
 * @see DetectHipDeviceInfo.hpp  Type and function declarations.
 * @see DetectHipDevice.cpp      Device availability detection.
 * @see device.c                 Core device layer that calls these.
 */

#include <iostream>
#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <sstream>
#include <string>

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "DetectHipDevice.hpp"
#include "DetectHipDeviceInfo.hpp"
#include <hip/hip_runtime_api.h>

namespace {

/**
 * @brief Format a byte count as a human-readable memory string.
 *
 * @details
 * Converts @p bytes to the most appropriate unit:
 * - 8 GiB or more → "X.X GiB"
 * - 1 MiB or more → "X.X MiB"
 * - Otherwise → "N bytes"
 *
 * @param[in] bytes  The byte count to format.
 *
 * @return A formatted string with one decimal place for GiB/MiB.
 */
std::string formatMemory(size_t bytes) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(1);

  if (bytes >= static_cast<size_t>(8) * 1024 * 1024 * 1024) {
    out << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return out.str();
  }

  if (bytes >= static_cast<size_t>(1024) * 1024) {
    out << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB";
    return out.str();
  }

  return std::to_string(bytes) + " bytes";
}

/**
 * @brief Convert a HIP integer version to a "major.minor.patch" string.
 *
 * @details
 * HIP encodes versions as `major * 10000000 + minor * 100000 +
 * patch`.  This function extracts and formats the three
 * components.
 *
 * @param[in] version  The HIP integer version encoding.
 *
 * @return A string in the format "major.minor.patch".
 */
std::string formatHipVersion(int version) {
  // HIP version encoding: major*10000000 + minor*100000 + patch
  const int major = version / 10000000;
  const int minor = (version % 10000000) / 100000;
  const int patch = (version % 100000);
  return std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch);
}

/**
 * @brief Query the HIP runtime and populate device properties.
 *
 * @details
 * Calls `hipGetDeviceProperties`, `hipDriverGetVersion`, and
 * `hipRuntimeGetVersion` to fill a @ref hipDetectedDeviceProps_t
 * struct.  The result is cached in a `static` local variable for
 * subsequent calls.
 *
 * If device detection has already been performed (via
 * @ref was_device_detection_done), uses the detected device index;
 * otherwise defaults to device 0.
 *
 * @param[out] status  Receives `novaSuccess` on success, or an
 *                     error code with the HIP error string on
 *                     failure.
 *
 * @return A cached @ref hipDetectedDeviceProps_t with valid data
 *         when @p status indicates success.
 */
hipDetectedDeviceProps_t initHipDeviceProperties(novaStatus_t *status) {
  static const hipDetectedDeviceProps_t result =
      [&status]() -> hipDetectedDeviceProps_t {
    hipDeviceProp_t prop{};
    hipError_t err = was_device_detection_done()
                         ? hipGetDeviceProperties(&prop, getHipDeviceId())
                         : hipGetDeviceProperties(&prop, 0);

    if (err != hipSuccess) {
      status->err = (err != hipErrorInvalidValue) ? novaInvalidValue
                                                   : novaDeviceNotAvailable;
      status->message = hipGetErrorString(err);
      return {.isAvailable = false};
    }

    int driverVer = 0;
    int runtimeVer = 0;

    hipError_t driverErr = hipDriverGetVersion(&driverVer);
    if (driverErr != hipSuccess) {
      status->err = (driverErr != hipErrorInvalidValue)
                        ? novaInvalidValue
                        : novaDeviceNotAvailable;
      status->message = hipGetErrorString(driverErr);
      return {};
    }

    hipError_t runtimeErr = hipRuntimeGetVersion(&runtimeVer);
    if (runtimeErr != hipSuccess) {
      status->err = (runtimeErr != hipErrorInvalidValue)
                        ? novaInvalidValue
                        : novaDeviceNotAvailable;
      status->message = hipGetErrorString(runtimeErr);
      return {};
    }

    return {.isAvailable = true,
            .name = prop.name,
            .runtimeVersion = formatHipVersion(runtimeVer),
            .driverVersion = formatHipVersion(driverVer),
            .totalGlobalMem = formatMemory(prop.totalGlobalMem),
            .gcnArchName = prop.gcnArchName,
            .multiProcessorCount = prop.multiProcessorCount,
            .warpSize = prop.warpSize,
            .maxThreadsPerBlock = prop.maxThreadsPerBlock,
            .maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor};
  }();

  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, NULL);
  return result;
}

} // namespace

/**
 * @brief Print HIP device properties to stdout.
 *
 * @details
 * Queries device 0 properties via @ref initHipDeviceProperties
 * and prints them using ANSI colour codes.  When @p verbose is
 * `false`, a concise two-line summary is printed.  When @p verbose
 * is `true`, a detailed multi-line block is printed.
 *
 * @param[in] verbose  If `true`, print the full property block
 *                     (name, GCN architecture, memory, CUs, warp
 *                     size, thread limits, driver/runtime versions).
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
novaStatus_t printHipDeviceInfo(bool verbose) {
  novaStatus_t status;
  hipDetectedDeviceProps_t result = initHipDeviceProperties(&status);

  if (status.err != novaSuccess) {
    return status;
  }

  if (verbose) {
    std::cout << NCORE_LOG_PREFIX NCORE_LOG_BOLD << " === HIP Device 0 ===\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Name:                  " << NCORE_LOG_VALUE << result.name
              << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   GCN Architecture:      " << NCORE_LOG_VALUE
              << result.gcnArchName << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Total Global Memory:   " << NCORE_LOG_VALUE
              << result.totalGlobalMem << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   CUs:                   " << NCORE_LOG_VALUE
              << result.multiProcessorCount << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Warp Size:             " << NCORE_LOG_VALUE
              << result.warpSize << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Max Threads/Block:     " << NCORE_LOG_VALUE
              << result.maxThreadsPerBlock << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Max Threads/CU:        " << NCORE_LOG_VALUE
              << result.maxThreadsPerMultiProcessor << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Driver Version:        " << NCORE_LOG_VALUE
              << result.driverVersion << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Runtime Version:       " << NCORE_LOG_VALUE
              << result.runtimeVersion << "\n"
              << NCORE_LOG_RESET;
    return {.err = novaSuccess,
            .message = nova_get_error_msg(novaSuccess, NULL)};
  }

  std::cout << NCORE_LOG_PREFIX << " [HIP] Device 0 "
            << NCORE_LOG_VALUE NCORE_LOG_BOLD << result.name << NCORE_LOG_RESET
            << " | Arch " << NCORE_LOG_VALUE << result.gcnArchName
            << NCORE_LOG_RESET << " | " << NCORE_LOG_VALUE
            << result.totalGlobalMem << NCORE_LOG_RESET << " | "
            << NCORE_LOG_VALUE << result.multiProcessorCount << " CUs\n"
            << NCORE_LOG_RESET << NCORE_LOG_PREFIX << " [HIP] Driver "
            << NCORE_LOG_VALUE << "v" << result.driverVersion << NCORE_LOG_RESET
            << " | Runtime " << NCORE_LOG_VALUE << "v" << result.runtimeVersion
            << "\n"
            << NCORE_LOG_RESET;

  return {.err = novaSuccess, .message = nova_get_error_msg(novaSuccess, NULL)};
}

/**
 * @brief Retrieve the cached HIP device properties.
 *
 * @details
 * Returns the cached @ref hipDetectedDeviceProps_t populated on
 * the first call to @ref initHipDeviceProperties.  Subsequent
 * calls return the cached value without additional HIP runtime
 * API calls.
 *
 * @param[out] status  Receives `novaSuccess` on success, or an
 *                     error code on failure.
 *
 * @return Cached device properties.  Check @ref isAvailable to
 *         determine whether the data is valid.
 *
 * @note Thread-safe.  The result is cached in a `static` local
 *       variable initialised exactly once (C++11 guarantee).
 */
hipDetectedDeviceProps_t getHipDeviceProperties(novaStatus_t *status) noexcept {
  static const hipDetectedDeviceProps_t props = initHipDeviceProperties(status);
  return props;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
novaStatus_t printHipDeviceInfo(bool) {
  return {.err = novaBackendNotCompiled,
          .message = nova_get_error_msg(novaBackendNotCompiled, NULL)};
}

/** @brief Stub: HIP runtime headers not available. */
hipDetectedDeviceProps_t getHipDeviceProperties(novaStatus_t *status) {
  status->err = novaBackendNotCompiled;
  status->message = nova_get_error_msg(novaBackendNotCompiled, NULL);
  return {.isAvailable = false};
}

#endif
#endif /* NOVA_HAS_HIP */
