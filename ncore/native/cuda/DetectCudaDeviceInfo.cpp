/**
 * @file DetectCudaDeviceInfo.cpp
 * @brief CUDA device property query and formatted output implementation.
 *
 * @details
 * Implements the property retrieval and printing API declared in
 * @ref DetectCudaDeviceInfo.hpp.  Queries the CUDA runtime for
 * device 0 properties (name, compute capability, memory, SMs,
 * warp size, thread limits, driver/runtime versions) and formats
 * them for human-readable output.
 *
 * The file is conditionally compiled behind `NOVA_HAS_CUDA` and
 * `__has_include(<cuda_runtime_api.h>)`.  When CUDA headers are
 * unavailable, stub functions returning error statuses are
 * provided.
 *
 * ## Caching
 *
 * Device properties are queried once and cached in a `static`
 * local variable within @ref initCudaDeviceProperties.  Subsequent
 * calls to @ref getCudaDeviceProperties and @ref
 * printCudaDeviceInfo return the cached values without additional
 * CUDA runtime API calls.
 *
 * ## Internal Helpers
 *
 * - @ref formatMemory — Converts byte counts to human-readable
 *   strings (GiB / MiB / bytes).
 * - @ref formatCudaVersion — Converts the CUDA integer version
 *   encoding to a "major.minor" string.
 * - @ref initCudaDeviceProperties — Performs the actual CUDA
 *   runtime queries and populates the cached struct.
 *
 * @see DetectCudaDeviceInfo.hpp  Type and function declarations.
 * @see DetectCudaDevice.cpp      Device availability detection.
 * @see device.c                  Core device layer that calls these.
 */

#include <iostream>
#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <sstream>
#include <string>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include "DetectCudaDevice.hpp"
#include "DetectCudaDeviceInfo.hpp"
#include <cuda_runtime_api.h>

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
 * @brief Convert a CUDA integer version to a "major.minor" string.
 *
 * @details
 * CUDA encodes versions as `major * 1000 + minor * 10`.  This
 * function extracts and formats the two components.
 *
 * @param[in] version  The CUDA integer version encoding.
 *
 * @return A string in the format "major.minor".
 */
std::string formatCudaVersion(int version) {
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

/**
 * @brief Query the CUDA runtime and populate device properties.
 *
 * @details
 * Calls `cudaGetDeviceProperties`, `cudaDriverGetVersion`, and
 * `cudaRuntimeGetVersion` to fill a @ref cudaDetectedDeviceProps_t
 * struct.  The result is cached in a `static` local variable for
 * subsequent calls.
 *
 * If device detection has already been performed (via
 * @ref was_device_detection_done), uses the detected device index;
 * otherwise defaults to device 0.
 *
 * @param[out] status  Receives `novaSuccess` on success, or an
 *                     error code with the CUDA error string on
 *                     failure.
 *
 * @return A cached @ref cudaDetectedDeviceProps_t with valid data
 *         when @p status indicates success.
 */
cudaDetectedDeviceProps_t initCudaDeviceProperties(novaStatus_t *status) {
  static const cudaDetectedDeviceProps_t result =
      [&status]() -> cudaDetectedDeviceProps_t {
    cudaDeviceProp prop{};
    cudaError_t err = was_device_detection_done()
                          ? cudaGetDeviceProperties(&prop, getCudaDeviceId())
                          : cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
      status->err = (err != cudaErrorInvalidValue) ? novaInvalidValue
                                                    : novaDeviceNotAvailable;
      status->message = cudaGetErrorString(err);
      return {.isAvailable = false};
    }

    int driverVer = 0;
    int runtimeVer = 0;

    cudaError_t driverErr = cudaDriverGetVersion(&driverVer);
    if (driverErr != cudaSuccess) {
      status->err = (driverErr != cudaErrorInvalidValue)
                        ? novaInvalidValue
                        : novaDeviceNotAvailable;
      status->message = cudaGetErrorString(driverErr);
      return {};
    }
    cudaError_t runtimeErr = cudaRuntimeGetVersion(&runtimeVer);
    if (runtimeErr != cudaSuccess) {
      status->err = (runtimeErr != cudaErrorInvalidValue)
                        ? novaInvalidValue
                        : novaDeviceNotAvailable;
      status->message = cudaGetErrorString(runtimeErr);
      return {};
    }

    return {.isAvailable = true,
            .name = prop.name,
            .runtimeVersion = formatCudaVersion(runtimeVer),
            .driverVersion = formatCudaVersion(driverVer),
            .totalGlobalMem = formatMemory(prop.totalGlobalMem),
            .comCapability =
                std::to_string(prop.major) + "." + std::to_string(prop.minor),
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
 * @brief Print CUDA device properties to stdout.
 *
 * @details
 * Queries device 0 properties via @ref initCudaDeviceProperties
 * and prints them using ANSI colour codes.  When @p verbose is
 * `false`, a concise two-line summary is printed.  When @p verbose
 * is `true`, a detailed multi-line block is printed.
 *
 * @param[in] verbose  If `true`, print the full property block
 *                     (name, compute capability, memory, SMs, warp
 *                     size, thread limits, driver/runtime versions).
 *                     If `false`, print a concise summary.
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
novaStatus_t printCudaDeviceInfo(bool verbose) {
  novaStatus_t status;
  cudaDetectedDeviceProps_t result = initCudaDeviceProperties(&status);

  if (status.err != novaSuccess) {
    return status;
  }

  if (verbose) {
    std::cout << NCORE_LOG_PREFIX NCORE_LOG_BOLD << " === CUDA Device 0 ===\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Name:                  " << NCORE_LOG_VALUE << result.name
              << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Compute Capability:    " << NCORE_LOG_VALUE
              << result.comCapability << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Total Global Memory:   " << NCORE_LOG_VALUE
              << result.totalGlobalMem << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   SMs:                   " << NCORE_LOG_VALUE
              << result.multiProcessorCount << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Warp Size:             " << NCORE_LOG_VALUE
              << result.warpSize << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Max Threads/Block:     " << NCORE_LOG_VALUE
              << result.maxThreadsPerBlock << "\n"
              << NCORE_LOG_RESET << NCORE_LOG_PREFIX
              << "   Max Threads/SM:        " << NCORE_LOG_VALUE
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

  std::cout << NCORE_LOG_PREFIX << " [CUDA] Device 0 "
            << NCORE_LOG_VALUE NCORE_LOG_BOLD << result.name << NCORE_LOG_RESET
            << " | Compute " << NCORE_LOG_VALUE << result.comCapability
            << NCORE_LOG_RESET << " | " << NCORE_LOG_VALUE
            << result.totalGlobalMem << NCORE_LOG_RESET << " | "
            << NCORE_LOG_VALUE << result.multiProcessorCount << " SMs\n"
            << NCORE_LOG_RESET << NCORE_LOG_PREFIX << " [CUDA] Driver "
            << NCORE_LOG_VALUE << "v" << result.driverVersion << NCORE_LOG_RESET
            << " | Runtime " << NCORE_LOG_VALUE << "v" << result.runtimeVersion
            << "\n"
            << NCORE_LOG_RESET;
  return {.err = novaSuccess, .message = nova_get_error_msg(novaSuccess, NULL)};
}

/**
 * @brief Retrieve the cached CUDA device properties.
 *
 * @details
 * Returns the cached @ref cudaDetectedDeviceProps_t populated on
 * the first call to @ref initCudaDeviceProperties.  Subsequent
 * calls return the cached value without additional CUDA runtime
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
cudaDetectedDeviceProps_t
getCudaDeviceProperties(novaStatus_t *status) noexcept {
  static const cudaDetectedDeviceProps_t props =
      initCudaDeviceProperties(status);
  return props;
}

#else // !__has_include(<cuda_runtime_api.h>)

/** @brief Stub: CUDA runtime headers not available. */
novaStatus_t printCudaDeviceInfo(bool verbose) {
  return {.err = novaBackendNotCompiled,
          .message = nova_get_error_msg(novaBackendNotCompiled, NULL)};
}
/** @brief Stub: CUDA runtime headers not available. */
cudaDetectedDeviceProps_t getCudaDeviceProperties(novaStatus_t *status) {
  status->err = novaBackendNotCompiled;
  status->message = nova_get_error_msg(novaBackendNotCompiled, NULL);
  return {.isAvailable = false};
};
#endif
#endif /* NOVA_HAS_CUDA */
