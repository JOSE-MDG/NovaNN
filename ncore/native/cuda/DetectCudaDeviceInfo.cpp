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
 * The file is conditionally compiled behind @c NOVA_HAS_CUDA and
 * @c __has_include(<cuda_runtime_api.h>).  When CUDA headers are
 * unavailable, stub functions returning error statuses are
 * provided.
 *
 * @section caching Caching
 *
 * Device properties are queried once and cached in a @c static
 * local variable within @ref initCudaDeviceProperties.  Subsequent
 * calls to @ref getCudaDeviceProperties and @ref
 * printCudaDeviceInfo return the cached values without additional
 * CUDA runtime API calls.
 *
 * @section internal-helpers Internal Helpers
 *
 * @li @ref formatMemory — Converts byte counts to human-readable
 *   strings (GiB / MiB / bytes).
 * @li @ref formatBandwidth — Converts bytes/s to human-readable
 *   bandwidth (GB/s).
 * @li @ref formatCudaVersion — Converts the CUDA integer version
 *   encoding to a "major.minor" string.
 * @li @ref initCudaDeviceProperties — Performs the actual CUDA
 *   runtime queries and populates the cached struct.
 *
 * @see DetectCudaDeviceInfo.hpp  Type and function declarations.
 * @see DetectCudaDevice.cpp      Device availability detection.
 * @see device.c                  Core device layer that calls these.
 */

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/heuristics/kernels/patterns/common.hh>
#include <ncore/headeronly/macros.h>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>

#include "DetectCudaDevice.hpp"
#include "DetectCudaDeviceInfo.hpp"

namespace {

/**
 * @brief Format a byte count as a human-readable memory string.
 *
 * @details
 * Converts @p bytes to the most appropriate unit:
 * @li 8 GiB or more → "X.X GiB"
 * @li 1 MiB or more → "X.X MiB"
 * @li Otherwise → "N bytes"
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
 * @brief Format a bandwidth figure as a human-readable string.
 *
 * @details
 * Converts @p bytesPerSec to gigabytes per second with one decimal
 * place (decimal giga, the industry unit for DRAM bandwidth).
 *
 * @param[in] bytesPerSec  Theoretical bandwidth in bytes per second.
 *
 * @return A formatted string (e.g., "672.0 GB/s").
 */
std::string formatBandwidth(uint64_t bytesPerSec) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(1);

  out << static_cast<double>(bytesPerSec) / 1e9 << " GB/s";
  return out.str();
}

/**
 * @brief Convert a CUDA integer version to a "major.minor" string.
 *
 * @details
 * CUDA encodes versions as @c major * 1000 + minor * 10.  This
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
 * Calls @c cudaGetDeviceProperties, @c cudaDriverGetVersion, and
 * @c cudaRuntimeGetVersion to fill a @ref cudaDetectedDeviceProps_t
 * struct.  The result is cached in a @c static local variable for
 * subsequent calls.
 *
 * If device detection has already been performed (via
 * @ref was_device_detection_done), uses the detected device index;
 * otherwise defaults to device 0.
 *
 * @param[out] status  Receives @c novaSuccess on success, or an
 *                     error code with the CUDA error string on
 *                     failure.
 *
 * @return A cached @ref cudaDetectedDeviceProps_t with valid data
 *         when @p status indicates success.
 */
cudaDetectedDeviceProps_t initCudaDeviceProperties(novaStatus_t *status) {
  static const cudaDetectedDeviceProps_t result =
      [&status]() -> cudaDetectedDeviceProps_t {
    const int deviceId = was_device_detection_done() ? getCudaDeviceId() : 0;
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);

    if (err != cudaSuccess) {
      status->err = (err == cudaErrorInvalidValue) ? novaInvalidValue
                                                   : novaDeviceNotAvailable;
      status->message = cudaGetErrorString(err);
      return {};
    }

    int driverVer = 0;
    int runtimeVer = 0;

    cudaError_t driverErr = cudaDriverGetVersion(&driverVer);
    if (driverErr != cudaSuccess) {
      status->err = (driverErr == cudaErrorInvalidValue)
                        ? novaInvalidValue
                        : novaDeviceNotAvailable;
      status->message = cudaGetErrorString(driverErr);
      return {};
    }
    cudaError_t runtimeErr = cudaRuntimeGetVersion(&runtimeVer);
    if (runtimeErr != cudaSuccess) {
      status->err = (runtimeErr == cudaErrorInvalidValue)
                        ? novaInvalidValue
                        : novaDeviceNotAvailable;
      status->message = cudaGetErrorString(runtimeErr);
      return {};
    }

    namespace hk = ncore::heuristics::kernels;
    novaStatus_t attrStatus{};
    const cudaDetectedDeviceAttrs_t attrs =
        getCudaDeviceAttributes(&attrStatus);
    const uint32_t arch =
        static_cast<uint32_t>((prop.major * 100) + prop.minor);
    const uint64_t peak = hk::detail::cudaPeakFp32Flops(
        arch, static_cast<uint32_t>(prop.multiProcessorCount),
        static_cast<uint32_t>(attrs.clockRate));
    const uint64_t bandwidth = hk::detail::theoreticalBandwidth(
        static_cast<uint32_t>(attrs.memoryClockRate),
        static_cast<uint32_t>(prop.memoryBusWidth));

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
            .maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor,
            .maxBlocksPerMultiProcessor = prop.maxBlocksPerMultiProcessor,
            .major = prop.major,
            .minor = prop.minor,
            .clockRate = attrs.clockRate,
            .memoryClockRate = attrs.memoryClockRate,
            .memoryBusWidth = prop.memoryBusWidth,
            .sharedMemPerBlock = prop.sharedMemPerBlock,
            .sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor,
            .regsPerMultiprocessor = prop.regsPerMultiprocessor,
            .maxGridSize = {prop.maxGridSize[0], prop.maxGridSize[1],
                            prop.maxGridSize[2]},
            .l2CacheSize = prop.l2CacheSize,
            .persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize,
            .peakFp32Flops = peak,
            .memBandwidth = formatBandwidth(bandwidth)};
  }();
  if (!result.isAvailable) {
    status->err = novaDeviceNotAvailable;
    status->message = nova_get_error_msg(status->err, nullptr);
    return result;
  }

  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return result;
}
} // namespace

/**
 * @brief Retrieve auxiliary device attributes.
 *
 * @details
 * Queries the SM and DRAM peak clocks via @c cudaDeviceGetAttribute
 * (absent from @c cudaDeviceProp in current toolkits) on the detected
 * device, defaulting to device 0 before detection ran. The result is
 * cached in a @c static local variable; later calls pay no runtime
 * cost. A failed query degrades that field to 0 without failing.
 * Query after detection: the device id is captured on the first call,
 * so a later device switch keeps returning the first device attrs
 * until process restart.
 *
 * @param[out] status  Receives @c novaSuccess.
 *
 * @return Cached attributes (zeros where unknown).
 */
cudaDetectedDeviceAttrs_t
getCudaDeviceAttributes(novaStatus_t *status) noexcept {
  static const cudaDetectedDeviceAttrs_t attrs = [] {
    const int deviceId = was_device_detection_done() ? getCudaDeviceId() : 0;
    int clockKHz = 0;
    int memClockKHz = 0;
    if (cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, deviceId) !=
        cudaSuccess) {
      clockKHz = 0;
    }
    if (cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate,
                               deviceId) != cudaSuccess) {
      memClockKHz = 0;
    }
    return cudaDetectedDeviceAttrs_t{.clockRate = clockKHz,
                                     .memoryClockRate = memClockKHz};
  }();
  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return attrs;
}

/**
 * @brief Print CUDA device properties to stdout.
 *
 * @details
 * Queries device 0 properties via @ref initCudaDeviceProperties
 * and prints them using ANSI colour codes.  When @p verbose is
 * @c false, a concise two-line summary is printed.  When @p verbose
 * is @c true, a detailed multi-line block is printed.
 *
 * @param[in] verbose  If @c true, print the full property block
 *                     (name, compute capability, memory, SMs, warp
 *                     size, thread limits, driver/runtime versions).
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
              << "   Memory Bandwidth:      " << NCORE_LOG_VALUE
              << result.memBandwidth << "\n"
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
            .message = nova_get_error_msg(novaSuccess, nullptr)};
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
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
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
 * @param[out] status  Receives @c novaSuccess on success, or an
 *                     error code on failure.
 *
 * @return Cached device properties.  Check @ref isAvailable to
 *         determine whether the data is valid.
 *
 * @note Thread-safe.  The result is cached in a @c static local
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
          .message = nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}
/** @brief Stub: CUDA runtime headers not available. */
cudaDetectedDeviceProps_t getCudaDeviceProperties(novaStatus_t *status) {
  status->err = novaBackendNotCompiled;
  status->message = nova_get_error_msg(novaBackendNotCompiled, nullptr);
  return {.isAvailable = false};
};
#endif
#endif /* NOVA_HAS_CUDA */
