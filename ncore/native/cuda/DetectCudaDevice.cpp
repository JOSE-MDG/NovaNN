/**
 * @file DetectCudaDevice.cpp
 * @brief CUDA device availability detection implementation.
 *
 * @details
 * Implements the public detection API declared in
 * @ref DetectCudaDevice.hpp.  Uses the CUDA runtime API
 * (`cudaGetDeviceCount`) to probe for an available GPU and caches
 * the result in module-level atomics for subsequent queries.
 *
 * The file is conditionally compiled behind `NOVA_HAS_CUDA` and
 * `__has_include(<cuda_runtime_api.h>)`.  When CUDA headers are
 * unavailable, stub functions that return `false` / `-1` are
 * provided.
 *
 * ## Thread Safety
 *
 * The initial probe is guarded by `std::call_once`, ensuring that
 * `cudaGetDeviceCount` is called exactly once even under concurrent
 * access.  The cached results are stored in `std::atomic` variables
 * with `std::memory_order_release` (write) and
 * `std::memory_order_acquire` (read) semantics.
 *
 * ## One-shot Caching
 *
 * After the first call to @ref isCudaDeviceAvailable, the probe
 * result is cached.  All subsequent calls — regardless of the
 * @p log or @p verbose arguments — return the cached value without
 * touching the CUDA runtime API.
 *
 * @see DetectCudaDevice.hpp  Public function declarations.
 * @see DetectCudaDeviceInfo.cpp  Device property queries.
 * @see device.c  Core device layer that calls these functions.
 */

#include <atomic>
#include <iostream>
#include <mutex>
#include <ncore/core/status.h>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include "DetectCudaDevice.hpp"
#include "DetectCudaDeviceInfo.hpp"
#include <cuda_runtime_api.h>

namespace {
/**
 * @var activeCudaDeviceId
 * @brief 0-based index of the detected CUDA device.
 *
 * Stores `-1` before detection, or `0` after a successful probe.
 * Accessed with acquire/release memory ordering.
 */
std::atomic<int> activeCudaDeviceId{-1};

/**
 * @var cudaDeviceAvailable
 * @brief Whether a CUDA-capable device was found.
 *
 * Set to `true` after a successful probe.  Accessed with
 * acquire/release memory ordering.
 */
std::atomic<bool> cudaDeviceAvailable{false};

/**
 * @var cudaDeviceProbeOnce
 * @brief One-shot initialisation guard for the CUDA device probe.
 *
 * Ensures @ref probeCudaDevice is called exactly once, even from
 * multiple threads.
 */
std::once_flag cudaDeviceProbeOnce;

/**
 * @var cudaDeviceLogMtx
 * @brief Mutex protecting stderr output during detection.
 *
 * Serialises error messages printed by @ref isCudaDeviceAvailable
 * when the probe fails.
 */
std::mutex cudaDeviceLogMtx;

/**
 * @struct DetectionResult
 * @brief Outcome of a CUDA device probe.
 *
 * @var DetectionResult::available
 * @brief `true` if a CUDA device was found.
 *
 * @var DetectionResult::errorMessage
 * @brief Human-readable error string from `cudaGetErrorString`,
 *        or `nullptr` on success.
 */
struct DetectionResult {
  bool available = false;
  const char *errorMessage = nullptr;
};

/**
 * @brief Perform the actual CUDA device detection.
 *
 * @details
 * Calls `cudaGetDeviceCount` to query the number of available CUDA
 * devices.  If at least one device is found, stores `0` in
 * @ref activeCudaDeviceId and `true` in @ref cudaDeviceAvailable.
 *
 * @return A @ref DetectionResult indicating whether a device was
 *         found and any error message from the CUDA runtime.
 */
DetectionResult probeCudaDevice() {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    return {.available = false, .errorMessage = cudaGetErrorString(err)};
  }

  if (count == 0) {
    return {};
  }

  activeCudaDeviceId.store(0, std::memory_order_release);
  cudaDeviceAvailable.store(true, std::memory_order_release);
  return {.available = true};
}
} // namespace

/**
 * @brief Probe the CUDA runtime for an available GPU device.
 *
 * @details
 * Uses `std::call_once` to ensure the probe runs exactly once.
 * On success, optionally prints device information via
 * @ref printCudaDeviceInfo when @p log is `true`.
 *
 * @param[in] log      If `true`, print error messages to stderr on
 *                     failure, or call @ref printCudaDeviceInfo on
 *                     success.
 * @param[in] verbose  If `true`, pass verbose flag to
 *                     @ref printCudaDeviceInfo for detailed output.
 *
 * @return `true` when a CUDA-capable device is found.  `false` if
 *         no device is available or the runtime reports an error.
 *
 * @note Thread-safe.  The probe is serialised by `std::call_once`.
 *       The log mutex serialises stderr output.
 */
bool isCudaDeviceAvailable(bool log, bool verbose) {
  DetectionResult result;
  novaStatus_t status;
  std::call_once(cudaDeviceProbeOnce, [&] { result = probeCudaDevice(); });

  if (!result.available &&
      !cudaDeviceAvailable.load(std::memory_order_acquire)) {
    if (log && result.errorMessage != nullptr) {
      const std::lock_guard<std::mutex> lock(cudaDeviceLogMtx);
      std::cerr
          << "[CUDA] isCudaDeviceAvailable: Error obtaining device count.\n"
          << "Error message: '" << result.errorMessage << "'\n";
    }
    return false;
  }

  if (log) {
    status = printCudaDeviceInfo(verbose);
    if (status.err != novaSuccess) {
      std::cerr << "[CUDA] isCudaDeviceAvailable -> printCudaDeviceInfo: Error "
                   "obtainig device info.\n"
                << "Details: '" << status.message << "'\n";
    }
  }

  return cudaDeviceAvailable.load(std::memory_order_acquire);
}

/**
 * @brief Return the 0-based device index of the detected CUDA GPU.
 *
 * @details
 * Returns the value stored in @ref activeCudaDeviceId by the most
 * recent successful probe.  The value is `0` when a GPU is found,
 * or `-1` if detection has not yet been performed.
 *
 * @return 0-based CUDA device index, or `-1` if unavailable.
 *
 * @note The return value is only meaningful after at least one call
 *       to @ref isCudaDeviceAvailable has completed.
 */
int getCudaDeviceId() {
  return activeCudaDeviceId.load(std::memory_order_acquire);
}

#else // !__has_include(<cuda_runtime_api.h>)

/** @brief Stub: CUDA runtime headers not available. */
bool isCudaDeviceAvailable(bool, bool) { return false; }
/** @brief Stub: CUDA runtime headers not available. */
int getCudaDeviceId() { return -1; }

#endif
#endif /* NOVA_HAS_CUDA */
