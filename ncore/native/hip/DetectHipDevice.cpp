/**
 * @file DetectHipDevice.cpp
 * @brief HIP device availability detection implementation.
 *
 * @details
 * Implements the public detection API declared in
 * @ref DetectHipDevice.hpp.  Uses the HIP runtime API
 * (`hipGetDeviceCount`) to probe for an available GPU and caches
 * the result in module-level atomics for subsequent queries.
 *
 * The file is conditionally compiled behind `NOVA_HAS_HIP` and
 * `__has_include(<hip/hip_runtime_api.h>)`.  When HIP headers are
 * unavailable, stub functions that return `false` / `-1` are
 * provided.
 *
 * A `__HIP_PLATFORM_AMD__` macro is defined when neither AMD nor
 * NVIDIA platform macros are set, ensuring clangd and clang-tidy
 * can parse the HIP headers correctly.
 *
 * ## Thread Safety
 *
 * The initial probe is guarded by `std::call_once`, ensuring that
 * `hipGetDeviceCount` is called exactly once even under concurrent
 * access.  The cached results are stored in `std::atomic` variables
 * with `std::memory_order_release` (write) and
 * `std::memory_order_acquire` (read) semantics.
 *
 * ## One-shot Caching
 *
 * After the first call to @ref isHipDeviceAvailable, the probe
 * result is cached.  All subsequent calls — regardless of the
 * @p log or @p verbose arguments — return the cached value without
 * touching the HIP runtime API.
 *
 * @see DetectHipDevice.hpp  Public function declarations.
 * @see DetectHipDeviceInfo.cpp  Device property queries.
 * @see device.c  Core device layer that calls these functions.
 */

#include <atomic>
#include <iostream>
#include <mutex>
#include <ncore/core/status.h>

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
 * @var activeHipDeviceId
 * @brief 0-based index of the detected HIP device.
 *
 * Stores `-1` before detection, or `0` after a successful probe.
 * Accessed with acquire/release memory ordering.
 */
std::atomic<int> activeHipDeviceId{-1};

/**
 * @var hipDeviceAvailable
 * @brief Whether a HIP-capable device was found.
 *
 * Set to `true` after a successful probe.  Accessed with
 * acquire/release memory ordering.
 */
std::atomic_bool hipDeviceAvailable{false};

/**
 * @var hipDeviceProbeOnce
 * @brief One-shot initialisation guard for the HIP device probe.
 *
 * Ensures @ref probeHipDevice is called exactly once, even from
 * multiple threads.
 */
std::once_flag hipDeviceProbeOnce;

/**
 * @var hipDeviceLogMtx
 * @brief Mutex protecting stderr output during detection.
 *
 * Serialises error messages printed by @ref isHipDeviceAvailable
 * when the probe fails.
 */
std::mutex hipDeviceLogMtx;

/**
 * @struct DetectionResult
 * @brief Outcome of a HIP device probe.
 *
 * @var DetectionResult::available
 * @brief `true` if a HIP device was found.
 *
 * @var DetectionResult::errorMessage
 * @brief Human-readable error string from `hipGetErrorString`,
 *        or `nullptr` on success.
 */
struct DetectionResult {
  bool available = false;
  const char *errorMessage = nullptr;
};

/**
 * @brief Perform the actual HIP device detection.
 *
 * @details
 * Calls `hipGetDeviceCount` to query the number of available HIP
 * devices.  If at least one device is found, stores `0` in
 * @ref activeHipDeviceId and `true` in @ref hipDeviceAvailable.
 *
 * @return A @ref DetectionResult indicating whether a device was
 *         found and any error message from the HIP runtime.
 */
DetectionResult probeHipDevice() {
  int count = 0;
  const hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) {
    return {.available = false, .errorMessage = hipGetErrorString(err)};
  }

  if (count == 0) {
    return {};
  }

  activeHipDeviceId.store(0, std::memory_order_release);
  hipDeviceAvailable.store(true, std::memory_order_release);
  return {.available = true};
}
} // namespace

/**
 * @brief Probe the HIP runtime for an available GPU device.
 *
 * @details
 * Uses `std::call_once` to ensure the probe runs exactly once.
 * On success, optionally prints device information via
 * @ref printHipDeviceInfo when @p log is `true`.
 *
 * @param[in] log      If `true`, print error messages to stderr on
 *                     failure, or call @ref printHipDeviceInfo on
 *                     success.
 * @param[in] verbose  If `true`, pass verbose flag to
 *                     @ref printHipDeviceInfo for detailed output.
 *
 * @return `true` when a HIP-capable device is found.  `false` if
 *         no device is available or the runtime reports an error.
 *
 * @note Thread-safe.  The probe is serialised by `std::call_once`.
 *       The log mutex serialises stderr output.
 */
bool isHipDeviceAvailable(bool log, bool verbose) {
  DetectionResult result;
  novaStatus_t status;
  std::call_once(hipDeviceProbeOnce, [&] { result = probeHipDevice(); });

  if (!result.available &&
      !hipDeviceAvailable.load(std::memory_order_acquire)) {
    if (log && result.errorMessage != nullptr) {
      const std::lock_guard<std::mutex> lock(hipDeviceLogMtx);
      std::cerr << "[HIP] isHipDeviceAvailable: error obtaining device count.\n"
                << "Error message: '" << result.errorMessage << "'\n";
    }
    return false;
  }

  if (log) {
    status = printHipDeviceInfo(verbose);
    if (status.err != novaSuccess) {
      std::cerr << "[HIP] isHipDeviceAvailable -> printHipDeviceInfo: Error "
                   "obtainig device info.\n"
                << "Details: '" << status.message << "'\n";
    }
  }

  return hipDeviceAvailable.load(std::memory_order_acquire);
}

/**
 * @brief Return the 0-based device index of the detected HIP GPU.
 *
 * @details
 * Returns the value stored in @ref activeHipDeviceId by the most
 * recent successful probe.  The value is `0` when a GPU is found,
 * or `-1` if detection has not yet been performed.
 *
 * @return 0-based HIP device index, or `-1` if unavailable.
 *
 * @note The return value is only meaningful after at least one call
 *       to @ref isHipDeviceAvailable has completed.
 */
int getHipDeviceId() {
  return activeHipDeviceId.load(std::memory_order_acquire);
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
bool isHipDeviceAvailable(bool, bool) { return false; }
/** @brief Stub: HIP runtime headers not available. */
int getHipDeviceId() { return -1; }

#endif
#endif /* NOVA_HAS_HIP */
