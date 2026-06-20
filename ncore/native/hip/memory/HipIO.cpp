/**
 * @file HipIO.cpp
 * @brief HIP data transfer implementation.
 *
 * @details
 * Implements the master memcpy dispatcher @ref hipTransfer using
 * `hipMemcpyAsync` on a reusable HIP stream for all copy
 * directions.  The stream is created once and lives for the
 * lifetime of the process (singleton pattern).
 *
 * The file is conditionally compiled behind `NOVA_HAS_HIP` and
 * `__has_include(<hip/hip_runtime_api.h>)`.  When HIP headers are
 * unavailable, a stub function returning an error status is
 * provided.
 *
 * A `__HIP_PLATFORM_AMD__` macro is defined when neither AMD not
 * NVIDIA platform macros are set, ensuring clangd and clang-tidy
 * can parse the HIP headers correctly.
 *
 * ## Architecture
 *
 * The module exposes a single public function (@ref hipTransfer)
 * that handles H2D, D2H, and D2D transfers.  Internally it uses:
 * - @ref mapError — maps `hipError_t` to @ref hipStatus_t.
 * - @ref mapMemcpyKind — converts @ref DeviceMemcpyKind to
 *   `hipMemcpyKind`.
 * - @ref getStream — returns a singleton HIP stream.
 *
 * ## Error Mapping
 *
 * All HIP errors are mapped to a @ref hipStatus_t via a single
 * @ref mapError function that covers `hipSuccess` (code 0),
 * `hipErrorInvalidValue` (code 1), `hipErrorInvalidMemcpyDirection`
 * (code 2), `hipErrorInvalidResourceHandle` (code 3),
 * `hipErrorOutOfMemory` (code 4), and everything else (code -1).
 *
 * @see HipIO.hpp        Function declaration.
 * @see HipAllocator.cpp HIP memory allocation implementation.
 * @see ffi.cpp          Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "HipIO.hpp"
#include "HipAllocator.hpp"
#include <hip/hip_runtime_api.h>

namespace {

/**
 * @brief Map a HIP runtime error to a @ref hipStatus_t.
 *
 * @details
 * Converts `hipError_t` codes returned by `hipMemcpyAsync`
 * and `hipStreamSynchronize` into the project-standard
 * @ref hipStatus_t format.  Each error code is mapped to a
 * unique integer for programmatic handling, and the human-readable
 * error string is obtained via `hipGetErrorString`.
 *
 * The mapped codes are: `hipSuccess` → 0,
 * `hipErrorInvalidValue` → 1, `hipErrorInvalidMemcpyDirection`
 * → 2, `hipErrorInvalidResourceHandle` → 3,
 * `hipErrorOutOfMemory` → 4, everything else → -1.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return @ref hipStatus_t with the mapped code and message.
 *
 * @post  On success, `status.code == 0` and `status.msg == "ok"`.
 * @post  On failure, `status.code != 0` and `status.msg` contains
 *        the error string from `hipGetErrorString`.
 */
hipStatus_t mapError(hipError_t err) {
  hipStatus_t status = {};
  switch (err) {
  case hipSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case hipErrorInvalidValue:
    status.code = 1;
    status.msg = hipGetErrorString(err);
    return status;
  case hipErrorInvalidMemcpyDirection:
    status.code = 2;
    status.msg = hipGetErrorString(err);
    return status;
  case hipErrorInvalidResourceHandle:
    status.code = 3;
    status.msg = hipGetErrorString(err);
    return status;
  case hipErrorOutOfMemory:
    status.code = 4;
    status.msg = hipGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = hipGetErrorString(err);
    return status;
  }
}

/**
 * @brief Convert @ref DeviceMemcpyKind to `hipMemcpyKind`.
 *
 * @details
 * Maps the backend-agnostic @ref DeviceMemcpyKind enum to the
 * HIP-specific `hipMemcpyKind` enum used by `hipMemcpyAsync`.
 * The mapping is: `deviceMemcpyHostToDevice` →
 * `hipMemcpyHostToDevice`, `deviceMemcpyDeviceToHost` →
 * `hipMemcpyDeviceToHost`, `deviceMemcpyDeviceToDevice` →
 * `hipMemcpyDeviceToDevice`, default → `hipMemcpyDefault`.
 *
 * @param[in] kind  The device-agnostic copy direction.
 *
 * @return The corresponding HIP memcpy kind.
 */
hipMemcpyKind mapMemcpyKind(DeviceMemcpyKind kind) {
  switch (kind) {
  case DeviceMemcpyKind::deviceMemcpyHostToDevice:
    return hipMemcpyHostToDevice;
  case DeviceMemcpyKind::deviceMemcpyDeviceToHost:
    return hipMemcpyDeviceToHost;
  case DeviceMemcpyKind::deviceMemcpyDeviceToDevice:
    return hipMemcpyDeviceToDevice;
  default:
    return hipMemcpyDefault;
  }
}

/**
 * @brief Get or create the reusable HIP stream.
 *
 * @details
 * Returns a singleton HIP stream that is created on first call
 * and reused for all subsequent @ref hipTransfer operations.  The
 * stream is created with default flags (non-blocking, no
 * priority override).
 *
 * The stream is never explicitly destroyed.  The HIP runtime
 * reclaims all resources on process exit.  This avoids the
 * overhead of create/destroy per transfer and eliminates the
 * risk of use-after-free in concurrent scenarios.
 *
 * The `static` local variable is initialised exactly once, even
 * under concurrent access (C++11 guarantee).  The
 * `hipStreamCreate` call is serialised by the C++ runtime.
 *
 * @param[in,out] status an error status
 *
 * @return The singleton `hipStream_t`.
 */
hipStream_t getStream(hipStatus_t *status) {
  static hipStream_t stream = nullptr;
  if (stream == nullptr) {
    const hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
      *status = mapError(err);
    }
  }
  return stream;
}

} // namespace

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using `hipMemcpyAsync` on a
 * reusable internal HIP stream, then synchronises the stream
 * before returning.  The transfer direction is determined by
 * @p kind.
 *
 * ### Execution Flow
 *
 * @code{.cpp}
 *   hipStatus_t status;
 *   stream = getStream(&status);        // singleton stream
 *    if(!status) {
 *       // code
 *    }
 *   err = hipMemcpyAsync(dst, src,      // enqueue transfer
 *                        bytes, kind,
 *                        stream);
 *   if (err != hipSuccess) return mapError(err);
 *   sync_err = hipStreamSynchronize(stream);  // block
 *   if (sync_err != hipSuccess) return mapError(sync_err);
 *   return HIP_OK;
 * @endcode
 *
 * ### Error Handling
 *
 * Errors are detected at two points:
 * 1. `hipMemcpyAsync` launch failure — returns immediately.
 * 2. `hipStreamSynchronize` failure — detected after transfer
 *    completes (or fails asynchronously).
 *
 * Both use @ref mapError for consistent error mapping.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 *
 * @return @ref HIP_OK on success, or a @ref hipStatus_t with
 *         a non-zero code and a descriptive message.
 */
hipStatus_t hipTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                        const void *src, void *dst) {
  hipStatus_t status;
  hipStream_t stream = getStream(&status);

  if (!status) {
    return status;
  }

  const hipError_t err =
      hipMemcpyAsync(dst, src, bytes, mapMemcpyKind(kind), stream);

  if (err != hipSuccess) {
    return mapError(err);
  }

  const hipError_t syncErr = hipStreamSynchronize(stream);
  if (syncErr != hipSuccess) {
    return mapError(syncErr);
  }

  return HIP_OK;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
hipStatus_t hipTransfer(std::size_t, DeviceMemcpyKind, const void *,
                        void *) {
  return hipStatus_t{.code = -1, .msg = "HIP runtime headers not available"};
}

#endif
#endif /* NOVA_HAS_HIP */
