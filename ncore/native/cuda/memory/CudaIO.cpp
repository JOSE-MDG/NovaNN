/**
 * @file CudaIO.cpp
 * @brief CUDA data transfer implementation.
 *
 * @details
 * Implements the master memcpy dispatcher @ref cudaTransfer using
 * `cudaMemcpyAsync` on a reusable CUDA stream for all copy
 * directions.  The stream is created once and lives for the
 * lifetime of the process (singleton pattern).
 *
 * The file is conditionally compiled behind `NOVA_HAS_CUDA` and
 * `__has_include(<cuda_runtime_api.h>)`.  When CUDA headers are
 * unavailable, a stub function returning an error status is
 * provided.
 *
 * ## Architecture
 *
 * The module exposes a single public function (@ref cudaTransfer)
 * that handles H2D, D2H, and D2D transfers.  Internally it uses:
 * - @ref mapError — maps `cudaError_t` to @ref cudaStatus_t.
 * - @ref mapMemcpyKind — converts @ref DeviceMemcpyKind to
 *   `cudaMemcpyKind`.
 * - @ref getStream — returns a singleton CUDA stream.
 *
 * ## Error Mapping
 *
 * All CUDA errors are mapped to a @ref cudaStatus_t via a single
 * @ref mapError function that covers `cudaSuccess` (code 0),
 * `cudaErrorInvalidValue` (code 1), `cudaErrorInvalidMemcpyDirection`
 * (code 2), `cudaErrorInvalidResourceHandle` (code 3), and all
 * others (code -1).
 *
 * @see CudaIO.hpp        Function declaration.
 * @see CudaAllocator.cpp CUDA memory allocation implementation.
 * @see ffi.cpp           Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include "CudaIO.hpp"
#include <cuda_runtime_api.h>

namespace {

/**
 * @brief Map a CUDA runtime error to a @ref cudaStatus_t.
 *
 * @details
 * Converts `cudaError_t` codes returned by `cudaMemcpyAsync`
 * and `cudaStreamSynchronize` into the project-standard
 * @ref cudaStatus_t format.  Each error code is mapped to a
 * unique integer for programmatic handling, and the human-readable
 * error string is obtained via `cudaGetErrorString`.
 *
 * The mapped codes are: `cudaSuccess` → 0,
 * `cudaErrorInvalidValue` → 1, `cudaErrorInvalidMemcpyDirection`
 * → 2, `cudaErrorInvalidResourceHandle` → 3, everything else
 * → -1.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return @ref cudaStatus_t with the mapped code and message.
 *
 * @post  On success, `status.code == 0` and `status.msg == "ok"`.
 * @post  On failure, `status.code != 0` and `status.msg` contains
 *        the error string from `cudaGetErrorString`.
 */
cudaStatus_t mapError(cudaError_t err) {
  cudaStatus_t status = {};
  switch (err) {
  case cudaSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case cudaErrorInvalidValue:
    status.code = 1;
    status.msg = cudaGetErrorString(err);
    return status;
  case cudaErrorInvalidMemcpyDirection:
    status.code = 2;
    status.msg = cudaGetErrorString(err);
    return status;
  case cudaErrorInvalidResourceHandle:
    status.code = 3;
    status.msg = cudaGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = cudaGetErrorString(err);
    return status;
  }
}

/**
 * @brief Convert @ref DeviceMemcpyKind to `cudaMemcpyKind`.
 *
 * @details
 * Maps the backend-agnostic @ref DeviceMemcpyKind enum to the
 * CUDA-specific `cudaMemcpyKind` enum used by `cudaMemcpyAsync`.
 * The mapping is: `deviceMemcpyHostToDevice` →
 * `cudaMemcpyHostToDevice`, `deviceMemcpyDeviceToHost` →
 * `cudaMemcpyDeviceToHost`, `deviceMemcpyDeviceToDevice` →
 * `cudaMemcpyDeviceToDevice`, default → `cudaMemcpyDefault`.
 *
 * @param[in] kind  The device-agnostic copy direction.
 *
 * @return The corresponding CUDA memcpy kind.
 */
cudaMemcpyKind mapMemcpyKind(DeviceMemcpyKind kind) {
  switch (kind) {
  case DeviceMemcpyKind::deviceMemcpyHostToDevice:
    return cudaMemcpyHostToDevice;
  case DeviceMemcpyKind::deviceMemcpyDeviceToHost:
    return cudaMemcpyDeviceToHost;
  case DeviceMemcpyKind::deviceMemcpyDeviceToDevice:
    return cudaMemcpyDeviceToDevice;
  default:
    return cudaMemcpyDefault;
  }
}

/**
 * @brief Get or create the reusable CUDA stream.
 *
 * @details
 * Returns a singleton CUDA stream that is created on first call
 * and reused for all subsequent @ref cudaTransfer operations.  The
 * stream is created with default flags (non-blocking, no
 * priority override).
 *
 * The stream is never explicitly destroyed.  The CUDA runtime
 * reclaims all resources on process exit.  This avoids the
 * overhead of create/destroy per transfer and eliminates the
 * risk of use-after-free in concurrent scenarios.
 *
 * The `static` local variable is initialised exactly once, even
 * under concurrent access (C++11 guarantee).  The
 * `cudaStreamCreate` call is serialised by the C++ runtime.
 *
 * @return The singleton `cudaStream_t`.
 */
cudaStream_t getStream() {
  static cudaStream_t stream = nullptr;
  if (stream == nullptr) {
    cudaStreamCreate(&stream);
  }
  return stream;
}

} // namespace

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using `cudaMemcpyAsync` on a
 * reusable internal CUDA stream, then synchronises the stream
 * before returning.  The transfer direction is determined by
 * @p kind.
 *
 * ### Execution Flow
 *
 * @code{.cpp}
 *   stream = getStream();              // singleton stream
 *   err = cudaMemcpyAsync(dst, src,     // enqueue transfer
 *                         bytes, kind,
 *                         stream);
 *   if (err != cudaSuccess) return mapError(err);
 *   syncErr = cudaStreamSynchronize(stream);  // block
 *   if (sync_err != cudaSuccess) return mapError(syncErr);
 *   return CUDA_OK;
 * @endcode
 *
 * ### Error Handling
 *
 * Errors are detected at two points:
 * 1. `cudaMemcpyAsync` launch failure — returns immediately.
 * 2. `cudaStreamSynchronize` failure — detected after transfer
 *    completes (or fails asynchronously).
 *
 * Both use @ref mapError for consistent error mapping.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 *
 * @return @ref CUDA_OK on success, or a @ref cudaStatus_t with
 *         a non-zero code and a descriptive message.
 */
cudaStatus_t cudaTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                          const void *src, void *dst) {
  cudaStream_t stream = getStream();

  const cudaError_t err =
      cudaMemcpyAsync(dst, src, bytes, mapMemcpyKind(kind), stream);

  if (err != cudaSuccess) {
    return mapError(err);
  }

  const cudaError_t syncErr = cudaStreamSynchronize(stream);
  if (syncErr != cudaSuccess) {
    return mapError(syncErr);
  }

  return CUDA_OK;
}

#else // !__has_include(<cuda_runtime_api.h>)

/** @brief Stub: CUDA runtime headers not available. */
cudaStatus_t cudaTransfer(std::size_t, DeviceMemcpyKind, const void *,
                          void *) {
  return cudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available\n"};
}

#endif
#endif /* NOVA_HAS_CUDA */
