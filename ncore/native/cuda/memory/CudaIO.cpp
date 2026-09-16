/**
 * @file CudaIO.cpp
 * @brief CUDA data transfer implementation.
 *
 * @details
 * Implements the master memcpy dispatcher @ref cudaTransfer using
 * @c cudaMemcpyAsync on a reusable CUDA stream for all copy
 * directions.  The stream is created once and lives for the
 * lifetime of the process (singleton pattern).
 *
 * The file is conditionally compiled behind @c NOVA_HAS_CUDA and
 * @c __has_include(<cuda_runtime_api.h>).  When CUDA headers are
 * unavailable, a stub function returning an error status is
 * provided.
 *
 * @section architecture Architecture
 *
 * The module exposes a single public function (@ref cudaTransfer)
 * that handles H2D, D2H, and D2D transfers.  Internally it uses:
 * @li @ref mapError — maps @c cudaError_t to @ref novaStatus_t.
 * @li @ref mapMemcpyKind — converts @ref DeviceMemcpyKind to
 *   @c cudaMemcpyKind.
 * @li @ref getStream — returns a singleton CUDA stream.
 *
 * @section error-mapping Error Mapping
 *
 * All CUDA errors are mapped to a @ref novaStatus_t via a single
 * @ref mapError function: @c cudaSuccess maps to @ref novaSuccess,
 * @c cudaErrorInvalidValue to @ref novaInvalidValue,
 * @c cudaErrorExternalDevice to @ref novaExternalDeviceError,
 * @c cudaErrorInvalidMemcpyDirection to @ref novaInvalidTransfDirection,
 * @c cudaErrorInvalidResourceHandle to @ref novaInvalidResourceHandle,
 * and every other error to @ref novaNotImplemented.
 *
 * @see CudaIO.hpp        Function declaration.
 * @see CudaAllocator.cpp CUDA memory allocation implementation.
 * @see ffi.cpp           Dispatch layer that calls into this file.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>

#include <mutex>

#include "../DetectCudaDevice.hpp"
#include "CudaAllocator.hpp"
#include "CudaIO.hpp"
namespace {

/**
 * @brief Map a CUDA runtime error to a @ref novaStatus_t.
 *
 * @details
 * Converts @c cudaError_t codes returned by @c cudaMemcpyAsync
 * and @c cudaStreamSynchronize into the project-standard
 * @ref novaStatus_t format.  Each error code is mapped to a
 * unique @ref novaError_t category, and the human-readable
 * error string is selected from the Nova status table via
 * @ref nova_get_error_msg.
 *
 * The mapped codes are: @c cudaSuccess → @ref novaSuccess,
 * @c cudaErrorInvalidValue → @ref novaInvalidValue,
 * @c cudaErrorExternalDevice → @ref novaExternalDeviceError,
 * @c cudaErrorInvalidMemcpyDirection → @ref novaInvalidTransfDirection,
 * @c cudaErrorInvalidResourceHandle → @ref novaInvalidResourceHandle,
 * and every other error → @ref novaNotImplemented.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return @ref novaStatus_t with the mapped @ref novaError_t code and
 *         the message string obtained from @ref nova_get_error_msg.
 */
novaStatus_t mapError(cudaError_t err) {
  novaStatus_t status = {};
  switch (err) {
  case cudaSuccess:
    status.err = novaSuccess;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  case cudaErrorInvalidValue:
    status.err = novaInvalidValue;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  case cudaErrorExternalDevice:
    status.err = novaExternalDeviceError;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  case cudaErrorInvalidMemcpyDirection:
    status.err = novaInvalidTransfDirection;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  case cudaErrorInvalidResourceHandle:
    status.err = novaInvalidResourceHandle;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  default:
    status.err = novaNotImplemented;
    status.message = nova_get_error_msg(status.err, nullptr);
    return status;
  }
}

/**
 * @brief Query whether the active CUDA device supports memory pools.
 *
 * @details
 * Uses @c getCudaDeviceId() to query the
 * @c cudaDevAttrMemoryPoolsSupported attribute.  This is safe
 * because CUDA device detection is performed before any memory is
 * allocated on the device; if detection failed, the internal
 * implementations saved the result under lock, ensuring that
 * @c getCudaDeviceId() always returns a valid value.
 *
 * @return @c true if memory pools are supported, @c false otherwise.
 *         Queried once; function-local statics initialize exactly once
 *         even under concurrent first calls.
 */
bool supportMemoryPool() {
  static const bool supported = [] {
    int value = 0;
    const cudaError_t err = cudaDeviceGetAttribute(
        &value, cudaDevAttrMemoryPoolsSupported, getCudaDeviceId());
    return err == cudaSuccess && value != 0;
  }();
  return supported;
}

/**
 * @brief Convert @ref DeviceMemcpyKind to @c cudaMemcpyKind.
 *
 * @details
 * Maps the backend-agnostic @ref DeviceMemcpyKind enum to the
 * CUDA-specific @c cudaMemcpyKind enum used by @c cudaMemcpyAsync.
 * The mapping is: @c deviceMemcpyHostToDevice →
 * @c cudaMemcpyHostToDevice, @c deviceMemcpyDeviceToHost →
 * @c cudaMemcpyDeviceToHost, @c deviceMemcpyDeviceToDevice →
 * @c cudaMemcpyDeviceToDevice, default → @c cudaMemcpyDefault.
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
 * Returns a singleton CUDA stream created exactly once via
 * @c std::call_once and reused for all subsequent default-path
 * @ref cudaTransfer operations.  The stream is created with default
 * flags (non-blocking, no priority override).
 *
 * The stream is never explicitly destroyed.  The CUDA runtime
 * reclaims all resources on process exit.  This avoids the
 * overhead of create/destroy per transfer and eliminates the
 * risk of use-after-free in concurrent scenarios.
 *
 * @param[out] status  Receives an error status if stream creation
 *                     fails.  Unchanged on success.
 *
 * @return The singleton @c cudaStream_t, or null on failure.
 *
 * @warning A failed creation is sticky: @c std::call_once never
 *          retries, so every later call also fails until process
 *          restart.  Stream creation failure implies a broken device.
 */
cudaStream_t getStream(novaStatus_t *status) {
  static cudaStream_t stream = nullptr;
  static std::once_flag flag;
  static novaStatus_t initStatus = {};
  std::call_once(flag, [] {
    const cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
      initStatus = mapError(err);
    }
  });
  if (stream == nullptr) {
    *status = initStatus;
  }
  return stream;
}

} // namespace

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using @c cudaMemcpyAsync on @p stream
 * when given (no synchronization), or on the singleton stream with
 * synchronization before returning when @p stream is null.  The
 * transfer direction is determined by @p kind.
 *
 * @subsection execution-flow Execution Flow
 *
 * @code{.cpp}
 *   work = (stream != nullptr) ? stream : getStream();  // select
 *   err = cudaMemcpyAsync(dst, src,                     // enqueue
 *                         bytes, kind, work);
 *   if (err != cudaSuccess) return mapError(err);
 *   if (stream == nullptr) {                            // sync iff internal
 *     syncErr = cudaStreamSynchronize(work);            // block
 *     if (sync_err != cudaSuccess) return mapError(syncErr);
 *   }
 *   return CUDA_OK;
 * @endcode
 *
 * Both use @ref mapError for consistent error mapping.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 * @param[in]  stream    Caller stream for chaining, or null for the
 *                       synchronous singleton path. Never destroyed here.
 *
 * @return @ref CUDA_OK on success, or a @ref novaStatus_t with
 *         a non-zero code and a descriptive message.
 */
novaStatus_t cudaTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                          const void *src, void *dst, cudaStream_t stream) {
  novaStatus_t status = {};
  cudaStream_t work = stream;
  if (work == nullptr) {
    work = getStream(&status);
    if (status.err != novaSuccess) {
      return status;
    }
  }

  if (supportMemoryPool()) {
    const cudaError_t err =
        cudaMemcpyAsync(dst, src, bytes, mapMemcpyKind(kind), work);

    if (err != cudaSuccess) {
      return mapError(err);
    }

    if (stream == nullptr) {
      const cudaError_t syncErr = cudaStreamSynchronize(work);
      if (syncErr != cudaSuccess) {
        return mapError(syncErr);
      }
    }
  } else {
    const cudaError_t err = cudaMemcpy(dst, src, bytes, mapMemcpyKind(kind));

    if (err != cudaSuccess) {
      return mapError(err);
    }
  }

  return CUDA_OK;
}

#else // !__has_include(<cuda_runtime_api.h>)

/** @brief Stub: CUDA runtime headers not available. */
novaStatus_t cudaTransfer(std::size_t, DeviceMemcpyKind, const void *, void *,
                          cudaStream_t) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

#endif
#endif /* NOVA_HAS_CUDA */
