/**
 * @file CudaIO.cpp
 * @brief CUDA data transfer implementation.
 *
 * @details
 * Implements the master memcpy dispatcher @ref cudaTransfer using
 * @c cudaMemcpyAsync.  A caller stream enqueues with no
 * synchronization; the synchronous null path runs on the singleton
 * stream, created once for the lifetime of the process.
 *
 * Compiled only behind @c NOVA_HAS_CUDA and
 * @c __has_include(<cuda_runtime_api.h>); otherwise a stub
 * returning an error status is provided.
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
 * Errors surface at the @c cudaMemcpyAsync launch and, on the
 * singleton path, at the @c cudaStreamSynchronize wait; both map
 * through @ref mapError.
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
