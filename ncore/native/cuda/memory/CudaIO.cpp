/**
 * @file CudaIO.cpp
 * @brief CUDA data transfer implementation.
 *
 * @details
 * Implements the master memcpy dispatcher @ref cudaTransfer using
 * @c cudaMemcpyAsync.  A caller stream enqueues with no
 * synchronization; the null path resolves the device engine stream
 * and drains it before returning.
 *
 * Compiled only behind @c NOVA_HAS_CUDA and
 * @c __has_include(<cuda_runtime_api.h>); otherwise a stub
 * returning an error status is provided.
 *
 * @see CudaIO.hpp        Function declaration.
 * @see CudaAllocator.cpp CUDA memory allocation implementation.
 * @see common.hh         Shared error-mapping and stream helpers.
 * @see ffi.cpp           Dispatch layer that calls into this file.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>

#include "CudaAllocator.hpp"
#include "CudaEngineStream.hpp"
#include "CudaIO.hpp"
#include "common.hh"

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using @c cudaMemcpyAsync on @p stream
 * when given (no synchronization), or on the device engine stream with
 * synchronization before returning when @p stream is null.  The
 * transfer direction is determined by @p kind.
 *
 * Errors surface at the @c cudaMemcpyAsync launch and, on the engine
 * path, at the @c cudaStreamSynchronize wait; both map through
 * @ref mapError2Status.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 * @param[in]  stream    Caller stream for chaining, or null to select
 *                       the device engine stream. Never destroyed here.
 *
 * @return @ref CUDA_OK on success, or a @ref novaStatus_t with
 *         a non-zero code and a descriptive message.
 */
novaStatus_t cudaTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                          const void *src, void *dst, cudaStream_t stream) {
  cudaStream_t work = stream;
  if (work == nullptr) {
    const novaStatus_t engineStatus = CudaEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }

  const cudaError_t err =
      cudaMemcpyAsync(dst, src, bytes, mapMemcpyKind(kind), work);
  if (err != cudaSuccess) {
    return mapError2Status(err);
  }
  if (stream == nullptr) {
    const cudaError_t syncErr = cudaStreamSynchronize(work);
    if (syncErr != cudaSuccess) {
      return mapError2Status(syncErr);
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
