/**
 * @file CudaAllocator.cpp
 * @brief CUDA memory allocation, release, and resize implementation.
 *
 * @details
 * Implements the three core allocation primitives used by the
 * device-agnostic FFI layer (@c ffi.cpp).  All device-memory
 * operations use a temporary CUDA stream for async allocation
 * and synchronize before returning; errors are mapped through
 * @ref mapError from @ref common.hh.
 *
 * Compiled only behind @c NOVA_HAS_CUDA with
 * @c __has_include(<cuda_runtime_api.h>); otherwise stubs that
 * return @ref novaBackendNotCompiled are provided.
 *
 * @see CudaAllocator.hpp  Type declarations and function signatures.
 * @see CudaIO.cpp         CUDA data transfer implementation.
 * @see common.hh          Shared error-mapping and stream helpers.
 * @see ffi.cpp            Dispatch layer that calls into this file.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include <cstring>
#include <cuda_runtime_api.h>

#include "CudaAllocator.hpp"
#include "common.hh"

/**
 * @brief Allocate a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls @c cudaMallocHost.  For device memory,
 * calls @c cudaMallocAsync on @p stream when given (no sync, caller
 * owns ordering), or on a temporary synchronized stream when @p stream
 * is null.
 *
 * @param[in]  bytes  Requested size in bytes.
 * @param[in]  pinned If @c true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 * @param[in]  stream Caller stream for chaining, or null for the
 *                    synchronous internal path. Never destroyed here.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p bytes must be greater than zero.
 */
novaStatus_t cudaReserve(std::size_t bytes, bool pinned, cudaBuffer_t *out,
                         cudaStream_t stream) {
  if (out == nullptr) {
    return novaStatus_t{.err = novaInvalidPointer,
                        .message =
                            nova_get_error_msg(novaInvalidPointer, nullptr)};
  }
  novaStatus_t status = {};
  void *ptr = nullptr;

  if (pinned) {
    const cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
      status.err = mapError(err);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }
  } else {
    if (supportMemoryPool()) {
      cudaStream_t work = stream;
      bool ownStream = false;
      if (work == nullptr) {
        if (!streamCreate(&work, &status)) {
          return status;
        }
        ownStream = true;
      }

      const cudaError_t err = cudaMallocAsync(&ptr, bytes, work);
      if (err != cudaSuccess) {
        if (ownStream) {
          cudaStreamDestroy(work);
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      if (ownStream) {
        if (!streamSync(work, &status)) {
          cudaStreamDestroy(work);
          return status;
        }
        if (!streamDestroy(work, &status)) {
          return status;
        }
      }
    } else {
      /* If device do not support MemoryPools fallback to cudaMalloc.  Normally,
       * it shouldn't reach this part of the code  */
      const cudaError_t err = cudaMalloc(&ptr, bytes);
      if (err != cudaSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }
    }
  }
  out->ptr = ptr;
  out->bytes = bytes;
  out->isPinned = pinned;
  return CUDA_OK;
}

/**
 * @brief Free a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls @c cudaFreeHost.  For device memory,
 * calls @c cudaFreeAsync on @p stream when given (no sync, caller
 * owns ordering), or on a temporary synchronized stream when @p stream
 * is null.
 *
 * @param[in,out] buf    Buffer descriptor to free.  Must not be null.
 * @param[in]     stream Caller stream for chaining, or null for the
 *                       synchronous internal path. Never destroyed here.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @post On success, @p buf is zeroed.
 */
novaStatus_t cudaRelease(cudaBuffer_t *buf, cudaStream_t stream) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return novaStatus_t{.err = novaInvalidPointer,
                        .message =
                            nova_get_error_msg(novaInvalidPointer, nullptr)};
  }

  novaStatus_t status = {};

  if (buf->isPinned) {
    const cudaError_t err = cudaFreeHost(buf->ptr);
    if (err != cudaSuccess) {
      status.err = mapError(err);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }
  } else {
    if (supportMemoryPool()) {
      cudaStream_t work = stream;
      bool ownStream = false;
      if (work == nullptr) {
        if (!streamCreate(&work, &status)) {
          return status;
        }
        ownStream = true;
      }

      const cudaError_t err = cudaFreeAsync(buf->ptr, work);
      if (err != cudaSuccess) {
        if (ownStream) {
          cudaStreamDestroy(work);
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      if (ownStream) {
        if (!streamSync(work, &status)) {
          cudaStreamDestroy(work);
          return status;
        }
        if (!streamDestroy(work, &status)) {
          return status;
        }
      }
    } else {
      const cudaError_t err = cudaFree(buf->ptr);
      if (err != cudaSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }
    }
  }

  buf->ptr = nullptr;
  buf->bytes = 0;
  buf->isPinned = false;
  return CUDA_OK;
}

/**
 * @brief Resize a CUDA memory buffer.
 *
 * @details
 * Allocates a new buffer, copies @c min(old, new) bytes, then frees
 * the old buffer.  For pinned memory the copy uses
 * @c std::memcpy; for device memory the alloc, copy, and free all run
 * on @p stream when given (no sync), or on a temporary synchronized
 * stream when @p stream is null.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     stream    Caller stream for chaining, or null for
 *                          the synchronous internal path. Never
 *                          destroyed here.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @warning On failure the original buffer may be freed.
 */
novaStatus_t cudaResize(cudaBuffer_t *buf, std::size_t new_bytes,
                        cudaStream_t stream) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return novaStatus_t{.err = novaInvalidPointer,
                        .message =
                            nova_get_error_msg(novaInvalidPointer, nullptr)};
  }

  novaStatus_t status = {};
  const std::size_t copyBytes = buf->bytes < new_bytes ? buf->bytes : new_bytes;
  void *newPtr = nullptr;

  if (buf->isPinned) {
    const cudaError_t err = cudaMallocHost(&newPtr, new_bytes);
    if (err != cudaSuccess) {
      status.err = mapError(err);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }

    std::memcpy(newPtr, buf->ptr, copyBytes);

    const cudaError_t freeErr = cudaFreeHost(buf->ptr);
    if (freeErr != cudaSuccess) {
      cudaFreeHost(newPtr);
      status.err = mapError(freeErr);
      status.message = nova_get_error_msg(status.err, nullptr);
      return status;
    }
  } else {
    if (supportMemoryPool()) {
      cudaStream_t work = stream;
      bool ownStream = false;
      if (work == nullptr) {
        if (!streamCreate(&work, &status)) {
          return status;
        }
        ownStream = true;
      }

      cudaError_t err = cudaMallocAsync(&newPtr, new_bytes, work);
      if (err != cudaSuccess) {
        if (ownStream && !streamDestroy(work, &status)) {
          return status;
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = cudaMemcpyAsync(newPtr, buf->ptr, copyBytes,
                            cudaMemcpyDeviceToDevice, work);
      if (err != cudaSuccess) {
        const cudaError_t freeAsyncErr = cudaFreeAsync(newPtr, work);
        if (freeAsyncErr != cudaSuccess) {
          if (ownStream) {
            streamSync(work, &status);
            streamDestroy(work, &status);
          }
          status.err = mapError(freeAsyncErr);
          status.message = nova_get_error_msg(status.err, nullptr);
          return status;
        }
        if (ownStream) {
          if (!streamSync(work, &status)) {
            streamDestroy(work, &status);
            return status;
          }
          if (!streamDestroy(work, &status)) {
            return status;
          }
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = cudaFreeAsync(buf->ptr, work);
      if (err != cudaSuccess) {
        const cudaError_t freeAsyncErr = cudaFreeAsync(newPtr, work);

        if (freeAsyncErr != cudaSuccess) {
          status.err = mapError(freeAsyncErr);
          status.message = nova_get_error_msg(status.err, nullptr);
          return status;
        }
        if (ownStream) {
          if (!streamSync(work, &status)) {
            cudaStreamDestroy(work);
            return status;
          }
          if (!streamDestroy(work, &status)) {
            return status;
          }
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      if (ownStream) {
        if (!streamSync(work, &status)) {
          cudaStreamDestroy(work);
          return status;
        }
        if (!streamDestroy(work, &status)) {
          return status;
        }
      }
    } else {
      cudaError_t err = cudaMalloc(&newPtr, new_bytes);
      if (err != cudaSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = cudaMemcpy(newPtr, buf->ptr, copyBytes, cudaMemcpyDeviceToDevice);

      if (err != cudaSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = cudaFree(buf->ptr);

      if (err != cudaSuccess) {
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }
    }
  }

  buf->ptr = newPtr;
  buf->bytes = new_bytes;
  return CUDA_OK;
}

#else // !__has_include(<cuda_runtime_api.h>)

/** @brief Stub: CUDA runtime headers not available. */
novaStatus_t cudaReserve(std::size_t, bool, cudaBuffer_t *, cudaStream_t) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

/** @brief Stub: CUDA runtime headers not available. */
novaStatus_t cudaRelease(cudaBuffer_t *, cudaStream_t) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

/** @brief Stub: CUDA runtime headers not available. */
novaStatus_t cudaResize(cudaBuffer_t *, std::size_t, cudaStream_t) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

#endif
#endif /* NOVA_HAS_CUDA */
