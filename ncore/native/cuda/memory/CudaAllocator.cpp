/**
 * @file CudaAllocator.cpp
 * @brief CUDA memory allocation, release, and resize implementation.
 *
 * @details
 * Implements the three core allocation primitives used by the
 * device-agnostic FFI layer (@c ffi.cpp).  All device-memory
 * operations use a temporary CUDA stream for async allocation
 * and synchronize before returning.
 *
 * The file is conditionally compiled behind @c NOVA_HAS_CUDA and
 * @c __has_include(<cuda_runtime_api.h>).  When CUDA headers are
 * unavailable (e.g., during linting), stub functions that return
 * an error status are provided.
 *
 * @section architecture Architecture
 *
 * Internal helpers (within anonymous namespace):
 * @li @ref mapError — maps any @c cudaError_t to a @ref novaError_t.
 * @li @ref streamCreate — creates a temporary CUDA stream.
 * @li @ref streamSync — blocks until stream work completes.
 * @li @ref streamDestroy — destroys a CUDA stream.
 *
 * @section error-mapping Error Mapping
 *
 * All CUDA errors are mapped via @ref mapError:
 * @li @c cudaSuccess → @ref novaSuccess
 * @li @c cudaErrorInvalidValue → @ref novaInvalidValue
 * @li @c cudaErrorMemoryAllocation → @ref novaOutOfMemory
 * @li @c cudaErrorNotSupported → @ref novaNotImplemented
 * @li @c cudaErrorExternalDevice → @ref novaExternalDeviceError
 * @li @c cudaErrorInvalidResourceHandle → @ref novaInvalidResourceHandle
 * @li Any other error → @ref novaNotImplemented
 *
 * @see CudaAllocator.hpp  Type declarations and function signatures.
 * @see CudaIO.cpp         CUDA data transfer implementation.
 * @see ffi.cpp            Dispatch layer that calls into this file.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include <cstring>
#include <cuda_runtime_api.h>

#include "../DetectCudaDevice.hpp"
#include "CudaAllocator.hpp"

namespace {

/**
 * @brief Map a CUDA error code to a @ref novaError_t.
 *
 * @details
 * Converts @c cudaError_t values into the project-standard
 * @ref novaError_t enumeration.  Covers errors from allocation
 * (@c cudaMallocAsync, @c cudaMallocHost), deallocation
 * (@c cudaFreeAsync, @c cudaFreeHost), stream operations, and
 * memcpy.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return The corresponding Nova error category.
 */
novaError_t mapError(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return novaSuccess;
  case cudaErrorInvalidValue:
    return novaInvalidValue;
  case cudaErrorMemoryAllocation:
    return novaOutOfMemory;
  case cudaErrorNotSupported:
    return novaNotImplemented;
  case cudaErrorExternalDevice:
    return novaExternalDeviceError;
  case cudaErrorInvalidResourceHandle:
    return novaInvalidResourceHandle;
  default:
    return novaNotImplemented;
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
 */
bool supportMemoryPool() {
  static int supported = 0;

  cudaError_t err = cudaDeviceGetAttribute(
      &supported, cudaDevAttrMemoryPoolsSupported, getCudaDeviceId());

  return err != cudaSuccess ? false : bool(supported);
}

/**
 * @brief Create a CUDA stream.
 *
 * @param[out] stream  Receives the new stream handle.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
bool streamCreate(cudaStream_t *stream, novaStatus_t *status) {
  const cudaError_t err = cudaStreamCreate(stream);
  if (err != cudaSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

/**
 * @brief Block until all work on @p stream has completed.
 *
 * @param[in]  stream  The stream to synchronize.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
bool streamSync(cudaStream_t stream, novaStatus_t *status) {
  const cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

/**
 * @brief Destroy a CUDA stream.
 *
 * @param[in]  stream  The stream to destroy.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
bool streamDestroy(cudaStream_t stream, novaStatus_t *status) {
  const cudaError_t err = cudaStreamDestroy(stream);
  if (err != cudaSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

} // namespace

/**
 * @brief Allocate a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls @c cudaMallocHost.  For device memory,
 * creates a temporary stream, calls @c cudaMallocAsync,
 * synchronizes, and destroys the stream.
 *
 * @param[in]  bytes  Requested size in bytes.
 * @param[in]  pinned If @c true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to valid CUDA memory.
 */
novaStatus_t cudaReserve(std::size_t bytes, bool pinned, cudaBuffer_t *out) {
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
      cudaStream_t stream = nullptr;
      if (!streamCreate(&stream, &status)) {
        return status;
      }

      const cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
      if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      if (!streamSync(stream, &status)) {
        cudaStreamDestroy(stream);
        return status;
      }
      if (!streamDestroy(stream, &status)) {
        return status;
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
 * creates a temporary stream, calls @c cudaFreeAsync, synchronizes,
 * and destroys the stream.  On success, the buffer is zeroed.
 *
 * @param[in,out] buf  Buffer descriptor to free.  Must not be null.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @post On success, @p buf is zeroed.
 */
novaStatus_t cudaRelease(cudaBuffer_t *buf) {
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
      cudaStream_t stream = nullptr;
      if (!streamCreate(&stream, &status)) {
        return status;
      }

      const cudaError_t err = cudaFreeAsync(buf->ptr, stream);
      if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      if (!streamSync(stream, &status)) {
        cudaStreamDestroy(stream);
        return status;
      }
      if (!streamDestroy(stream, &status)) {
        return status;
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
 * @c std::memcpy; for device memory it uses @c cudaMemcpyAsync on a
 * temporary stream.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @post On success, @p buf->ptr and @p buf->bytes are updated.
 *
 * @warning On failure the original buffer may be freed.
 */
novaStatus_t cudaResize(cudaBuffer_t *buf, std::size_t new_bytes) {
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
      cudaStream_t stream = nullptr;
      if (!streamCreate(&stream, &status)) {
        return status;
      }

      cudaError_t err = cudaMallocAsync(&newPtr, new_bytes, stream);
      if (err != cudaSuccess) {
        if (!streamDestroy(stream, &status)) {
          return status;
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = cudaMemcpyAsync(newPtr, buf->ptr, copyBytes,
                            cudaMemcpyDeviceToDevice, stream);
      if (err != cudaSuccess) {
        const cudaError_t freeAsyncErr = cudaFreeAsync(newPtr, stream);
        if (freeAsyncErr != cudaSuccess) {
          streamSync(stream, &status);
          streamDestroy(stream, &status);
          status.err = mapError(freeAsyncErr);
          status.message = nova_get_error_msg(status.err, nullptr);
          return status;
        }
        if (!streamSync(stream, &status)) {
          streamDestroy(stream, &status);
          return status;
        }
        if (!streamDestroy(stream, &status)) {
          return status;
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      err = cudaFreeAsync(buf->ptr, stream);
      if (err != cudaSuccess) {
        const cudaError_t freeAsyncErr = cudaFreeAsync(newPtr, stream);

        if (freeAsyncErr != cudaSuccess) {
          status.err = mapError(freeAsyncErr);
          status.message = nova_get_error_msg(status.err, nullptr);
          return status;
        }
        if (!streamSync(stream, &status)) {
          cudaStreamDestroy(stream);
          return status;
        }
        if (!streamDestroy(stream, &status)) {
          return status;
        }
        status.err = mapError(err);
        status.message = nova_get_error_msg(status.err, nullptr);
        return status;
      }

      if (!streamSync(stream, &status)) {
        cudaStreamDestroy(stream);
        return status;
      }
      if (!streamDestroy(stream, &status)) {
        return status;
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
novaStatus_t cudaReserve(std::size_t, bool, cudaBuffer_t *) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

/** @brief Stub: CUDA runtime headers not available. */
novaStatus_t cudaRelease(cudaBuffer_t *) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

/** @brief Stub: CUDA runtime headers not available. */
novaStatus_t cudaResize(cudaBuffer_t *, std::size_t) {
  return novaStatus_t{.err = novaBackendNotCompiled,
                      .message =
                          nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}

#endif
#endif /* NOVA_HAS_CUDA */
