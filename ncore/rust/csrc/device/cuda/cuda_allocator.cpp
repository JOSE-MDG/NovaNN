/**
 * @file cuda_allocator.cpp
 * @brief Backend implementation of CUDA device/pinned-host memory allocation.
 *
 * Implements cuda_reserve, cuda_resize and cuda_release by wrapping the
 * CUDA Runtime API (cudaMallocAsync / cudaMallocHost / cudaFreeAsync /
 * cudaFreeHost / cudaMemcpyAsync) with alignment rounding, descriptive
 * error reporting, and a mapping from CUDA runtime error codes to
 * application-level status codes.
 *
 * Device-memory operations are all issued onto a temporary per-call stream
 * so that they can be awaited before the stream is torn down.  Pinned-host
 * operations are fully synchronous (cudaMallocHost / cudaFreeHost / memcpy)
 * and therefore never require a stream.
 *
 * cuda_resize allocates a new buffer of the requested size, copies the
 * minimum of the old and new sizes into it, frees the old buffer, and
 * updates the descriptor — cleaning up all intermediate resources on any
 * failure so that the original descriptor is always left untouched.
 */

#include "cuda_allocator.hpp"
#include <cstring>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>

/**
 * @brief Map a CUDA runtime error to an application-level status code.
 *
 * @param err CUDA error value returned by the runtime API.
 * @return 0  (cudaSuccess),
 *         1  (cudaErrorInvalidValue),
 *         2  (cudaErrorMemoryAllocation),
 *         3  (cudaErrorNotSupported),
 *        -1  (any other error).
 */
static int map_cuda_error(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return 0;
  case cudaErrorInvalidValue:
    return 1;
  case cudaErrorMemoryAllocation:
    return 2;
  case cudaErrorNotSupported:
    return 3;
  default:
    return -1;
  }
}

/**
 * @brief Map a CUDA stream-creation error to an application-level code.
 *
 * @param err CUDA error value from cudaStreamCreate.
 * @return 0  (cudaSuccess),
 *         1  (cudaErrorInvalidValue),
 *         2  (cudaErrorExternalDevice),
 *        -1  (any other error).
 */
static int map_cuda_stream_error(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return 0;
  case cudaErrorInvalidValue:
    return 1;
  case cudaErrorExternalDevice:
    return 2;
  default:
    return -1;
  }
}

/**
 * @brief Map a CUDA stream-destroy error to an application-level code.
 *
 * @param err CUDA error value from cudaStreamDestroy.
 * @return 0  (cudaSuccess),
 *         1  (cudaErrorInvalidValue),
 *         3  (cudaErrorInvalidResourceHandle),
 *         4  (cudaErrorExternalDevice),
 *        -1  (any other error).
 */
static int map_cuda_destroy_error(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return 0;
  case cudaErrorInvalidValue:
    return 1;
  case cudaErrorInvalidResourceHandle:
    return 3;
  case cudaErrorExternalDevice:
    return 4;
  default:
    return -1;
  }
}

/**
 * @brief Map a CUDA stream-synchronise error to an application-level code.
 *
 * @param err CUDA error value from cudaStreamSynchronize.
 * @return 0  (cudaSuccess),
 *         1  (cudaErrorInvalidResourceHandle),
 *        -1  (any other error).
 */
static int map_cuda_sync_error(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return 0;
  case cudaErrorInvalidResourceHandle:
    return 1;
  default:
    return -1;
  }
}

/**
 * @brief Round @p bytes up to the next multiple of @p align.
 *
 * @param bytes Size to round.
 * @param align Alignment boundary (must be a power of two).
 * @return The smallest multiple of @p align that is >= @p bytes.
 */
static constexpr std::size_t align_up(std::size_t bytes,
                                      std::size_t align) noexcept {
  return (bytes + (align - 1)) & ~(align - 1);
}

/**
 * @brief Create a CUDA stream, populating @p status on failure.
 *
 * @param[out] stream  Receives the new stream handle on success.
 * @param[out] status  Populated with a non-zero code and message on failure.
 * @return true on success, false on failure.
 */
static bool stream_create(cudaStream_t *stream, CudaStatus_t *status) {
  const cudaError_t err = cudaStreamCreate(stream);
  if (err != cudaSuccess) {
    status->code = map_cuda_stream_error(err);
    status->msg = cudaGetErrorString(err);
    return false;
  }
  return true;
}

/**
 * @brief Synchronise @p stream, populating @p status on failure.
 *
 * @param stream  Stream to synchronise.
 * @param[out] status  Populated with a non-zero code and message on failure.
 * @return true on success, false on failure.
 */
static bool stream_sync(cudaStream_t stream, CudaStatus_t *status) {
  const cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    status->code = map_cuda_sync_error(err);
    status->msg = cudaGetErrorString(err);
    return false;
  }
  return true;
}

/**
 * @brief Destroy @p stream, populating @p status on failure.
 *
 * @param stream       Stream to destroy.
 * @param[out] status  Populated with a non-zero code and message on failure.
 * @return true on success, false on failure.
 */
static bool stream_destroy(cudaStream_t stream, CudaStatus_t *status) {
  const cudaError_t err = cudaStreamDestroy(stream);
  if (err != cudaSuccess) {
    status->code = map_cuda_destroy_error(err);
    status->msg = cudaGetErrorString(err);
    return false;
  }
  return true;
}

/**
 * @brief Allocate a CUDA buffer with optional alignment.
 *
 * Uses cudaMallocAsync on a temporary stream for device memory, or
 * cudaMallocHost for pinned host memory.  The stream is created, used,
 * synchronised and destroyed within the call.
 *
 * @param bytes  Minimum number of bytes to allocate.
 * @param align  Alignment requirement (power of two, or 1 for default).
 * @param pinned If true, allocate page-locked host memory; otherwise
 *               allocate device memory.
 * @param out    Output buffer descriptor (valid only when code == 0).
 * @return CUDA_OK on success, or a CudaStatus_t with a positive error
 *         code and a descriptive message on failure.
 */
CudaStatus_t cuda_reserve(std::size_t bytes, std::size_t align, bool pinned,
                          CudaBuffer_t *out) {
  CudaStatus_t status = {};
  const std::size_t alloc_bytes = (align > 1) ? align_up(bytes, align) : bytes;
  void *ptr = nullptr;

  if (pinned) {
    const cudaError_t err = cudaMallocHost(&ptr, alloc_bytes);
    if (err != cudaSuccess) {
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }
  } else {
    cudaStream_t stream = nullptr;
    if (!stream_create(&stream, &status)) {
      return status;
    }

    const cudaError_t err = cudaMallocAsync(&ptr, alloc_bytes, stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    if (!stream_sync(stream, &status)) {
      cudaStreamDestroy(stream);
      return status;
    }
    if (!stream_destroy(stream, &status)) {
      return status;
    }
  }

  out->ptr = ptr;
  out->bytes = alloc_bytes;
  out->is_pinned = pinned;
  return CUDA_OK;
}

/**
 * @brief Free a CUDA buffer previously allocated with cuda_reserve().
 *
 * Uses cudaFreeAsync on a temporary stream for device memory, or
 * cudaFreeHost for pinned host memory.  On success the descriptor is
 * zeroed so that a subsequent release is a safe no-op.
 *
 * @param buf Pointer to the buffer descriptor to free.  If @p buf or
 *            @p buf->ptr is NULL the function returns an error status
 *            without calling any CUDA API.
 * @return CUDA_OK on success, or a CudaStatus_t with a positive error
 *         code and a descriptive message on failure.
 */
CudaStatus_t cuda_release(CudaBuffer_t *buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return CudaStatus_t{.code = 1,
                        .msg = "cuda_release: buf or buf->ptr is null"
                               " — nothing to free"};
  }

  CudaStatus_t status = {};

  if (buf->is_pinned) {
    const cudaError_t err = cudaFreeHost(buf->ptr);
    if (err != cudaSuccess) {
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }
  } else {
    cudaStream_t stream = nullptr;
    if (!stream_create(&stream, &status)) {
      return status;
    }

    const cudaError_t err = cudaFreeAsync(buf->ptr, stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    if (!stream_sync(stream, &status)) {
      cudaStreamDestroy(stream);
      return status;
    }
    if (!stream_destroy(stream, &status)) {
      return status;
    }
  }

  buf->ptr = nullptr;
  buf->bytes = 0;
  buf->is_pinned = false;
  return CUDA_OK;
}

/**
 * @brief Reallocate a CUDA buffer to a new size, preserving content.
 *
 * Allocates a new buffer of @p new_bytes (rounded up to @p align),
 * copies min(old_bytes, new_bytes) from the old buffer, frees the old
 * buffer, and updates the descriptor.
 *
 * For device memory all three operations (alloc, copy, free) are issued
 * onto a single temporary stream, which is then synchronised and destroyed.
 * For pinned host memory the allocation and copy use synchronous host-side
 * operations (cudaMallocHost / memcpy / cudaFreeHost); no stream is needed.
 *
 * On any failure the original buffer descriptor is left unchanged and
 * all newly-allocated resources are cleaned up before returning.
 *
 * @param buf       Pointer to the buffer descriptor to reallocate.
 *                  Must have been previously allocated with cuda_reserve.
 * @param new_bytes Target size in bytes (before alignment rounding).
 * @param align     Alignment requirement (power of two, or 1 for default).
 * @return CUDA_OK on success, or a CudaStatus_t with a positive error
 *         code and a descriptive message on failure.
 */
CudaStatus_t cuda_resize(CudaBuffer_t *buf, std::size_t new_bytes,
                         std::size_t align) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return CudaStatus_t{.code = 1,
                        .msg = "cuda_resize: buf or buf->ptr is null"
                               " — nothing to reallocate"};
  }

  CudaStatus_t status = {};
  const std::size_t alloc_bytes =
      (align > 1) ? align_up(new_bytes, align) : new_bytes;
  const std::size_t copy_bytes =
      buf->bytes < alloc_bytes ? buf->bytes : alloc_bytes;
  void *new_ptr = nullptr;

  if (buf->is_pinned) {
    const cudaError_t err = cudaMallocHost(&new_ptr, alloc_bytes);
    if (err != cudaSuccess) {
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    std::memcpy(new_ptr, buf->ptr, copy_bytes);

    const cudaError_t free_err = cudaFreeHost(buf->ptr);
    if (free_err != cudaSuccess) {
      cudaFreeHost(new_ptr);
      status.code = map_cuda_error(free_err);
      status.msg = cudaGetErrorString(free_err);
      return status;
    }
  } else {
    cudaStream_t stream = nullptr;
    if (!stream_create(&stream, &status)) {
      return status;
    }

    cudaError_t err = cudaMallocAsync(&new_ptr, alloc_bytes, stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    err = cudaMemcpyAsync(new_ptr, buf->ptr, copy_bytes,
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      cudaFreeAsync(new_ptr, stream);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    err = cudaFreeAsync(buf->ptr, stream);
    if (err != cudaSuccess) {
      cudaFreeAsync(new_ptr, stream);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      status.code = map_cuda_error(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    if (!stream_sync(stream, &status)) {
      cudaStreamDestroy(stream);
      return status;
    }
    if (!stream_destroy(stream, &status)) {
      return status;
    }
  }

  buf->ptr = new_ptr;
  buf->bytes = alloc_bytes;
  return CUDA_OK;
}

#else // !__has_include(<cuda_runtime_api.h>)

/**
 * @brief Fallback allocation entry point when CUDA headers are unavailable.
 */
CudaStatus_t cuda_reserve(std::size_t, std::size_t, bool, CudaBuffer_t *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/**
 * @brief Fallback release entry point when CUDA headers are unavailable.
 */
CudaStatus_t cuda_release(CudaBuffer_t *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/**
 * @brief Fallback realloc entry point when CUDA headers are unavailable.
 */
CudaStatus_t cuda_resize(CudaBuffer_t *, std::size_t, std::size_t) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

#endif
