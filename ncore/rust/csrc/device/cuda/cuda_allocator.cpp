/**
 * @file cuda_allocator.cpp
 * @brief CUDA memory allocation, release, and resize implementation.
 *
 * @details
 * Implements the three core allocation primitives used by the
 * device-agnostic FFI layer (`ffi.cpp`).  All device-memory
 * operations use a temporary CUDA stream for async allocation
 * and synchronise before returning.
 *
 * The file is conditionally compiled behind `NOVA_HAS_CUDA` and
 * `__has_include(<cuda_runtime_api.h>)`.  When CUDA headers are
 * unavailable (e.g., during linting), stub functions that return
 * an error status are provided.
 *
 * ## Error Mapping
 *
 * Each CUDA error code is mapped to an integer:
 * - `0` — success
 * - `1` — invalid value
 * - `2` — memory allocation failure (reserve only)
 * - `3` — not supported / invalid resource handle
 * - `-1` — unrecognised error
 *
 * @see cuda_allocator.hpp  Type declarations and function signatures.
 * @see cuda_io.cpp         CUDA data transfer implementation.
 * @see ffi.cpp             Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include "cuda_allocator.hpp"
#include <cstring>
#include <cuda_runtime_api.h>

/**
 * @brief Map a CUDA error code to an integer error code.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return `0` for success, `1` for invalid value, `2` for memory
 *         allocation failure, `3` for not supported, `-1` for
 *         unrecognised errors.
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
 * @brief Map a CUDA stream operation error to an integer code.
 *
 * @param[in] err  The CUDA error from a stream operation.
 *
 * @return `0` for success, `1` for invalid value, `2` for
 *         external device error, `-1` for unrecognised errors.
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
 * @brief Map a CUDA stream destruction error to an integer code.
 *
 * @param[in] err  The CUDA error from stream destruction.
 *
 * @return `0` for success, `1` for invalid value, `3` for
 *         invalid resource handle, `4` for external device
 *         error, `-1` for unrecognised errors.
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
 * @brief Map a CUDA stream synchronisation error to an integer code.
 *
 * @param[in] err  The CUDA error from stream synchronisation.
 *
 * @return `0` for success, `1` for invalid resource handle,
 *         `-1` for unrecognised errors.
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
 * @brief Round @p bytes up to the nearest multiple of @p align.
 *
 * @param[in] bytes  The value to align.
 * @param[in] align  The alignment (must be a power of two).
 *
 * @return The smallest value >= @p bytes that is a multiple of
 *         @p align.
 */
static constexpr std::size_t align_up(std::size_t bytes,
                                      std::size_t align) noexcept {
  return (bytes + (align - 1)) & ~(align - 1);
}

/**
 * @brief Create a CUDA stream.
 *
 * @param[out] stream  Receives the new stream handle.
 * @param[out] status  Receives the error status on failure.
 *
 * @return `true` on success, `false` on failure.
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
 * @brief Block until all work on @p stream has completed.
 *
 * @param[in]  stream  The stream to synchronise.
 * @param[out] status  Receives the error status on failure.
 *
 * @return `true` on success, `false` on failure.
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
 * @brief Destroy a CUDA stream.
 *
 * @param[in]  stream  The stream to destroy.
 * @param[out] status  Receives the error status on failure.
 *
 * @return `true` on success, `false` on failure.
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
 * @brief Allocate a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls `cudaMallocHost`.  For device memory,
 * creates a temporary stream, calls `cudaMallocAsync`,
 * synchronises, and destroys the stream.
 *
 * @param[in]  bytes  Requested size in bytes.
 * @param[in]  align  Alignment in bytes.  If `> 1`, the allocation
 *                    is rounded up to the next multiple.
 * @param[in]  pinned If `true`, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p bytes must be greater than zero.
 * @post On success, @p out->ptr points to valid CUDA memory.
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
 * @brief Free a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls `cudaFreeHost`.  For device memory,
 * creates a temporary stream, calls `cudaFreeAsync`, synchronises,
 * and destroys the stream.  On success, the buffer is zeroed.
 *
 * @param[in,out] buf  Buffer descriptor to free.  Must not be null.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @post On success, @p buf is zeroed.
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
 * @brief Resize a CUDA memory buffer.
 *
 * @details
 * Allocates a new buffer, copies `min(old, new)` bytes, then frees
 * the old buffer.  For pinned memory the copy uses
 * `std::memcpy`; for device memory it uses `cudaMemcpyAsync` on a
 * temporary stream.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     align     Alignment in bytes.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @post On success, @p buf->ptr and @p buf->bytes are updated.
 *
 * @warning On failure the original buffer may be freed.
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

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_reserve(std::size_t, std::size_t, bool, CudaBuffer_t *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_release(CudaBuffer_t *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_resize(CudaBuffer_t *, std::size_t, std::size_t) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

#endif
#endif /* NOVA_HAS_CUDA */
