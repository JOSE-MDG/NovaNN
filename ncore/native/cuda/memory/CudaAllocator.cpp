/**
 * @file CudaAllocator.cpp
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
 * ## Architecture
 *
 * Internal helpers (within anonymous namespace):
 * - @ref mapError — maps any `cudaError_t` to an integer code.
 * - @ref alignUp — rounds a byte count up to a multiple.
 * - @ref streamCreate — creates a temporary CUDA stream.
 * - @ref stream_sync — blocks until stream work completes.
 * - @ref streamDestroy — destroys a CUDA stream.
 *
 * ## Error Mapping
 *
 * All CUDA errors are mapped via @ref mapError:
 * - `cudaSuccess` → 0
 * - `cudaErrorInvalidValue` → 1
 * - `cudaErrorMemoryAllocation` → 2
 * - `cudaErrorNotSupported` → 3
 * - `cudaErrorInvalidResourceHandle` → 4
 * - All others → -1
 *
 * @see CudaAllocator.hpp  Type declarations and function signatures.
 * @see CudaIO.cpp         CUDA data transfer implementation.
 * @see ffi.cpp            Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include "CudaAllocator.hpp"
#include <cstring>
#include <cuda_runtime_api.h>

namespace {

/**
 * @brief Map a CUDA error code to an integer.
 *
 * @details
 * Converts `cudaError_t` values into project-standard integer
 * codes.  Covers errors from allocation (`cudaMallocAsync`,
 * `cudaMallocHost`), deallocation (`cudaFreeAsync`,
 * `cudaFreeHost`), stream operations, and memcpy.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return Integer code: 0 for success, 1-4 for specific errors,
 *         -1 for unrecognised errors.
 */
int mapError(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return 0;
  case cudaErrorInvalidValue:
    return 1;
  case cudaErrorMemoryAllocation:
    return 2;
  case cudaErrorNotSupported:
    return 3;
  case cudaErrorInvalidResourceHandle:
    return 4;
  default:
    return -1;
  }
}

/**
 * @brief Round @p bytes up to the nearest multiple of @p align.
 *
 * @details
 * Uses bitwise AND to round up efficiently.  Requires @p align
 * to be a power of two.
 *
 * @param[in] bytes  The value to align.
 * @param[in] align  The alignment (must be a power of two).
 *
 * @return The smallest value >= @p bytes that is a multiple of
 *         @p align.
 */
constexpr std::size_t alignUp(std::size_t bytes, std::size_t align) noexcept {
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
bool streamCreate(cudaStream_t *stream, cudaStatus_t *status) {
  const cudaError_t err = cudaStreamCreate(stream);
  if (err != cudaSuccess) {
    status->code = mapError(err);
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
bool streamSync(cudaStream_t stream, cudaStatus_t *status) {
  const cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    status->code = mapError(err);
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
bool streamDestroy(cudaStream_t stream, cudaStatus_t *status) {
  const cudaError_t err = cudaStreamDestroy(stream);
  if (err != cudaSuccess) {
    status->code = mapError(err);
    status->msg = cudaGetErrorString(err);
    return false;
  }
  return true;
}

} // namespace

/**
 * @brief Allocate a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls `cudaMallocHost`.  For device memory,
 * creates a temporary stream, calls `cudaMallocAsync`,
 * synchronises, and destroys the stream.
 *
 * @param[in]  bytes  Requested size in bytes.
 * @param[in]  align  Alignment in bytes.
 * @param[in]  pinned If `true`, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to valid CUDA memory.
 */
cudaStatus_t cudaReserve(std::size_t bytes, std::size_t align, bool pinned,
                         cudaBuffer_t *out) {
  cudaStatus_t status = {};
  const std::size_t allocBytes = (align > 1) ? alignUp(bytes, align) : bytes;
  void *ptr = nullptr;

  if (pinned) {
    const cudaError_t err = cudaMallocHost(&ptr, allocBytes);
    if (err != cudaSuccess) {
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }
  } else {
    cudaStream_t stream = nullptr;
    if (!streamCreate(&stream, &status)) {
      return status;
    }

    const cudaError_t err = cudaMallocAsync(&ptr, allocBytes, stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    if (!streamSync(stream, &status)) {
      cudaStreamDestroy(stream);
      return status;
    }
    if (!streamDestroy(stream, &status)) {
      return status;
    }
  }

  out->ptr = ptr;
  out->bytes = allocBytes;
  out->isPinned = pinned;
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
cudaStatus_t cudaRelease(cudaBuffer_t *buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return cudaStatus_t{.code = 1,
                        .msg = "cudaRelease: buf or buf->ptr is null"
                               " — nothing to free\n"};
  }

  cudaStatus_t status = {};

  if (buf->isPinned) {
    const cudaError_t err = cudaFreeHost(buf->ptr);
    if (err != cudaSuccess) {
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }
  } else {
    cudaStream_t stream = nullptr;
    if (!streamCreate(&stream, &status)) {
      return status;
    }

    const cudaError_t err = cudaFreeAsync(buf->ptr, stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    if (!streamSync(stream, &status)) {
      cudaStreamDestroy(stream);
      return status;
    }
    if (!streamDestroy(stream, &status)) {
      return status;
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
cudaStatus_t cudaResize(cudaBuffer_t *buf, std::size_t new_bytes,
                        std::size_t align) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return cudaStatus_t{.code = 1,
                        .msg = "cudaResize: buf or buf->ptr is null"
                               " — nothing to reallocate\n"};
  }

  cudaStatus_t status = {};
  const std::size_t allocBytes =
      (align > 1) ? alignUp(new_bytes, align) : new_bytes;
  const std::size_t copyBytes =
      buf->bytes < allocBytes ? buf->bytes : allocBytes;
  void *newPtr = nullptr;

  if (buf->isPinned) {
    const cudaError_t err = cudaMallocHost(&newPtr, allocBytes);
    if (err != cudaSuccess) {
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    std::memcpy(newPtr, buf->ptr, copyBytes);

    const cudaError_t freeErr = cudaFreeHost(buf->ptr);
    if (freeErr != cudaSuccess) {
      cudaFreeHost(newPtr);
      status.code = mapError(freeErr);
      status.msg = cudaGetErrorString(freeErr);
      return status;
    }
  } else {
    cudaStream_t stream = nullptr;
    if (!streamCreate(&stream, &status)) {
      return status;
    }

    cudaError_t err = cudaMallocAsync(&newPtr, allocBytes, stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    err = cudaMemcpyAsync(newPtr, buf->ptr, copyBytes, cudaMemcpyDeviceToDevice,
                          stream);
    if (err != cudaSuccess) {
      cudaFreeAsync(newPtr, stream);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    err = cudaFreeAsync(buf->ptr, stream);
    if (err != cudaSuccess) {
      cudaFreeAsync(newPtr, stream);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      status.code = mapError(err);
      status.msg = cudaGetErrorString(err);
      return status;
    }

    if (!streamSync(stream, &status)) {
      cudaStreamDestroy(stream);
      return status;
    }
    if (!streamDestroy(stream, &status)) {
      return status;
    }
  }

  buf->ptr = newPtr;
  buf->bytes = allocBytes;
  return CUDA_OK;
}

#else // !__has_include(<cuda_runtime_api.h>)

/** @brief Stub: CUDA runtime headers not available. */
cudaStatus_t cudaReserve(std::size_t, std::size_t, bool, cudaBuffer_t *) {
  return cudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available\n"};
}

/** @brief Stub: CUDA runtime headers not available. */
cudaStatus_t cudaRelease(cudaBuffer_t *) {
  return cudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available\n"};
}

/** @brief Stub: CUDA runtime headers not available. */
cudaStatus_t cudaResize(cudaBuffer_t *, std::size_t, std::size_t) {
  return cudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available\n"};
}

#endif
#endif /* NOVA_HAS_CUDA */
