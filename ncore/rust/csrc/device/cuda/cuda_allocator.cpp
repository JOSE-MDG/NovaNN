/**
 * @file cuda_allocator.cpp
 * @brief Backend implementation of CUDA device/pinned-host memory allocation.
 *
 * Implements cuda_reserve and cuda_release by wrapping the CUDA Driver
 * API (cudaMalloc / cudaMallocHost / cudaFree / cudaFreeHost) with
 * alignment rounding, descriptive error reporting, and a mapping from
 * CUDA runtime error codes to application-level status codes.
 */

#include "cuda_allocator.hpp"
#include <cuda_runtime_api.h>

/**
 * @brief Map a CUDA runtime error to an application-level status code.
 *
 * @param err CUDA error value returned by the runtime API.
 * @return 0 for cudaSuccess, 1 for cudaErrorInvalidValue,
 *         2 for cudaErrorMemoryAllocation, -1 for any other error.
 */
static int map_cuda_error(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return 0;
  case cudaErrorInvalidValue:
    return 1;
  case cudaErrorMemoryAllocation:
    return 2;
  default:
    return -1;
  }
}

/**
 * @brief Return a human-readable description for an application-level
 *        CUDA status code.
 *
 * @param code Status code produced by map_cuda_error().
 * @return Static string describing the error (never NULL).
 */
static const char *cuda_error_msg(int code) {
  switch (code) {
  case 0:
    return "ok";
  case 1:
    return "cudaErrorInvalidValue: one or more parameters is invalid (e.g., "
           "null pointer, misaligned size, or out-of-range argument)";
  case 2:
    return "cudaErrorMemoryAllocation: the device or host memory allocation "
           "failed — possible causes: out of memory, fragmented heap, or "
           "driver limit reached";
  default:
    return "unknown cuda error: an unrecognized CUDA driver API error occurred";
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
 * @brief Allocate a CUDA buffer with optional alignment.
 *
 * Wraps cudaMalloc (device memory) or cudaMallocHost (pinned host memory).
 * If @p align > 1, the allocation size is rounded up so that the returned
 * buffer satisfies the alignment constraint.
 *
 * @param bytes  Minimum number of bytes to allocate.
 * @param align  Alignment requirement (must be a power of two, or 1 for
 *               default alignment).
 * @param pinned If true, allocate page-locked host memory via
 *               cudaMallocHost; otherwise allocate device memory via
 *               cudaMalloc.
 * @param out    Output buffer descriptor (only valid when the returned
 *               status has code == 0).
 * @return CUDA_OK on success, or a CudaStatus_t with a positive error code
 *         and a descriptive message on failure.
 */
CudaStatus_t cuda_reserve(std::size_t bytes, std::size_t align, bool pinned,
                          CudaBuffer_t *out) {
  void *ptr = nullptr;
  const std::size_t alloc_bytes = (align > 1) ? align_up(bytes, align) : bytes;
  const cudaError_t err = pinned ? cudaMallocHost(&ptr, alloc_bytes)
                                 : cudaMalloc(&ptr, alloc_bytes);
  const int code = map_cuda_error(err);
  if (code != 0) {
    return CudaStatus_t{.code = code, .msg = cuda_error_msg(code)};
  }
  *out = CudaBuffer_t{.ptr = ptr, .bytes = alloc_bytes, .is_pinned = pinned};
  return CUDA_OK;
}

/**
 * @brief Free a CUDA buffer previously allocated with cuda_reserve().
 *
 * Wraps cudaFreeHost or cudaFree depending on the buffer's is_pinned flag.
 * On success the output descriptor is zeroed out so that a subsequent
 * release is a safe no-op (aside from the null-pointer error check).
 *
 * @param buf Pointer to the buffer descriptor to free.  If buf or
 *            buf->ptr is NULL the function returns an error status
 *            without calling any CUDA API.
 * @return CUDA_OK on success, or a CudaStatus_t with a positive error code
 *         and a descriptive message on failure.
 */
CudaStatus_t cuda_release(CudaBuffer_t *buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return CudaStatus_t{
        .code = 1,
        .msg = "cuda_release: buf or buf->ptr is null — nothing to free"};
  }
  const cudaError_t err =
      buf->is_pinned ? cudaFreeHost(buf->ptr) : cudaFree(buf->ptr);
  const int code = map_cuda_error(err);
  if (code != 0) {
    return CudaStatus_t{.code = code, .msg = cuda_error_msg(code)};
  }
  *buf = CudaBuffer_t{};
  return CUDA_OK;
}
