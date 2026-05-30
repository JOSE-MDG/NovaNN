/**
 * @file cuda_allocator.hpp
 * @brief Device memory allocator for CUDA buffers.
 *
 * Provides a thin wrapper around cudaMallocHost / cudaMallocAsync
 * and cudaFreeHost / cudaFreeAsync, with alignment support and
 * descriptive error reporting via CudaStatus_t.
 */

#pragma once

#include <cstddef>

/**
 * @brief Descriptor for an allocated CUDA buffer.
 *
 * @var ptr       Pointer to the device (or pinned host) memory.
 * @var bytes     Usable size of the allocation in bytes (after alignment).
 * @var is_pinned true if the buffer lives in pinned (page-locked) host memory.
 */
struct CudaBuffer_t {
  void *ptr = nullptr;
  std::size_t bytes = 0;
  bool is_pinned = false;
};

/**
 * @brief Result type returned by cuda_reserve, cuda_resize and cuda_release.
 *
 * @var code  Zero on success, a positive error code on failure
 *            (1 = invalid value, 2 = allocation failure, -1 = unknown).
 * @var msg   Human-readable error description (valid even on success).
 */
struct CudaStatus_t {
  int code = 0;
  const char *msg = "ok";

  /** @brief Explicit conversion to bool for checking success status. */
  explicit operator bool() const noexcept { return code == 0; }
};

/** @brief Shorthand for a successful CudaStatus_t. */
inline constexpr CudaStatus_t CUDA_OK{.code = 0, .msg = "ok"};

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
                          CudaBuffer_t *out);

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
CudaStatus_t cuda_release(CudaBuffer_t *buf);

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
                         std::size_t align);
