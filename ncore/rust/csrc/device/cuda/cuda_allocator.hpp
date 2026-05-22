/**
 * @file cuda_allocator.hpp
 * @brief Device memory allocator for CUDA buffers.
 *
 * Provides a thin wrapper around cudaMalloc / cudaMallocHost and
 * cudaFree / cudaFreeHost, with alignment support and descriptive
 * error reporting via CudaStatus.
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
 * @brief Result type returned by cuda_reserve and cuda_release.
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

/** @brief Shorthand for a successful CudaStatus. */
inline constexpr CudaStatus_t CUDA_OK{.code = 0, .msg = "ok"};

/**
 * @brief Allocate a CUDA buffer.
 *
 * Wraps cudaMalloc (device memory) or cudaMallocHost (pinned host memory).
 * The requested size is rounded up to the next multiple of @p align when
 * @p align > 1.
 *
 * @param bytes  Minimum number of bytes to allocate.
 * @param align  Alignment requirement (must be a power of two).
 * @param pinned If true, allocate page-locked host memory via
 *               cudaMallocHost; otherwise allocate device memory via
 *               cudaMalloc.
 * @param out    Output buffer descriptor (only valid when the returned
 *               status is CUDA_OK).
 * @return CudaStatus with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
CudaStatus_t cuda_reserve(std::size_t bytes, std::size_t align, bool pinned,
                          CudaBuffer_t *out);

/**
 * @brief Free a CUDA buffer previously allocated with cuda_reserve.
 *
 * Wraps cudaFreeHost or cudaFree depending on the buffer's is_pinned flag.
 * The buffer descriptor is zeroed out on success.
 *
 * @param buf Pointer to the buffer descriptor to free.  If buf or
 *            buf->ptr is NULL the function returns an error status.
 * @return CudaStatus with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
CudaStatus_t cuda_release(CudaBuffer_t *buf);
