/**
 * @file hip_allocator.hpp
 * @brief Device memory allocator for HIP (ROCm) buffers.
 *
 * Provides a thin wrapper around hipMalloc / hipHostMalloc and
 * hipFree / hipHostFree, with alignment support and descriptive
 * error reporting via HipStatus_t.
 */

#pragma once

#include <cstddef>

/**
 * @brief Descriptor for an allocated HIP buffer.
 *
 * @var ptr       Pointer to the device (or pinned host) memory.
 * @var bytes     Usable size of the allocation in bytes (after alignment).
 * @var is_pinned true if the buffer lives in pinned (page-locked) host memory.
 */
struct HipBuffer_t {
  void *ptr = nullptr;
  std::size_t bytes = 0;
  bool is_pinned = false;
};

/**
 * @brief Result type returned by hip_reserve, hip_realloc and hip_release.
 *
 * @var code  Zero on success, a positive error code on failure
 *            (1 = invalid value, 2 = allocation failure, -1 = unknown).
 * @var msg   Human-readable error description (valid even on success).
 */
struct HipStatus_t {
  int code = 0;
  const char *msg = "ok";

  /** @brief Explicit conversion to bool for checking success status. */
  explicit operator bool() const noexcept { return code == 0; }
};

/** @brief Shorthand for a successful HipStatus_t. */
inline constexpr HipStatus_t HIP_OK{.code = 0, .msg = "ok"};

/**
 * @brief Allocate a HIP buffer.
 *
 * Wraps hipMalloc (device memory) or hipMallocHost (pinned host memory).
 * The requested size is rounded up to the next multiple of @p align when
 * @p align > 1.
 *
 * @param bytes  Minimum number of bytes to allocate.
 * @param align  Alignment requirement (must be a power of two, or 1 for
 *               default alignment).
 * @param pinned If true, allocate page-locked host memory via
 *               hipMallocHost; otherwise allocate device memory via
 *               hipMalloc.
 * @param out    Output buffer descriptor (only valid when the returned
 *               status is HIP_OK).
 * @return HipStatus_t with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
HipStatus_t hip_reserve(std::size_t bytes, std::size_t align, bool pinned,
                        HipBuffer_t *out);

/**
 * @brief Free a HIP buffer previously allocated with hip_reserve.
 *
 * Wraps hipFreeHost or hipFree depending on the buffer's is_pinned flag.
 * The buffer descriptor is zeroed out on success.
 *
 * @param buf Pointer to the buffer descriptor to free.  If buf or
 *            buf->ptr is NULL the function returns an error status.
 * @return HipStatus_t with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
HipStatus_t hip_release(HipBuffer_t *buf);

/**
 * @brief Reallocate a HIP buffer to a new size, preserving content.
 *
 * Allocates a new buffer of @p new_bytes (rounded up to @p align),
 * copies min(old_bytes, new_bytes) from the old buffer, frees the old
 * buffer, and updates the descriptor.
 *
 * For device memory the sequence uses hipMalloc, hipMemcpy(DeviceToDevice)
 * and hipFree — all synchronous.  For pinned host memory it uses
 * hipHostMalloc, host-side memcpy, and hipHostFree.
 *
 * On any failure the original buffer descriptor is left unchanged and
 * all newly-allocated resources are cleaned up before returning.
 *
 * @param buf       Pointer to the buffer descriptor to reallocate.
 *                  Must have been previously allocated with hip_reserve.
 * @param new_bytes Target size in bytes (before alignment rounding).
 * @param align     Alignment requirement (power of two, or 1 for default).
 * @return HIP_OK on success, or a HipStatus_t with a positive error
 *         code and a descriptive message on failure.
 */
HipStatus_t hip_realloc(HipBuffer_t *buf, std::size_t new_bytes,
                        std::size_t align);
