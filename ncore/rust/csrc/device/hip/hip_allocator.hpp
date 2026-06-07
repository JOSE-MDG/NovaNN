/**
 * @file hip_allocator.hpp
 * @brief HIP memory allocation types and operations.
 *
 * @details
 * Declares the @ref HipBuffer_t descriptor, the @ref HipStatus_t
 * result type, and the three core allocation functions
 * (@ref hip_reserve, @ref hip_release, @ref hip_resize) that
 * manage HIP device and pinned-host memory.
 *
 * This header is the HIP counterpart of `cuda_allocator.hpp` and
 * provides an identical API surface.  The dispatch layer in
 * `ffi.cpp` selects between CUDA and HIP at runtime.
 *
 * @see hip_allocator.cpp  Implementation of the allocation functions.
 * @see hip_io.hpp         HIP data transfer functions.
 * @see ffi.hpp            Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>

/**
 * @struct HipBuffer_t
 * @brief Descriptor for a HIP-allocated memory region.
 *
 * @details
 * Tracks the raw pointer, usable size, and allocation type
 * (device or pinned-host).  The @ref ptr member is the pointer
 * returned by the HIP runtime; callers never touch the HIP
 * API directly.
 */
struct HipBuffer_t {
  void *ptr = nullptr;    ///< Device or pinned-host pointer.
  std::size_t bytes = 0;  ///< Usable size in bytes.
  bool is_pinned = false; ///< Page-locked host memory flag.
};

/**
 * @struct HipStatus_t
 * @brief Result type for HIP allocation and copy operations.
 *
 * @details
 * Carries a numeric error code and a human-readable message.
 * A @ref code of `0` indicates success.  The @ref msg member
 * points to a string literal or a HIP error string; it is
 * valid for the lifetime of the program.
 *
 * Provides an `explicit operator bool()` for convenient
 * success/failure checking:
 *
 * @code{.cpp}
 * HipStatus_t status = hip_reserve(...);
 * if (!status) {
 *     // handle error using status.code and status.msg
 * }
 * @endcode
 */
struct HipStatus_t {
  int code = 0;           ///< Zero on success, positive on failure.
  const char *msg = "ok"; ///< Human-readable error description.
  explicit operator bool() const noexcept { return code == 0; }
};

/**
 * @var HIP_OK
 * @brief Sentinel value representing a successful HIP operation.
 */
inline constexpr HipStatus_t HIP_OK{.code = 0, .msg = "ok"};

/**
 * @brief Allocate a HIP memory buffer.
 *
 * @details
 * For pinned (page-locked) allocations, uses `hipHostMalloc`.
 * For device allocations, creates a temporary HIP stream, calls
 * `hipMallocAsync`, synchronises, and destroys the stream.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  align  Required alignment in bytes.  If `> 1`, the
 *                    allocated size is rounded up to the next
 *                    multiple of @p align.
 * @param[in]  pinned If `true`, allocate page-locked host memory
 *                    via `hipHostMalloc`.
 * @param[out] out    Receives the allocated buffer descriptor on
 *                    success.
 *
 * @return @ref HIP_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @post On success, @p out->ptr points to a valid HIP memory
 *       region of at least @p bytes (aligned to @p align).
 *
 * @see hip_release()  Frees a buffer allocated by this function.
 * @see hip_resize()   Resizes an existing buffer.
 */
HipStatus_t hip_reserve(std::size_t bytes, std::size_t align, bool pinned,
                        HipBuffer_t *out);

/**
 * @brief Free a HIP memory buffer previously allocated by
 *        @ref hip_reserve.
 *
 * @details
 * For pinned memory, calls `hipFreeHost`.  For device memory,
 * creates a temporary stream, calls `hipFreeAsync`, synchronises,
 * and destroys the stream.  On success, the buffer descriptor
 * is zeroed.
 *
 * @param[in,out] buf  Pointer to the buffer descriptor to free.
 *                     Must not be null, and @p buf->ptr must be
 *                     valid.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @pre  @p buf must point to a valid @ref HipBuffer_t whose
 *       @ref ptr member was returned by @ref hip_reserve.
 * @post On success, @p buf is zeroed (ptr = nullptr, bytes = 0,
 *       is_pinned = false).
 *
 * @note Safe to call with a null @p buf or null @p buf->ptr;
 *       returns a non-zero status without crashing.
 *
 * @see hip_reserve()  Allocates the buffer freed here.
 */
HipStatus_t hip_release(HipBuffer_t *buf);

/**
 * @brief Resize an existing HIP memory buffer.
 *
 * @details
 * Allocates a new buffer of @p new_bytes (aligned to @p align),
 * copies `min(old_size, new_size)` bytes from the old buffer to
 * the new one, then frees the old buffer.  For pinned memory the
 * copy uses `std::memcpy`; for device memory it uses
 * `hipMemcpyAsync` on a temporary stream.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to
 *                          resize.  Must not be null.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     align     Required alignment in bytes.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @pre  @p buf must point to a valid @ref HipBuffer_t whose
 *       @ref ptr member was returned by @ref hip_reserve.
 * @post On success, @p buf->ptr and @p buf->bytes are updated to
 *       reflect the new allocation.
 *
 * @warning On failure the original buffer may already be freed
 *          (e.g., if the copy succeeded but the free failed).
 *          Do not use the old @p buf->ptr after a failed resize.
 *
 * @see hip_reserve()  Initial allocation.
 * @see hip_release()  Explicit deallocation.
 */
HipStatus_t hip_resize(HipBuffer_t *buf, std::size_t new_bytes,
                       std::size_t align);
