/**
 * @file HipAllocator.hpp
 * @brief HIP memory allocation types and operations.
 *
 * @details
 * Declares the @ref hipBuffer_t descriptor, the @ref hipStatus_t
 * result type, and the three core allocation functions
 * (@ref hipReserve, @ref hipRelease, @ref hipResize) that
 * manage HIP device and pinned-host memory.
 *
 * ## Memory Types
 *
 * This module handles two kinds of HIP memory:
 * - **Device memory** — allocated via `hipMallocAsync` on a
 *   temporary stream, suitable for GPU kernel access.
 * - **Pinned (page-locked) host memory** — allocated via
 *   `hipHostMalloc`, suitable for fast host-device transfers
 *   with `hipMemcpyAsync`.
 *
 * ## Error Handling
 *
 * All functions return a @ref hipStatus_t.  A @ref code of `0`
 * indicates success.  The @ref msg field carries a human-readable
 * description (string literal or HIP error string).
 *
 * ## Stream Lifecycle
 *
 * Device-memory operations (reserve, release, resize) create a
 * temporary HIP stream, perform the operation asynchronously,
 * synchronise, and destroy the stream before returning.  This
 * ensures each call is self-contained and thread-safe.
 *
 * This header is the HIP counterpart of `CudaAllocator.hpp` and
 * provides an identical API surface.  The dispatch layer in
 * `ffi.cpp` selects between CUDA and HIP at runtime.
 *
 * @see HipAllocator.cpp  Implementation of the allocation functions.
 * @see HipIO.hpp         HIP data transfer functions.
 * @see ffi.hpp           Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>

/**
 * @struct hipBuffer_t
 * @brief Descriptor for a HIP-allocated memory region.
 *
 * @details
 * Tracks the raw pointer, usable size, and allocation type
 * (device or pinned-host).  This struct is owned by the
 * @ref deviceBuffer_t in the FFI layer and must not be freed
 * directly — use @ref hipRelease instead.
 */
struct hipBuffer_t {
  void *ptr = nullptr;   ///< Device or pinned-host pointer.
  std::size_t bytes = 0; ///< Usable size in bytes.
  bool isPinned = false; ///< Page-locked host memory flag.
};

/**
 * @struct hipStatus_t
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
 * hipStatus_t status = hipReserve(...);
 * if (!status) {
 *     // handle error using status.code and status.msg
 * }
 * @endcode
 */
struct hipStatus_t {
  int code = 0;           ///< Zero on success, positive on failure.
  const char *msg = "ok"; ///< Human-readable error description.
  explicit operator bool() const noexcept { return code == 0; }
};

/**
 * @brief Sentinel value representing a successful HIP operation.
 * @var HIP_OK
 */
inline constexpr hipStatus_t HIP_OK{.code = 0, .msg = "ok"};

/**
 * @brief Allocate a HIP memory buffer.
 *
 * @details
 * For pinned memory, calls `hipHostMalloc`.  For device memory,
 * creates a temporary stream, calls `hipMallocAsync`,
 * synchronises, and destroys the stream.
 *
 * If @p align is greater than 1, the allocated size is rounded
 * up to the nearest multiple of @p align.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  align  Required alignment in bytes.
 * @param[in]  pinned If `true`, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref HIP_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to a valid HIP memory
 *       region of at least @p bytes (aligned to @p align if
 *       applicable).
 *
 * @see hipRelease()  Frees a buffer allocated by this function.
 * @see hipResize()   Resizes an existing buffer.
 */
hipStatus_t hipReserve(std::size_t bytes, std::size_t align, bool pinned,
                       hipBuffer_t *out);

/**
 * @brief Free a HIP memory buffer previously allocated by
 *        @ref hipReserve.
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
 * @pre  @p buf must point to a valid @ref hipBuffer_t whose
 *       @ref ptr member was returned by @ref hipReserve.
 * @post On success, @p buf is zeroed (ptr = nullptr, bytes = 0,
 *       isPinned = false).
 *
 * @note Safe to call with a null @p buf or null @p buf->ptr;
 *       returns a non-zero status without crashing.
 *
 * @see hipReserve()  Allocates the buffer freed here.
 */
hipStatus_t hipRelease(hipBuffer_t *buf);

/**
 * @brief Resize an existing HIP memory buffer.
 *
 * @details
 * Allocates a new buffer of @p new_bytes, copies
 * `min(old_size, new_size)` bytes from the old buffer to
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
 * @pre  @p buf must point to a valid @ref hipBuffer_t whose
 *       @ref ptr member was returned by @ref hipReserve.
 * @post On success, @p buf->ptr and @p buf->bytes are updated to
 *       reflect the new allocation.
 *
 * @warning On failure the original buffer may already be freed
 *          (e.g., if the copy succeeded but the free failed).
 *          Do not use the old @p buf->ptr after a failed resize.
 *
 * @see hipReserve()  Initial allocation.
 * @see hipRelease()  Explicit deallocation.
 */
hipStatus_t hipResize(hipBuffer_t *buf, std::size_t new_bytes,
                      std::size_t align);
