/**
 * @file CudaAllocator.hpp
 * @brief CUDA memory allocation types and operations.
 *
 * @details
 * Declares the @ref cudaBuffer_t descriptor, the @ref cudaStatus_t
 * result type, and the three core allocation functions
 * ( @ref cudaReserve, @ref cudaRelease, @ref cudaResize) that
 * manage CUDA device and pinned-host memory.
 *
 * ## Memory Types
 *
 * This module handles two kinds of CUDA memory:
 * - **Device memory** — allocated via `cudaMallocAsync` on a
 *   temporary stream, suitable for GPU kernel access.
 * - **Pinned (page-locked) host memory** — allocated via
 *   `cudaMallocHost`, suitable for fast host-device transfers
 *   with `cudaMemcpyAsync`.
 *
 * ## Error Handling
 *
 * All functions return a @ref cudaStatus_t.  A @ref code of `0`
 * indicates success.  The @ref msg field carries a human-readable
 * description (string literal or CUDA error string).
 *
 * ## Stream Lifecycle
 *
 * Device-memory operations (reserve, release, resize) create a
 * temporary CUDA stream, perform the operation asynchronously,
 * synchronise, and destroy the stream before returning.  This
 * ensures each call is self-contained and thread-safe.
 *
 * This header is the CUDA counterpart of `HipAllocator.hpp` and
 * provides an identical API surface.  The dispatch layer in
 * `ffi.cpp` selects between CUDA and HIP at runtime.
 *
 * @see CudaAllocator.cpp  Implementation of the allocation functions.
 * @see CudaIO.hpp         CUDA data transfer functions.
 * @see ffi.hpp            Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>

/**
 * @struct cudaBuffer_t
 * @brief Descriptor for a CUDA-allocated memory region.
 *
 * @details
 * Tracks the raw pointer, usable size, and allocation type
 * (device or pinned-host).  This struct is owned by the
 * @ref deviceBuffer_t in the FFI layer and must not be freed
 * directly — use @ref cudaRelease instead.
 */
struct cudaBuffer_t {
  void *ptr = nullptr;   ///< Device or pinned-host pointer.
  std::size_t bytes = 0; ///< Usable size in bytes.
  bool isPinned = false; ///< Page-locked host memory flag.
};

/**
 * @struct cudaStatus_t
 * @brief Result type for CUDA operations.
 *
 * @details
 * Carries a numeric error code and a human-readable message.
 * A @ref code of `0` indicates success.  The @ref msg member
 * points to a string literal or a CUDA error string; it is
 * valid for the lifetime of the program.
 *
 * Provides an `explicit operator bool()` for convenient
 * success/failure checking:
 *
 * @code{.cpp}
 * cudaStatus_t status = cudaReserve(...);
 * if (!status) {
 *     // handle error using status.code and status.msg
 * }
 * @endcode
 */
struct cudaStatus_t {
  int code = 0;           ///< Zero on success, positive on failure.
  const char *msg = "ok"; ///< Human-readable error description.
  explicit operator bool() const noexcept { return code == 0; }
};

/**
 * @brief Sentinel value representing a successful CUDA operation.
 * @var CUDA_OK
 */
inline constexpr cudaStatus_t CUDA_OK{.code = 0, .msg = "ok"};

/**
 * @brief Allocate a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls `cudaMallocHost`.  For device memory,
 * creates a temporary stream, calls `cudaMallocAsync`,
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
 * @return @ref CUDA_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to a valid CUDA memory
 *       region of at least @p bytes (aligned to @p align if
 *       applicable).
 *
 * @see cudaRelease()  Frees a buffer allocated by this function.
 * @see cudaResize()   Resizes an existing buffer.
 */
cudaStatus_t cudaReserve(std::size_t bytes, std::size_t align, bool pinned,
                         cudaBuffer_t *out);

/**
 * @brief Free a CUDA memory buffer previously allocated by
 *        @ref cudaReserve.
 *
 * @details
 * For pinned memory, calls `cudaFreeHost`.  For device memory,
 * creates a temporary stream, calls `cudaFreeAsync`, synchronises,
 * and destroys the stream.  On success, the buffer descriptor
 * is zeroed.
 *
 * @param[in,out] buf  Pointer to the buffer descriptor to free.
 *                     Must not be null, and @p buf->ptr must be
 *                     valid.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p buf must point to a valid @ref cudaBuffer_t whose
 *       @ref ptr member was returned by @ref cudaReserve.
 * @post On success, @p buf is zeroed (ptr = nullptr, bytes = 0,
 *       isPinned = false).
 *
 * @note Safe to call with a null @p buf or null @p buf->ptr;
 *       returns a non-zero status without crashing.
 *
 * @see cudaReserve()  Allocates the buffer freed here.
 */
cudaStatus_t cudaRelease(cudaBuffer_t *buf);

/**
 * @brief Resize an existing CUDA memory buffer.
 *
 * @details
 * Allocates a new buffer of @p new_bytes, copies
 * `min(old_size, new_size)` bytes from the old buffer to
 * the new one, then frees the old buffer.  For pinned memory the
 * copy uses `std::memcpy`; for device memory it uses
 * `cudaMemcpyAsync` on a temporary stream.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to
 *                          resize.  Must not be null.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     align     Required alignment in bytes.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p buf must point to a valid @ref cudaBuffer_t whose
 *       @ref ptr member was returned by @ref cudaReserve.
 * @post On success, @p buf->ptr and @p buf->bytes are updated to
 *       reflect the new allocation.
 *
 * @warning On failure the original buffer may already be freed
 *          (e.g., if the copy succeeded but the free failed).
 *          Do not use the old @p buf->ptr after a failed resize.
 *
 * @see cudaReserve()  Initial allocation.
 * @see cudaRelease()  Explicit deallocation.
 */
cudaStatus_t cudaResize(cudaBuffer_t *buf, std::size_t new_bytes,
                        std::size_t align);
