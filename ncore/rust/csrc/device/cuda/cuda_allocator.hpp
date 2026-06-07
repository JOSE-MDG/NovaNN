/**
 * @file cuda_allocator.hpp
 * @brief CUDA memory allocation types and operations.
 *
 * @details
 * Declares the @ref CudaBuffer_t descriptor, the @ref CudaStatus_t
 * result type, and the three core allocation functions
 * (@ref cuda_reserve, @ref cuda_release, @ref cuda_resize) that
 * manage CUDA device and pinned-host memory.
 *
 * This header is consumed by `ffi.cpp` to implement the
 * device-agnostic allocation API.  It is never included directly
 * from outside the csrc module.
 *
 * @see cuda_allocator.cpp  Implementation of the allocation functions.
 * @see cuda_io.hpp         CUDA data transfer functions.
 * @see ffi.hpp             Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>

/**
 * @struct CudaBuffer_t
 * @brief Descriptor for a CUDA-allocated memory region.
 *
 * @details
 * Tracks the raw pointer, usable size, and allocation type
 * (device or pinned-host).  The @ref ptr member is the pointer
 * returned by the CUDA runtime; callers never touch the CUDA
 * API directly.
 */
struct CudaBuffer_t {
  void *ptr = nullptr;    ///< Device or pinned-host pointer.
  std::size_t bytes = 0;  ///< Usable size in bytes.
  bool is_pinned = false; ///< Page-locked host memory flag.
};

/**
 * @struct CudaStatus_t
 * @brief Result type for CUDA allocation and copy operations.
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
 * CudaStatus_t status = cuda_reserve(...);
 * if (!status) {
 *     // handle error using status.code and status.msg
 * }
 * @endcode
 */
struct CudaStatus_t {
  int code = 0;           ///< Zero on success, positive on failure.
  const char *msg = "ok"; ///< Human-readable error description.
  explicit operator bool() const noexcept { return code == 0; }
};

/**
 * @var CUDA_OK
 * @brief Sentinel value representing a successful CUDA operation.
 */
inline constexpr CudaStatus_t CUDA_OK{.code = 0, .msg = "ok"};

/**
 * @brief Allocate a CUDA memory buffer.
 *
 * @details
 * For pinned (page-locked) allocations, uses `cudaMallocHost`.
 * For device allocations, creates a temporary CUDA stream, calls
 * `cudaMallocAsync`, synchronises, and destroys the stream.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  align  Required alignment in bytes.  If `> 1`, the
 *                    allocated size is rounded up to the next
 *                    multiple of @p align.
 * @param[in]  pinned If `true`, allocate page-locked host memory
 *                    via `cudaMallocHost`.
 * @param[out] out    Receives the allocated buffer descriptor on
 *                    success.
 *
 * @return @ref CUDA_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @post On success, @p out->ptr points to a valid CUDA memory
 *       region of at least @p bytes (aligned to @p align).
 *
 * @see cuda_release()  Frees a buffer allocated by this function.
 * @see cuda_resize()   Resizes an existing buffer.
 */
CudaStatus_t cuda_reserve(std::size_t bytes, std::size_t align, bool pinned,
                          CudaBuffer_t *out);

/**
 * @brief Free a CUDA memory buffer previously allocated by
 *        @ref cuda_reserve.
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
 * @pre  @p buf must point to a valid @ref CudaBuffer_t whose
 *       @ref ptr member was returned by @ref cuda_reserve.
 * @post On success, @p buf is zeroed (ptr = nullptr, bytes = 0,
 *       is_pinned = false).
 *
 * @note Safe to call with a null @p buf or null @p buf->ptr;
 *       returns a non-zero status without crashing.
 *
 * @see cuda_reserve()  Allocates the buffer freed here.
 */
CudaStatus_t cuda_release(CudaBuffer_t *buf);

/**
 * @brief Resize an existing CUDA memory buffer.
 *
 * @details
 * Allocates a new buffer of @p new_bytes (aligned to @p align),
 * copies `min(old_size, new_size)` bytes from the old buffer to
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
 * @pre  @p buf must point to a valid @ref CudaBuffer_t whose
 *       @ref ptr member was returned by @ref cuda_reserve.
 * @post On success, @p buf->ptr and @p buf->bytes are updated to
 *       reflect the new allocation.
 *
 * @warning On failure the original buffer may already be freed
 *          (e.g., if the copy succeeded but the free failed).
 *          Do not use the old @p buf->ptr after a failed resize.
 *
 * @see cuda_reserve()  Initial allocation.
 * @see cuda_release()  Explicit deallocation.
 */
CudaStatus_t cuda_resize(CudaBuffer_t *buf, std::size_t new_bytes,
                         std::size_t align);
