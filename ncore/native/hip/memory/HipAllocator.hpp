/**
 * @file HipAllocator.hpp
 * @brief HIP memory allocation types and operations.
 *
 * @details
 * Declares the @ref hipBuffer_t descriptor and the three core allocation
 * functions (@ref hipReserve, @ref hipRelease, @ref hipResize) that
 * manage HIP device and pinned-host memory.
 *
 * @section memory-types Memory Types
 *
 * This module handles two kinds of HIP memory:
 * @li Device memory — allocated via @c hipMallocAsync on a temporary stream
 *   when memory pools are supported, or @c hipMalloc otherwise.
 * @li Pinned (page-locked) host memory — allocated via
 *   @c hipHostMalloc, suitable for fast host-device transfers
 *   with @c hipMemcpyAsync.
 *
 * @section error-handling Error Handling
 *
 * All functions return a @ref novaStatus_t defined in @ref status.h.
 *
 * @section stream-lifecycle Stream Lifecycle
 *
 * Device-memory operations use a temporary HIP stream when memory pools are
 * available and fall back to synchronous HIP allocation APIs otherwise.
 *
 * This header is the HIP counterpart of @c CudaAllocator.hpp and
 * provides an identical API surface.  The dispatch layer in
 * @c ffi.cpp selects between CUDA and HIP at runtime.
 *
 * @see HipAllocator.cpp  Implementation of the allocation functions.
 * @see HipIO.hpp         HIP data transfer functions.
 * @see ffi.hpp           Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>
#include <ncore/core/status.h>

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
 * @brief Sentinel value representing a successful HIP operation.
 * @var HIP_OK
 */
const inline novaStatus_t HIP_OK{
    .err = novaSuccess, .message = nova_get_error_msg(novaSuccess, nullptr)};

/**
 * @brief Allocate a HIP memory buffer.
 *
 * @details
 * For pinned memory, calls @c hipHostMalloc.  For device memory,
 * creates a temporary stream, calls @c hipMallocAsync,
 * synchronises, and destroys the stream.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  pinned If @c true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref HIP_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to a valid HIP memory
 *       region of at least @p bytes.
 *
 * @see hipRelease()  Frees a buffer allocated by this function.
 * @see hipResize()   Resizes an existing buffer.
 */
novaStatus_t hipReserve(std::size_t bytes, bool pinned, hipBuffer_t *out);

/**
 * @brief Free a HIP memory buffer previously allocated by
 *        @ref hipReserve.
 *
 * @details
 * For pinned memory, calls @c hipFreeHost.  For device memory,
 * creates a temporary stream, calls @c hipFreeAsync, synchronises,
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
novaStatus_t hipRelease(hipBuffer_t *buf);

/**
 * @brief Resize an existing HIP memory buffer.
 *
 * @details
 * Allocates a new buffer of @p new_bytes, copies
 * @c min(old_size, new_size) bytes from the old buffer to
 * the new one, then frees the old buffer.  For pinned memory the
 * copy uses @c std::memcpy; for device memory it uses
 * @c hipMemcpyAsync on a temporary stream.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to
 *                          resize.  Must not be null.
 * @param[in]     new_bytes New size in bytes.
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
novaStatus_t hipResize(hipBuffer_t *buf, std::size_t new_bytes);
