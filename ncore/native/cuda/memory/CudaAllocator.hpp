/**
 * @file CudaAllocator.hpp
 * @brief CUDA memory allocation types and operations.
 *
 * @details
 * Declares the @ref cudaBuffer_t descriptor, the @ref CUDA_OK
 * sentinel, and the three core allocation functions
 * (@ref cudaReserve, @ref cudaRelease, @ref cudaResize) that manage
 * CUDA device and pinned-host memory.
 *
 * @section memory-types Memory Types
 *
 * This module handles two kinds of CUDA memory:
 * @li Device memory — allocated via @c cudaMallocAsync on a
 *   temporary stream or @c cudaMalloc if MemoryPools is not supported,
 *   suitable for GPU kernel access.
 * @li Pinned (page-locked) host memory — allocated via
 *   @c cudaMallocHost, suitable for fast host-device transfers
 *   with @c cudaMemcpyAsync.
 *
 * @section error-handling Error Handling
 *
 * All functions return a @ref novaStatus_t defined in @ref status.h
 *
 * @section stream-lifecycle Stream Lifecycle
 *
 * Device-memory operations (reserve, release, resize) create a
 * temporary CUDA stream, perform the operation asynchronously,
 * synchronize, and destroy the stream before returning. Provided that
 * MemoryPools is available on the device (which is usually the case).
 *
 * This header is the CUDA counterpart of @c HipAllocator.hpp and
 * provides an identical API surface.  The dispatch layer in
 * @c ffi.cpp selects between CUDA and HIP at runtime.
 *
 * @see CudaAllocator.cpp  Implementation of the allocation functions.
 * @see CudaIO.hpp         CUDA data transfer functions.
 * @see ffi.hpp            Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>
#include <ncore/core/status.h>

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
 * @brief Sentinel value representing a successful CUDA operation.
 * @var CUDA_OK
 */
const inline novaStatus_t CUDA_OK{
    .err = novaSuccess, .message = nova_get_error_msg(novaSuccess, nullptr)};

/**
 * @brief Allocate a CUDA memory buffer.
 *
 * @details
 * For pinned memory, calls @c cudaMallocHost.  For device memory,
 * creates a temporary stream, calls @c cudaMallocAsync,
 * synchronizes, and destroys the stream.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  pinned If @c true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 *
 * @return @ref CUDA_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success, @p out->ptr points to a valid CUDA memory
 *       region of at least @p bytes.
 *
 * @see cudaRelease()  Frees a buffer allocated by this function.
 * @see cudaResize()   Resizes an existing buffer.
 */
novaStatus_t cudaReserve(std::size_t bytes, bool pinned, cudaBuffer_t *out);

/**
 * @brief Free a CUDA memory buffer previously allocated by
 *        @ref cudaReserve.
 *
 * @details
 * For pinned memory, calls @c cudaFreeHost.  For device memory,
 * creates a temporary stream, calls @c cudaFreeAsync, synchronizes,
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
novaStatus_t cudaRelease(cudaBuffer_t *buf);

/**
 * @brief Resize an existing CUDA memory buffer.
 *
 * @details
 * Allocates a new buffer of @p new_bytes, copies
 * @c min(old_size, new_size) bytes from the old buffer to
 * the new one, then frees the old buffer.  For pinned memory the
 * copy uses @c std::memcpy; for device memory it uses
 * @c cudaMemcpyAsync on a temporary stream.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to
 *                          resize.  Must not be null.
 * @param[in]     new_bytes New size in bytes.
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
novaStatus_t cudaResize(cudaBuffer_t *buf, std::size_t new_bytes);
