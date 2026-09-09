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
 * Device-memory operations (reserve, release, resize) run on a CUDA
 * stream selected per call:
 * @li @c stream == nullptr (default) — a temporary stream is created,
 *   the operation runs on it, the stream is synchronized, and the
 *   stream is destroyed before returning. Synchronous contract.
 * @li @c stream != nullptr — the operation is enqueued on the caller
 *   stream with no synchronization and no stream destruction. The
 *   caller owns ordering: every access to the buffer must be ordered
 *   after the allocation and before the free in stream order, per the
 *   stream-ordered allocator rules. Pinned-host paths ignore @p stream.
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

#if defined(NOVA_HAS_CUDA) && __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st *;
#endif

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
 * For pinned memory, calls @c cudaMallocHost (synchronous; @p stream
 * is ignored).  For device memory with memory pools, calls
 * @c cudaMallocAsync on @p stream when given, or on a temporary
 * synchronized stream when @p stream is null.  Without memory pools,
 * falls back to @c cudaMalloc (synchronous; @p stream is ignored).
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  pinned If @c true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 * @param[in]  stream Caller stream for chaining, or null for the
 *                    synchronous internal path. Never destroyed here.
 *
 * @return @ref CUDA_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success with @p stream == null, @p out->ptr points to a
 *       valid CUDA memory region of at least @p bytes usable from
 *       any stream.
 * @post On success with @p stream != null, @p out->ptr is valid only
 *       in @p stream order (ordered after this call).
 *
 * @warning With @p stream != null there is no synchronization: any
 *          access outside @p stream order is undefined behavior.
 *
 * @see cudaRelease()  Frees a buffer allocated by this function.
 * @see cudaResize()   Resizes an existing buffer.
 */
novaStatus_t cudaReserve(std::size_t bytes, bool pinned, cudaBuffer_t *out,
                         cudaStream_t stream = nullptr);

/**
 * @brief Free a CUDA memory buffer previously allocated by
 *        @ref cudaReserve.
 *
 * @details
 * For pinned memory, calls @c cudaFreeHost (synchronous; @p stream
 * is ignored).  For device memory with memory pools, calls
 * @c cudaFreeAsync on @p stream when given, or on a temporary
 * synchronized stream when @p stream is null.  On success, the
 * buffer descriptor is zeroed.
 *
 * @param[in,out] buf    Pointer to the buffer descriptor to free.
 *                       Must not be null, and @p buf->ptr must be
 *                       valid.
 * @param[in]     stream Caller stream for chaining, or null for the
 *                       synchronous internal path. Never destroyed here.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p buf must point to a valid @ref cudaBuffer_t whose
 *       @ref ptr member was returned by @ref cudaReserve.
 * @pre  With @p stream != null, every prior access to @p buf->ptr
 *       must be ordered before this call in @p stream order.
 * @post On success, @p buf is zeroed (ptr = nullptr, bytes = 0,
 *       isPinned = false).
 *
 * @note Safe to call with a null @p buf or null @p buf->ptr;
 *       returns a non-zero status without crashing.
 *
 * @warning With @p stream != null there is no synchronization: the
 *          memory must not be accessed after this call in any stream
 *          unless reordered with events.
 *
 * @see cudaReserve()  Allocates the buffer freed here.
 */
novaStatus_t cudaRelease(cudaBuffer_t *buf, cudaStream_t stream = nullptr);

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
 * @param[in]     stream    Caller stream for chaining, or null for
 *                          the synchronous internal path. The alloc,
 *                          copy, and free all run on this stream.
 *                          Never destroyed here.
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
novaStatus_t cudaResize(cudaBuffer_t *buf, std::size_t new_bytes,
                         cudaStream_t stream = nullptr);
