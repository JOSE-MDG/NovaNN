/**
 * @file CudaIO.hpp
 * @brief CUDA data transfer function for host-to-device,
 * device-to-host, and device-to-device copies.
 *
 * @details
 * Declares the single memcpy function @ref cudaTransfer that
 * handles all copy directions via a @ref DeviceMemcpyKind tag.
 * By default the module maintains a single reusable CUDA stream,
 * created on first call to @ref cudaTransfer under @c std::call_once:
 * transfers are serialized on it and synchronized before returning.
 * The stream is never destroyed during execution; the CUDA runtime
 * reclaims it at process exit.  A caller stream may instead be
 * passed via @p stream to chain the copy with surrounding work:
 * enqueued with no synchronization, the caller owns ordering.
 *
 * Errors map to @ref novaStatus_t through the internal @ref mapError.
 * @ref cudaTransfer is safe to call from multiple threads: the
 * internal stream serializes transfers and CUDA runtime calls are
 * thread-safe.
 *
 * This header is the CUDA counterpart of @c HipIO.hpp with an
 * identical API surface; the dispatch layer in @c ffi.cpp selects
 * between CUDA and HIP at runtime.
 *
 * @see CudaIO.cpp        Implementation of the transfer function.
 * @see CudaAllocator.hpp CUDA memory allocation operations.
 * @see ffi.hpp           Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>

#include <ncore/core/status.h>

#include "CudaAllocator.hpp"
#include "ffi.hpp"

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using @c cudaMemcpyAsync on the singleton
 * stream when @p stream is null (synchronized before returning), or
 * enqueued on @p stream with no synchronization when given.  The
 * transfer direction is determined by @p kind.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 * @param[in]  stream    Caller stream for chaining, or null for the
 *                       synchronous singleton path. Never destroyed here.
 *
 * @return @ref CUDA_OK on success, or a @ref novaStatus_t with
 *         a non-zero error code and a descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p src and @p dst must point to valid memory regions of at
 *       least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src
 *       and @p dst (e.g., host pointer for H2D source).
 * @pre  @p src and @p dst must not overlap.
 *
 * @post On success with @p stream == null, @p dst contains
 *       @p bytes copied from @p src.
 * @post On success with @p stream != null, the copy is ordered in
 *       @p stream after this call returns.
 *
 * @warning With @p stream != null there is no synchronization: the
 *          destination must not be read before @p stream passes this
 *          point.
 *
 * @note With pageable host memory @c cudaMemcpyAsync internally
 *       stages through a pinned buffer; page-locked pointers avoid
 *       that extra hop.
 *
 * @see deviceMemcpy()  Device-agnostic wrapper that calls this.
 */
novaStatus_t cudaTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                          const void *src, void *dst,
                          cudaStream_t stream = nullptr);
