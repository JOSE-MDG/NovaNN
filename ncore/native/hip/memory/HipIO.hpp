/**
 * @file HipIO.hpp
 * @brief HIP data transfer function for host-to-device,
 * device-to-host, and device-to-device copies.
 *
 * @details
 * Declares the single memcpy function @ref hipTransfer that
 * handles all copy directions via a @ref DeviceMemcpyKind tag.
 * Internally uses a reusable HIP stream and @c hipMemcpyAsync
 * for all transfers, followed by stream synchronisation.
 *
 * @section stream-model Stream Model
 *
 * This module maintains a single reusable HIP stream (singleton
 * pattern) created on first call to @ref hipTransfer.  All
 * subsequent transfers are serialised on this stream.  The stream
 * is not destroyed during program execution; the HIP runtime
 * reclaims it on process exit.
 *
 * @section error-handling Error Handling
 *
 * Errors from @c hipMemcpyAsync and @c hipStreamSynchronize are
 * mapped to @ref novaStatus_t codes via the internal @ref mapError
 * function.  The caller receives @c HIP_OK on success or a
 * descriptive error status otherwise.
 *
 * @section thread-safety Thread Safety
 *
 * @ref hipTransfer is safe to call from multiple threads.  The
 * internal stream serialises all transfers, and HIP runtime
 * calls are thread-safe.
 *
 * This header is the HIP counterpart of @c CudaIO.hpp and
 * provides an identical API surface.  The dispatch layer in
 * @c ffi.cpp selects between CUDA and HIP at runtime.
 *
 * @see HipIO.cpp        Implementation of the transfer function.
 * @see HipAllocator.hpp HIP memory allocation operations.
 * @see ffi.hpp          Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>

#include <ncore/core/status.h>

#include "ffi.hpp"

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using @c hipMemcpyAsync on a
 * reusable internal HIP stream, then synchronises the stream
 * before returning.  The transfer direction is determined by
 * @p kind.
 *
 * @subsection supported-directions Supported Directions
 *
 * @li @c deviceMemcpyHostToDevice — H2D (host source, device dest).
 * @li @c deviceMemcpyDeviceToHost — D2H (device source, host dest).
 * @li @c deviceMemcpyDeviceToDevice — D2D (device source, device dest).
 *
 * @subsection execution-flow Execution Flow
 *
 * @li 1. Obtain the singleton stream via @ref get_stream.
 * @li 2. Call @c hipMemcpyAsync(dst, src, bytes, kind, stream).
 * @li 3. If step 2 fails, return mapped error status.
 * @li 4. Call @c hipStreamSynchronize(stream) to block until the
 *    transfer completes.
 * @li 5. If step 4 fails, return mapped error status.
 * @li 6. Return @c HIP_OK.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 *
 * @return @ref HIP_OK on success, or a @ref novaStatus_t with
 *         a non-zero error and a descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p src must point to a valid memory region of at least
 *       @p bytes.
 * @pre  @p dst must point to a valid memory region of at least
 *       @p bytes.
 * @pre  @p kind must match the actual memory types of @p src
 *       and @p dst (e.g., host pointer for H2D source).
 * @pre  @p src and @p dst must not overlap.
 *
 * @post On success, @p dst contains @p bytes copied from @p src.
 *
 * @note If @p kind is @c deviceMemcpyHostToDevice or
 *       @c deviceMemcpyDeviceToHost, the host-side pointer should
 *       ideally be page-locked for best async performance.
 *       However, @c hipMemcpyAsync handles pageable memory by
 *       internally staging through a pinned buffer.
 *
 * @see deviceMemcpy()  Device-agnostic wrapper that calls this.
 */
novaStatus_t hipTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst);
