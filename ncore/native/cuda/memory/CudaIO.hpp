/**
 * @file CudaIO.hpp
 * @brief CUDA data transfer function for host-to-device,
 * device-to-host, and device-to-device copies.
 *
 * @details
 * Declares the single memcpy function @ref cudaTransfer that
 * handles all copy directions via a @ref DeviceMemcpyKind tag.
 * Internally uses a reusable CUDA stream and `cudaMemcpyAsync`
 * for all transfers, followed by stream synchronisation.
 *
 * ## Stream Model
 *
 * This module maintains a single reusable CUDA stream (singleton
 * pattern) created on first call to @ref cudaTransfer.  All
 * subsequent transfers are serialised on this stream.  The stream
 * is not destroyed during program execution; the CUDA runtime
 * reclaims it on process exit.
 *
 * ## Error Handling
 *
 * Errors from `cudaMemcpyAsync` and `cudaStreamSynchronize` are
 * mapped to @ref cudaStatus_t codes via the internal @ref map_error
 * function.  The caller receives `CUDA_OK` on success or a
 * descriptive error status otherwise.
 *
 * ## Thread Safety
 *
 * @ref cudaTransfer is safe to call from multiple threads.  The
 * internal stream serialises all transfers, and CUDA runtime
 * calls are thread-safe.
 *
 * This header is the CUDA counterpart of `HipIO.hpp` and
 * provides an identical API surface.  The dispatch layer in
 * `ffi.cpp` selects between CUDA and HIP at runtime.
 *
 * @see CudaIO.cpp        Implementation of the transfer function.
 * @see CudaAllocator.hpp CUDA memory allocation operations.
 * @see ffi.hpp           Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include "CudaAllocator.hpp"
#include "ffi.hpp"
#include <cstddef>

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using `cudaMemcpyAsync` on a
 * reusable internal CUDA stream, then synchronises the stream
 * before returning.  The transfer direction is determined by
 * @p kind.
 *
 * ### Supported Directions
 *
 * - `deviceMemcpyHostToDevice` — H2D (host source, device dest).
 * - `deviceMemcpyDeviceToHost` — D2H (device source, host dest).
 * - `deviceMemcpyDeviceToDevice` — D2D (device source, device dest).
 *
 * ### Execution Flow
 *
 * 1. Obtain the singleton stream via @ref get_stream.
 * 2. Call `cudaMemcpyAsync(dst, src, bytes, kind, stream)`.
 * 3. If step 2 fails, return mapped error status.
 * 4. Call `cudaStreamSynchronize(stream)` to block until the
 *    transfer completes.
 * 5. If step 4 fails, return mapped error status.
 * 6. Return `CUDA_OK`.
 *
 * ### Error Codes
 *
 * - `0` — Success.
 * - `1` — Invalid value (null pointer, zero bytes, etc).
 * - `2` — Invalid memcpy direction.
 * - `3` — Invalid resource handle (bad stream).
 * - `-1` — Unrecognised CUDA error.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 *
 * @return @ref CUDA_OK on success, or a @ref cudaStatus_t with
 *         a non-zero code and a descriptive message.
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
 * @note If @p kind is `deviceMemcpyHostToDevice` or
 *       `deviceMemcpyDeviceToHost`, the host-side pointer should
 *       ideally be page-locked for best async performance.
 *       However, `cudaMemcpyAsync` handles pageable memory by
 *       internally staging through a pinned buffer.
 *
 * @see deviceMemcpy()  Device-agnostic wrapper that calls this.
 */
cudaStatus_t cudaTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                          const void *src, void *dst);
