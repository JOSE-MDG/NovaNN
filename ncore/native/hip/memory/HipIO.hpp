/**
 * @file HipIO.hpp
 * @brief HIP data transfer function for host-to-device,
 * device-to-host, and device-to-device copies.
 *
 * @details
 * Declares the single memcpy function @ref hipTransfer that
 * handles all copy directions via a @ref DeviceMemcpyKind tag.
 * Transfers use @c hipMemcpyAsync everywhere: caller-stream copies
 * chain with no synchronization, and the null path runs on the null
 * stream, draining it before returning.
 *
 * With @p stream null the copy enqueues on the null stream and
 * @ref hipTransfer blocks on @c hipStreamSynchronize(nullptr), so the
 * device null queue drains first: the caller observes every kernel it
 * already enqueued there.  No private stream is created: rocclr
 * silently cancels a pending null-stream kernel at
 * @c hipStreamCreate, and a non-blocking work stream never drains the
 * null queue.
 *
 * Errors from @c hipMemcpyAsync and @c hipStreamSynchronize are
 * mapped to @ref novaStatus_t codes via @ref mapError2Status.
 * @ref hipTransfer is safe to call from multiple threads:
 * the null path serializes through the device null stream, and HIP
 * runtime calls are thread-safe.
 *
 * This header is the HIP counterpart of @c CudaIO.hpp with an
 * identical API surface; the dispatch layer in @c ffi.cpp selects
 * between CUDA and HIP at runtime.
 *
 * @see HipIO.cpp        Implementation of the transfer function.
 * @see HipAllocator.hpp HIP memory allocation operations.
 * @see common.hh        Shared error-mapping and stream helpers.
 * @see ffi.hpp          Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>

#include <ncore/core/status.h>

#include "HipAllocator.hpp"
#include "ffi.hpp"

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Performs a memory transfer using @c hipMemcpyAsync on the null
 * stream when @p stream is null, blocking until the device null queue
 * drains before returning, or enqueued on @p stream with no
 * synchronization when given.  The transfer direction is determined
 * by @p kind.
 *
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src       Source pointer (host or device memory).
 * @param[out] dst       Destination pointer (host or device memory).
 * @param[in]  stream    Caller stream for chaining, or null for the
 *                       synchronous null-stream path.
 *
 * @return @ref HIP_OK on success, or a @ref novaStatus_t with
 *         a non-zero error and a descriptive message.
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
 * @note With pageable host memory @c hipMemcpyAsync internally
 *       stages through a pinned buffer; page-locked pointers avoid
 *       that extra hop.
 *
 * @see deviceMemcpy()  Device-agnostic wrapper that calls this.
 */
novaStatus_t hipTransfer(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst,
                         hipStream_t stream = nullptr);
