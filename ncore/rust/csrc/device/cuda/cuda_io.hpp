/**
 * @file cuda_io.hpp
 * @brief CUDA data transfer functions for host-device and
 *        device-device copies.
 *
 * @details
 * Declares synchronous and asynchronous memcpy variants that
 * move data between host and device memory, or between two
 * device allocations.  The master dispatcher @ref cuda_memcpy
 * routes to the correct variant based on a @ref DeviceMemcpyKind
 * tag.
 *
 * This header is consumed by `ffi.cpp` and is never included
 * directly from outside the csrc module.
 *
 * @see cuda_io.cpp        Implementation of the transfer functions.
 * @see cuda_allocator.hpp CUDA memory allocation operations.
 * @see ffi.hpp            Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include "cuda_allocator.hpp"
#include "ffi.hpp"
#include <cstddef>

/**
 * @brief Synchronous host-to-device memory copy.
 *
 * @details
 * Calls `cudaMemcpy` with `cudaMemcpyHostToDevice`.  The
 * transfer blocks until complete.
 *
 * @param[in] bytes  Number of bytes to copy.
 * @param[in] src    Source pointer (host memory).
 * @param[out] dst   Destination pointer (device memory).
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p src must point to valid host memory of at least
 *       @p bytes.
 * @pre  @p dst must point to valid device memory of at least
 *       @p bytes.
 *
 * @see cuda_memcpy_host2device_async()  Async variant.
 */
CudaStatus_t cuda_memcpy_host2device(std::size_t bytes, const void *src,
                                     void *dst);

/**
 * @brief Asynchronous host-to-device memory copy.
 *
 * @details
 * When @p pinned is `true`, uses `cudaMemcpyAsync` on a stream
 * for non-blocking transfer.  When @p pinned is `false`, falls
 * back to the synchronous @ref cuda_memcpy path.
 *
 * @param[in] bytes  Number of bytes to copy.
 * @param[in] src    Source pointer (host memory).
 * @param[out] dst   Destination pointer (device memory).
 * @param[in] pinned Whether @p src is page-locked host memory.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @see cuda_memcpy_host2device()  Synchronous variant.
 */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned);

/**
 * @brief Synchronous device-to-host memory copy.
 *
 * @details
 * Calls `cudaMemcpy` with `cudaMemcpyDeviceToHost`.  The
 * transfer blocks until complete.
 *
 * @param[in] bytes  Number of bytes to copy.
 * @param[in] src    Source pointer (device memory).
 * @param[out] dst   Destination pointer (host memory).
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @see cuda_memcpy_device2host_async()  Async variant.
 */
CudaStatus_t cuda_memcpy_device2host(std::size_t bytes, const void *src,
                                     void *dst);

/**
 * @brief Asynchronous device-to-host memory copy.
 *
 * @details
 * When @p pinned is `true`, uses `cudaMemcpyAsync` on a stream
 * for non-blocking transfer.  When @p pinned is `false`, falls
 * back to the synchronous @ref cuda_memcpy path.
 *
 * @param[in] bytes  Number of bytes to copy.
 * @param[in] src    Source pointer (device memory).
 * @param[out] dst   Destination pointer (host memory).
 * @param[in] pinned Whether @p dst is page-locked host memory.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @see cuda_memcpy_device2host()  Synchronous variant.
 */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned);

/**
 * @brief Device-to-device memory copy.
 *
 * @details
 * Uses @ref cuda_memcpy internally with
 * `DeviceMemcpyKind::deviceMemcpyDeviceToDevice`.  The transfer
 * is performed on a temporary stream and synchronised before
 * returning.
 *
 * @param[in] bytes  Number of bytes to copy.
 * @param[in] src    Source pointer (device memory).
 * @param[out] dst   Destination pointer (device memory).
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p src and @p dst must not overlap.
 */
CudaStatus_t cuda_memcpy_device2device(std::size_t bytes, const void *src,
                                       void *dst);

/**
 * @brief Master memcpy dispatcher for CUDA.
 *
 * @details
 * Creates a temporary CUDA stream, performs the copy in the
 * direction specified by @p kind, synchronises the stream, and
 * destroys it before returning.  The @p is_pinned flag controls
 * whether async transfers are used for host-side pointers.
 *
 * @param[in] bytes     Number of bytes to copy.
 * @param[in] kind      Copy direction (@ref DeviceMemcpyKind).
 * @param[in] src       Source pointer.
 * @param[out] dst      Destination pointer.
 * @param[in] is_pinned Whether the host-side pointer is
 *                      page-locked.
 *
 * @return @ref CUDA_OK on success, or an error status.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src
 *       and @p dst.
 *
 * @see device_memcpy()  Device-agnostic wrapper that calls this.
 */
CudaStatus_t cuda_memcpy(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst, bool is_pinned);
