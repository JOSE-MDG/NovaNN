/**
 * @file cuda_io.hpp
 * @brief Convenience wrappers for CUDA memory copies.
 *
 * Provides simple host↔device and device↔device copy helpers with optional
 * asynchronous transfers when host memory is pinned.
 *
 * These helpers return CudaStatus_t for consistent error reporting across
 * the CUDA backend. Runtime-specific CUDA types are intentionally kept out
 * of this header so it can be included beside HIP headers.
 */

#pragma once

#include "cuda_allocator.hpp"
#include "../../ffi.hpp"
#include <cstddef>

/**
 * @brief Copy bytes from host to device (synchronous).
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in host memory.
 * @param dst   Destination pointer in device memory.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_host2device(std::size_t bytes, const void *src,
                                     void *dst);

/**
 * @brief Copy bytes from host to device (async when pinned).
 *
 * Delegates to cuda_memcpy with the HostToDevice direction.
 *
 * @param bytes  Number of bytes to copy.
 * @param src    Source pointer in host memory.
 * @param dst    Destination pointer in device memory.
 * @param pinned Whether the host buffer is pinned (page-locked).
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned);

/**
 * @brief Copy bytes from device to host (synchronous).
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in host memory.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_device2host(std::size_t bytes, const void *src,
                                     void *dst);

/**
 * @brief Copy bytes from device to host (async when pinned).
 *
 * Delegates to cuda_memcpy with the DeviceToHost direction.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in host memory.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned);

/**
 * @brief Copy bytes from device to device (async).
 *
 * Delegates to cuda_memcpy with the DeviceToDevice direction.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in device memory.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_device2device(std::size_t bytes, const void *src,
                                       void *dst);

/**
 * @brief Unified copy entry point — owns the full stream lifetime.
 *
 * Creates a stream, dispatches to the appropriate internal helper based on
 * @p kind, then synchronises and destroys the stream unconditionally before
 * returning.  A sync or destroy error takes priority over a copy error only
 * when the copy itself succeeded.
 *
 * @param bytes     Number of bytes to copy.
 * @param kind      Copy direction.
 * @param src       Source pointer.
 * @param dst       Destination pointer.
 * @param is_pinned Whether the host buffer is pinned (page-locked).
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst, bool is_pinned);
