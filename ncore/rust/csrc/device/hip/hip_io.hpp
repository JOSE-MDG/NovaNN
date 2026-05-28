/**
 * @file hip_io.hpp
 * @brief Convenience wrappers for HIP memory copies.
 *
 * Provides simple host↔device and device↔device copy helpers with optional
 * asynchronous transfers when host memory is pinned.
 *
 * These functions are the HIP equivalent of the CUDA helpers in
 * cuda_io.hpp/cuda_io.cpp. Runtime-specific HIP types are intentionally kept
 * out of this header so it can be included beside CUDA headers.
 */

#pragma once

#include "hip_allocator.hpp"
#include "../../ffi.hpp"
#include <cstddef>

/**
 * @brief Copy bytes from host to device (synchronous).
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in host memory.
 * @param dst   Destination pointer in device memory.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_host2device(std::size_t bytes, const void *src,
                                   void *dst);

/**
 * @brief Copy bytes from host to device (async when pinned).
 *
 * Delegates to hip_memcpy with the HostToDevice direction.
 *
 * @param bytes  Number of bytes to copy.
 * @param src    Source pointer in host memory.
 * @param dst    Destination pointer in device memory.
 * @param pinned Whether the host buffer is pinned (page-locked).
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_host2device_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned);

/**
 * @brief Copy bytes from device to host (synchronous).
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in host memory.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_device2host(std::size_t bytes, const void *src,
                                   void *dst);

/**
 * @brief Copy bytes from device to host (async when pinned).
 *
 * Delegates to hip_memcpy with the DeviceToHost direction.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in host memory.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_device2host_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned);

/**
 * @brief Copy bytes from device to device (async).
 *
 * Delegates to hip_memcpy with the DeviceToDevice direction.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in device memory.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_device2device(std::size_t bytes, const void *src,
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
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy(std::size_t bytes, DeviceMemcpyKind kind,
                       const void *src, void *dst, bool is_pinned);
