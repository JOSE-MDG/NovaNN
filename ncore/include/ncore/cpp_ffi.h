/**
 * @file cpp_ffi.h
 * @brief C-callable wrappers for GPU memory operations.
 *
 * Declares device_memcpy_c, an extern-"C" entry point that accepts the
 * C-side TransferKind enum and dispatches to the active CUDA or HIP
 * backend through the C++ FFI layer in ffi.hpp / ffi.cpp.
 *
 * Include this header from pure-C translation units that need to copy
 * device memory without pulling in C++ type definitions.
 */

#pragma once
#include <ncore/device.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copy bytes through the active GPU backend (C-callable wrapper).
 *
 * Dispatches to cuda_memcpy or hip_memcpy according to the active
 * backend.  If no backend is available the returned status has code -1
 * and a descriptive message.
 *
 * @param src       Source pointer.
 * @param dst       Destination pointer.
 * @param is_pinned Whether the host-side pointer is pinned/page-locked.
 * @param kind      Device-agnostic copy direction (TransferKind).
 * @param bytes     Number of bytes to copy.
 * @return DeviceStatus with code 0 on success, or an error status.
 */
DeviceStatus device_memcpy_c(const void *src, void *dst, bool is_pinned,
                             TransferKind kind, size_t bytes);

#ifdef __cplusplus
}
#endif
