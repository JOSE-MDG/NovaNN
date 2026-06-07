/**
 * @file cpp_ffi.h
 * @brief C-callable wrappers for GPU memory operations.
 *
 * @details
 * Provides a pure-C entry point for GPU memory copies, allowing
 * C translation units to perform device-to-host, host-to-device,
 * and device-to-device transfers without including any C++ headers.
 *
 * ## Architecture
 *
 * ```
 * C code  →  device_memcpy_c()  →  ffi.hpp (C++ FFI)
 *                                      ↓
 *                               cuda_memcpy() / hip_memcpy()
 * ```
 *
 * The actual CUDA/HIP dispatch lives in the C++ layer
 * (`ffi.hpp` / `ffi.cpp`).  This header exposes only the
 * `extern "C"` shim that bridges the two worlds.
 *
 * ## Usage
 *
 * Include this header from pure-C files that need device memory
 * copies.  Do **not** include `ffi.hpp` directly from C code.
 *
 * @see device.h       Device enum, TransferKind, DeviceStatus.
 * @see ffi.hpp        C++ FFI implementation (not for C callers).
 */

#pragma once
#include <ncore/device.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copy bytes through the active GPU backend (C-callable
 *        wrapper).
 *
 * @details
 * Routes the copy to the correct backend based on the globally
 * detected device:
 * - **CUDA** → `cuda_memcpy()`
 * - **HIP**  → `hip_memcpy()`
 * - **CPU / Meta** → returns error status (code -1).
 *
 * If no GPU backend is available, the returned @ref DeviceStatus
 * has `code == -1` and a descriptive message.
 *
 * @param[in]      src       Source pointer.  For
 *                           `DEVICE_TO_DEVICE` this is device
 *                           memory; for `HOST_TO_DEVICE` it is
 *                           host memory.
 * @param[out]     dst       Destination pointer.  Semantics
 *                           mirror @p src.
 * @param[in]      is_pinned Whether the host-side pointer is
 *                           page-locked (pinned).  Affects
 *                           whether the DMA engine can be used.
 * @param[in]      kind      Copy direction (@ref TransferKind):
 *                           `HOST_TO_DEVICE`,
 *                           `DEVICE_TO_HOST`, or
 *                           `DEVICE_TO_DEVICE`.
 * @param[in]      bytes     Number of bytes to copy.
 *
 * @return @ref DeviceStatus with `code == 0` on success, or an
 *         error status with `code == -1` and a message.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes each.
 * @pre  @p kind must match the actual memory types of @p src
 *       and @p dst.
 *
 * @see transfer_to()  High-level tensor transfer (uses this
 *      internally).
 */
DeviceStatus device_memcpy_c(const void *src, void *dst, bool is_pinned,
                             TransferKind kind, size_t bytes);

#ifdef __cplusplus
}
#endif
