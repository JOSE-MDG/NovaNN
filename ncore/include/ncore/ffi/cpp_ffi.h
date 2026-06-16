/**
 * @file cpp_ffi.h
 * @brief C-callable bridge to the C++ device-agnostic FFI layer.
 *
 * @details
 * Provides an `extern "C"` interface so that pure-C translation
 * units can perform GPU memory transfers without linking against
 * C++ symbols directly.  The C-side @ref TransferKind enum is
 * translated to the corresponding C++ enum internally, and the
 * appropriate backend (CUDA or HIP) is selected at runtime.
 *
 * ## Error Handling
 *
 * Returns a @ref DeviceStatus struct with a numeric code and a
 * human-readable message.  A code of `0` indicates success;
 * negative values indicate backend-level errors.
 */

#pragma once

#include <ncore/core/device.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * C-callable wrapper that dispatches to the correct GPU backend
 * based on runtime detection.  The @ref TransferKind parameter
 * specifies the copy direction.
 *
 * This function does not perform any allocation; both @p src and
 * @p dst must point to valid memory regions of at least @p bytes.
 *
 * @param[in]  src    Source pointer.  Must be a valid host or device
 *                    memory region.
 * @param[out] dst    Destination pointer.  Must be a valid host or
 *                    device memory region.
 * @param[in]  kind   Copy direction (@ref TransferKind).
 * @param[in]  bytes  Number of bytes to copy.
 *
 * @return @ref DeviceStatus with `code == 0` on success.
 *
 * @retval {0, "ok"}           Transfer completed successfully.
 * @retval {-1, "..."}         Backend error or no device found.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src and
 *       @p dst.
 */
extern DeviceStatus device_memcpy_c(const void *src, void *dst,
                                    TransferKind kind, size_t bytes);

#ifdef __cplusplus
}
#endif
