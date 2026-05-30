/**
 * @file ffi.hpp
 * @brief C++-side FFI declarations for device-agnostic memory management.
 *
 * Provides the C++ type definitions (DeviceBuffer_t, DeviceStatus_t,
 * DeviceMemcpyKind) and the extern-"C" function declarations for
 * device_reserve, device_resize, device_release, and device_memcpy.
 * The corresponding C-callable wrapper device_memcpy_c (which uses
 * TransferKind instead of DeviceMemcpyKind) is declared in cpp_ffi.h.
 */

#pragma once

#include "device/admin.hpp"
#include <cstddef>

/**
 * @brief Opaque descriptor for an allocated device buffer.
 *
 * @var ptr              Pointer to the device (or pinned host) memory.
 * @var bytes            Usable size of the allocation in bytes.
 * @var is_pinned        true if the buffer lives in pinned (page-locked)
 *                       host memory.
 * @var device_kind      Identifies the active backend (CUDA or HIP).
 * @var device_buf_ptr   Opaque pointer to the backend-specific buffer
 *                       descriptor (CudaBuffer_t or HipBuffer_t).
 */
struct DeviceBuffer_t {
  void *ptr = nullptr;
  std::size_t bytes = 0;
  bool is_pinned = false;
  DeviceKind_t device_kind = DeviceKind_t::DeviceCUDA;
  void *device_buf_ptr;
};

/**
 * @brief Result type returned by device management operations.
 *
 * @var code     Zero on success, a positive error code on failure.
 * @var message  Human-readable error description.
 */
struct DeviceStatus_t {
  int code = 0;
  const char *message = "ok";
};

/**
 * @brief Device-agnostic memory copy direction.
 *
 * Mirrors the host/device copy directions supported by CUDA and HIP while
 * keeping runtime-specific enums out of public csrc headers.
 *
 * @var deviceMemcpyHostToDevice Copy from host memory into device memory.
 * @var deviceMemcpyDeviceToHost Copy from device memory into host memory.
 * @var deviceMemcpyDeviceToDevice Copy between two device-memory pointers.
 */
enum class DeviceMemcpyKind : std::int8_t {
  deviceMemcpyHostToDevice = 1,
  deviceMemcpyDeviceToHost = 2,
  deviceMemcpyDeviceToDevice = 3
};

extern "C" {

/**
 * @brief Allocate a device or pinned-host buffer through the active backend.
 *
 * Dispatches to cuda_reserve or hip_reserve depending on @p kind.
 *
 * @param bytes   Minimum number of bytes to allocate.
 * @param out_buf [out] Output buffer descriptor (valid only when the
 *                      returned status has code == 0).
 * @param pinned  If true, allocate page-locked host memory.
 * @param align   Alignment requirement (power of two).
 * @param kind    Target backend (DeviceCUDA or DeviceHIP).
 * @return DeviceStatus_t with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
DeviceStatus_t device_reserve(std::size_t bytes, DeviceBuffer_t *out_buf,
                              bool pinned = false, std::size_t align = 512,
                              DeviceKind_t kind = DeviceKind_t::DeviceCUDA);

/**
 * @brief Free a buffer previously allocated with device_reserve.
 *
 * Dispatches to cuda_release or hip_release based on the buffer's
 * device_kind field.  The backend buffer descriptor (device_buf_ptr) is
 * deleted and the generic descriptor is zeroed on success.
 *
 * @param buf Pointer to the buffer descriptor to free.  The descriptor
 *            is zeroed out on success.
 * @return DeviceStatus_t with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
DeviceStatus_t device_release(DeviceBuffer_t *buf);

/**
 * @brief Reallocate a device or pinned-host buffer, preserving content.
 *
 * Dispatches to cuda_realloc or hip_realloc based on the buffer's
 * device_kind field.  Allocates a new buffer of @p new_bytes (rounded
 * up to @p align), copies the minimum of the old and new sizes, and
 * frees the old buffer.  On success the generic buffer descriptor is
 * updated with the new pointer and size; on failure it is left unchanged.
 *
 * @param buf       Pointer to the buffer descriptor to reallocate.
 *                  Must have been previously allocated with device_reserve.
 * @param new_bytes Target size in bytes.
 * @param align     Alignment requirement (must be a power of two).
 * @return DeviceStatus_t with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
DeviceStatus_t device_resize(DeviceBuffer_t *buf, std::size_t new_bytes,
                             std::size_t align);

/**
 * @brief Copy memory through the active backend.
 *
 * Dispatches to cuda_memcpy or hip_memcpy according to get_device_backend().
 * If the active backend is DeviceNull, an error status is returned.
 *
 * @param src       Source pointer.
 * @param dst       Destination pointer.
 * @param is_pinned Whether the host-side pointer is pinned/page-locked.
 * @param kind      Device-agnostic copy direction.
 * @param bytes     Number of bytes to copy.
 * @return DeviceStatus_t with code 0 on success, or a backend-specific error.
 */
DeviceStatus_t device_memcpy(const void *src, void *dst, bool is_pinned,
                             DeviceMemcpyKind kind, std::size_t bytes);
}
