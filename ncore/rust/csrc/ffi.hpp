/**
 * @file ffi.hpp
 * @brief C++-side FFI declarations for device-agnostic memory management.
 *
 * Provides the C++ type definitions (DeviceBuffer_t, DeviceStatus_t,
 * DeviceMemcpyKind) and the extern-"C" function declarations for
 * device_reserve, device_release, and device_memcpy.  The corresponding
 * C-callable wrapper device_memcpy_c (which uses TransferKind instead of
 * DeviceMemcpyKind) is declared in cpp_ffi.h.
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
 * @brief Allocate a device or pinned-host buffer through the active
 *        backend (CUDA or HIP).
 *
 * The chosen backend is determined by the @p kind parameter.  The
 * requested size is rounded up to the next multiple of @p align when
 * @p align > 1.
 *
 * @param bytes   Minimum number of bytes to allocate.
 * @param out_buf Output buffer descriptor (valid only when the returned
 *                status has code == 0).
 * @param pinned  If true, allocate page-locked host memory; otherwise
 *                allocate device memory.
 * @param align   Alignment requirement (must be a power of two).
 * @param kind    Target backend (DeviceCUDA or DeviceHIP).
 * @return DeviceStatus_t with code 0 on success, or a positive error
 *         code with a descriptive message on failure.
 */
DeviceStatus_t device_reserve(std::size_t bytes, DeviceBuffer_t *out_buf,
                              bool pinned = false, std::size_t align = 512,
                              DeviceKind_t kind = DeviceKind_t::DeviceCUDA);

/**
 * @brief Free a buffer previously allocated with device_reserve.
 *
 * Dispatches to cuda_release or hip_release based on the buffer's
 * device_kind field.  The buffer descriptor is zeroed on success.
 *
 * @param buf Pointer to the buffer descriptor to free.  If buf or
 *            buf->ptr is NULL the function returns an error status.
 * @return DeviceStatus_t with code 0 on success, or a positive error
 *         code with a descriptive message on failure.
 */
DeviceStatus_t device_release(DeviceBuffer_t *buf);

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
