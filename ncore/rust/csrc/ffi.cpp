/**
 * @file ffi.cpp
 * @brief Dispatch layer that routes memory requests to the CUDA or HIP
 *        backend at run time.
 *
 * device_reserve instantiates a backend-specific buffer descriptor on the
 * stack, calls the corresponding reserve function, and copies the result
 * into the generic DeviceBuffer_t / DeviceStatus_t structures.
 * device_release reads the device_kind field and calls the matching backend
 * free routine. device_memcpy queries the active backend and forwards the
 * copy request without exposing CUDA or HIP runtime enums to callers.
 * device_memcpy_c is an extern-"C" wrapper that accepts the C-side
 * TransferKind enum and converts it to DeviceMemcpyKind before
 * dispatching, making it directly usable from pure-C translation units.
 */

#include "ffi.hpp"
#include "device/cuda/cuda_allocator.hpp"
#include "device/cuda/cuda_io.hpp"
#include "device/hip/hip_allocator.hpp"
#include "device/hip/hip_io.hpp"
#include <cstdlib>
#include <cstring>
#include <ncore/cpp_ffi.h>

/**
 * @brief Template helper that calls a backend-specific reserve function
 *        and fills the generic DeviceBuffer_t / DeviceStatus_t outputs.
 *
 * @tparam BufKind     Backend-specific buffer type (CudaBuffer_t or
 *                     HipBuffer_t).
 * @tparam StatusKind  Backend-specific status type (CudaStatus_t or
 *                     HipStatus_t).
 * @tparam func_kind   Pointer to the backend's reserve function (cuda_reserve
 *                     or hip_reserve).
 * @tparam DeviceKind  DeviceKind_t enumerator identifying the active backend.
 *
 * @param bytes   Number of bytes to allocate.
 * @param pinned  Whether to allocate pinned (page-locked) host memory.
 * @param align   Alignment requirement.
 * @param[out] dbuf    Generic device-buffer descriptor to fill.
 * @param[out] buf     Backend-specific buffer descriptor (also filled).
 * @param[out] dstatus Generic status descriptor to fill.
 * @param[out] status  Backend-specific status descriptor (also filled).
 */
template <typename BufKind, typename StatusKind, auto func_kind,
          auto DeviceKind>
constexpr static void
device_reserve_dispatch(std::size_t bytes, bool pinned, std::size_t align,
                        DeviceBuffer_t *dbuf, BufKind *buf,
                        DeviceStatus_t *dstatus, StatusKind *status) {
  *status = func_kind(bytes, align, pinned, buf);

  dstatus->code = status->code;
  dstatus->message = status->msg;

  dbuf->ptr = buf->ptr;
  dbuf->bytes = buf->bytes;
  dbuf->is_pinned = buf->is_pinned;
  dbuf->device_kind = DeviceKind;
  dbuf->device_buf_ptr = buf;
}

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
                              bool pinned, std::size_t align,
                              DeviceKind_t kind) {
  DeviceStatus_t dstatus = {};
  if (kind == DeviceKind_t::DeviceCUDA) {
    CudaBuffer_t cbuf = {};
    CudaStatus_t cstatus = {};
    device_reserve_dispatch<CudaBuffer_t, CudaStatus_t, cuda_reserve,
                            DeviceKind_t::DeviceCUDA>(
        bytes, pinned, align, out_buf, &cbuf, &dstatus, &cstatus);
  } else {
    HipBuffer_t hbuf = {};
    HipStatus_t hstatus = {};
    device_reserve_dispatch<HipBuffer_t, HipStatus_t, hip_reserve,
                            DeviceKind_t::DeviceHIP>(
        bytes, pinned, align, out_buf, &hbuf, &dstatus, &hstatus);
  }
  return dstatus;
}

/**
 * @brief Free a buffer previously allocated with device_reserve.
 *
 * Dispatches to cuda_release or hip_release based on the buffer's
 * device_kind field.
 *
 * @param buf Pointer to the buffer descriptor to free.  The descriptor
 *            is zeroed out on success.
 * @return DeviceStatus_t with code 0 on success, or a positive error code
 *         with a descriptive message on failure.
 */
DeviceStatus_t device_release(DeviceBuffer_t *buf) {
  DeviceStatus_t dstatus = {};
  if (buf->device_kind == DeviceKind_t::DeviceCUDA) {
    CudaStatus_t cstatus =
        cuda_release(static_cast<CudaBuffer_t *>(buf->device_buf_ptr));
    dstatus.code = cstatus.code;
    dstatus.message = cstatus.msg;
  } else {
    HipStatus_t hstatus =
        hip_release(static_cast<HipBuffer_t *>(buf->device_buf_ptr));
    dstatus.code = hstatus.code;
    dstatus.message = hstatus.msg;
  }

  return dstatus;
}

/**
 * @brief Copy bytes using the currently active GPU backend.
 *
 * CUDA is used when get_device_backend() reports DeviceCUDA; HIP is used
 * when it reports DeviceHIP. If no backend is available, the returned status
 * has code -1 and a descriptive message.
 *
 * @param src       Source pointer.
 * @param dst       Destination pointer.
 * @param is_pinned Whether the host-side buffer is pinned/page-locked.
 * @param kind      Device-agnostic copy direction.
 * @param bytes     Number of bytes to copy.
 * @return DeviceStatus_t populated from the selected backend status.
 */
DeviceStatus_t device_memcpy(const void *src, void *dst, bool is_pinned,
                             DeviceMemcpyKind kind, std::size_t bytes) {
  DeviceStatus_t status = {};
  DeviceKind_t device = get_device_backend();

  if (device == DeviceKind_t::DeviceCUDA) {
    CudaStatus_t cstatus = cuda_memcpy(bytes, kind, src, dst, is_pinned);
    status.code = cstatus.code;
    status.message = cstatus.msg;
    return status;
  }
  if (device == DeviceKind_t::DeviceHIP) {
    HipStatus_t hstatus = hip_memcpy(bytes, kind, src, dst, is_pinned);
    status.code = hstatus.code;
    status.message = hstatus.msg;
    return status;
  }
  status.code = -1;
  status.message = "No device was found";
  return status;
}

/**
 * @brief Copy bytes through the active GPU backend (C-callable wrapper).
 *
 * Dispatches to cuda_memcpy or hip_memcpy according to get_device_backend().
 * Converts the C-side TransferKind enum to DeviceMemcpyKind for the
 * C++ backend functions.  If no backend is available the returned status
 * has code -1 and a descriptive message.
 *
 * @param src       Source pointer.
 * @param dst       Destination pointer.
 * @param is_pinned Whether the host-side pointer is pinned/page-locked.
 * @param kind      Device-agnostic copy direction (TransferKind).
 * @param bytes     Number of bytes to copy.
 * @return DeviceStatus with code 0 on success, or an error status.
 */
DeviceStatus device_memcpy_c(const void *src, void *dst, bool is_pinned,
                             TransferKind kind, size_t bytes) {
  DeviceStatus status = {};
  DeviceKind_t device = get_device_backend();

  if (device == DeviceKind_t::DeviceCUDA) {
    CudaStatus_t cstatus = cuda_memcpy(
        bytes, static_cast<DeviceMemcpyKind>(kind), src, dst, is_pinned);
    status.code = cstatus.code;
    status.message = cstatus.msg;
    return status;
  }
  if (device == DeviceKind_t::DeviceHIP) {
    HipStatus_t hstatus = hip_memcpy(bytes, static_cast<DeviceMemcpyKind>(kind),
                                     src, dst, is_pinned);
    status.code = hstatus.code;
    status.message = hstatus.msg;
    return status;
  }
  status.code = -1;
  status.message = "No device was found";
  return status;
}
