/**
 * @file ffi.cpp
 * @brief Dispatch layer that routes allocation requests to the CUDA or HIP
 *        backend at run time.
 *
 * device_reserve instantiates a backend-specific buffer descriptor on the
 * stack, calls the corresponding reserve function, and copies the result
 * into the generic DeviceBuffer_t / DeviceStatus_t structures.
 * device_release reads the device_kind field and calls the matching backend
 * free routine.
 */

#include "ffi.hpp"
#include "device/admin.hpp"
#include "device/cuda/cuda_allocator.hpp"
#include "device/hip/hip_allocator.hpp"
#include <cstring>

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
  dstatus->message = std::string(status->msg);

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
