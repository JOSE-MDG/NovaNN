/**
 * @file ffi.cpp
 * @brief Dispatch layer that routes memory requests to the CUDA or HIP
 *        backend at run time.
 *
 * device_reserve allocates a backend-specific buffer descriptor, calls the
 * corresponding reserve function, and copies the result into the generic
 * DeviceBuffer_t / DeviceStatus_t structures.
 * device_realloc dispatches to cuda_realloc or hip_realloc based on
 * the buffer's device_kind and updates the generic descriptor on success.
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
#include <memory>
#include <ncore/cpp_ffi.h>

/**
 * @brief Template helper that calls a backend-specific reserve function
 *        and fills the generic DeviceBuffer_t / DeviceStatus_t outputs.
 *
 * The backend buffer is heap-allocated via a unique_ptr.  On success
 * ownership is released into dbuf->device_buf_ptr (a raw owning pointer
 * whose lifetime is managed by device_release).  On failure the
 * unique_ptr destroys the backend buffer automatically.
 *
 * @tparam BufKind     Backend-specific buffer type (CudaBuffer_t or
 *                     HipBuffer_t).
 * @tparam StatusKind  Backend-specific status type (CudaStatus_t or
 *                     HipStatus_t).
 * @tparam func_kind   Pointer to the backend's reserve function (cuda_reserve
 *                     or hip_reserve).
 * @tparam DeviceKind  DeviceKind_t enumerator identifying the active backend.
 *
 * @param bytes        Number of bytes to allocate.
 * @param pinned       Whether to allocate pinned (page-locked) host memory.
 * @param align        Alignment requirement.
 * @param[out] dbuf    Generic device-buffer descriptor to fill.
 * @param[out] dstatus Generic status descriptor to fill.
 */
template <typename BufKind, typename StatusKind, auto func_kind,
          auto DeviceKind>
constexpr static void
device_reserve_dispatch(std::size_t bytes, bool pinned, std::size_t align,
                        DeviceBuffer_t *dbuf, DeviceStatus_t *dstatus) {
  auto buf = std::make_unique<BufKind>();
  StatusKind status = func_kind(bytes, align, pinned, buf.get());

  dstatus->code = status.code;
  dstatus->message = status.msg;

  if (status.code != 0) {
    dbuf->ptr = nullptr;
    dbuf->bytes = 0;
    dbuf->is_pinned = false;
    dbuf->device_kind = DeviceKind_t::DeviceNull;
    dbuf->device_buf_ptr = nullptr;
    return;
  }

  dbuf->ptr = buf->ptr;
  dbuf->bytes = buf->bytes;
  dbuf->is_pinned = buf->is_pinned;
  dbuf->device_kind = DeviceKind;
  dbuf->device_buf_ptr = buf.release();
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
#if NOVA_CUDA
    device_reserve_dispatch<CudaBuffer_t, CudaStatus_t, cuda_reserve,
                            DeviceKind_t::DeviceCUDA>(bytes, pinned, align,
                                                      out_buf, &dstatus);
#else
    dstatus.code = -1;
    dstatus.message = "CUDA backend was not built";
#endif
  } else if (kind == DeviceKind_t::DeviceHIP) {
#if NOVA_HIP
    device_reserve_dispatch<HipBuffer_t, HipStatus_t, hip_reserve,
                            DeviceKind_t::DeviceHIP>(bytes, pinned, align,
                                                     out_buf, &dstatus);
#else
    dstatus.code = -1;
    dstatus.message = "HIP backend was not built";
#endif
  } else {
    dstatus.code = -1;
    dstatus.message = "No device was found";
  }
  return dstatus;
}

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
DeviceStatus_t device_release(DeviceBuffer_t *buf) {
  DeviceStatus_t dstatus = {};
  if (buf->device_kind == DeviceKind_t::DeviceCUDA) {
#if NOVA_CUDA
    auto backend_buf = std::unique_ptr<CudaBuffer_t>(
        static_cast<CudaBuffer_t *>(buf->device_buf_ptr));
    CudaStatus_t cstatus = cuda_release(backend_buf.get());
    dstatus.code = cstatus.code;
    dstatus.message = cstatus.msg;
#else
    dstatus.code = -1;
    dstatus.message = "CUDA backend was not built";
#endif
  } else if (buf->device_kind == DeviceKind_t::DeviceHIP) {
#if NOVA_HIP
    auto backend_buf = std::unique_ptr<HipBuffer_t>(
        static_cast<HipBuffer_t *>(buf->device_buf_ptr));
    HipStatus_t hstatus = hip_release(backend_buf.get());
    dstatus.code = hstatus.code;
    dstatus.message = hstatus.msg;
#else
    dstatus.code = -1;
    dstatus.message = "HIP backend was not built";
#endif
  } else {
    dstatus.code = -1;
    dstatus.message = "No device was found";
  }

  if (dstatus.code == 0) {
    *buf = DeviceBuffer_t{};
  }

  return dstatus;
}

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
DeviceStatus_t device_realloc(DeviceBuffer_t *buf, std::size_t new_bytes,
                              std::size_t align) {
  DeviceStatus_t dstatus = {};
  if (buf->device_kind == DeviceKind_t::DeviceCUDA) {
#if NOVA_CUDA
    auto *backend_buf = static_cast<CudaBuffer_t *>(buf->device_buf_ptr);
    CudaStatus_t cstatus = cuda_realloc(backend_buf, new_bytes, align);
    dstatus.code = cstatus.code;
    dstatus.message = cstatus.msg;
    if (dstatus.code == 0) {
      buf->ptr = backend_buf->ptr;
      buf->bytes = backend_buf->bytes;
    }
#else
    dstatus.code = -1;
    dstatus.message = "CUDA backend was not built";
#endif
  } else if (buf->device_kind == DeviceKind_t::DeviceHIP) {
#if NOVA_HIP
    auto *backend_buf = static_cast<HipBuffer_t *>(buf->device_buf_ptr);
    HipStatus_t hstatus = hip_realloc(backend_buf, new_bytes, align);
    dstatus.code = hstatus.code;
    dstatus.message = hstatus.msg;
    if (dstatus.code == 0) {
      buf->ptr = backend_buf->ptr;
      buf->bytes = backend_buf->bytes;
    }
#else
    dstatus.code = -1;
    dstatus.message = "HIP backend was not built";
#endif
  } else {
    dstatus.code = -1;
    dstatus.message = "No device was found";
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
#if NOVA_CUDA
    CudaStatus_t cstatus = cuda_memcpy(bytes, kind, src, dst, is_pinned);
    status.code = cstatus.code;
    status.message = cstatus.msg;
    return status;
#else
    status.code = -1;
    status.message = "CUDA backend was not built";
    return status;
#endif
  }
  if (device == DeviceKind_t::DeviceHIP) {
#if NOVA_HIP
    HipStatus_t hstatus = hip_memcpy(bytes, kind, src, dst, is_pinned);
    status.code = hstatus.code;
    status.message = hstatus.msg;
    return status;
#else
    status.code = -1;
    status.message = "HIP backend was not built";
    return status;
#endif
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
#if NOVA_CUDA
    CudaStatus_t cstatus = cuda_memcpy(
        bytes, static_cast<DeviceMemcpyKind>(kind), src, dst, is_pinned);
    status.code = cstatus.code;
    status.message = cstatus.msg;
    return status;
#else
    status.code = -1;
    status.message = "CUDA backend was not built";
    return status;
#endif
  }
  if (device == DeviceKind_t::DeviceHIP) {
#if NOVA_HIP
    HipStatus_t hstatus = hip_memcpy(bytes, static_cast<DeviceMemcpyKind>(kind),
                                     src, dst, is_pinned);
    status.code = hstatus.code;
    status.message = hstatus.msg;
    return status;
#else
    status.code = -1;
    status.message = "HIP backend was not built";
    return status;
#endif
  }
  status.code = -1;
  status.message = "No device was found";
  return status;
}
