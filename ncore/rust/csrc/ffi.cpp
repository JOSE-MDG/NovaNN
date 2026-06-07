/**
 * @file ffi.cpp
 * @brief Device-agnostic FFI dispatch implementation.
 *
 * @details
 * Implements the top-level `extern "C"` functions declared in
 * `ffi.hpp`.  Each function dispatches to the correct backend
 * (CUDA or HIP) based on the @ref DeviceKind_t tag stored in
 * the buffer descriptor or detected at runtime.
 *
 * Backend availability is gated at two levels:
 * 1. **Compile-time**: `NOVA_HAS_CUDA` / `NOVA_HAS_HIP`
 *    preprocessor macros control whether backend code is
 *    compiled.
 * 2. **Runtime**: @ref get_device_backend() probes CUDA and
 *    HIP runtime availability.
 *
 * ## Dispatch Strategy
 *
 * - **Allocation** (`device_reserve`): The caller specifies the
 *   target backend via the @p kind parameter.
 * - **Release and resize**: The backend is read from the buffer
 *   descriptor's @ref DeviceBuffer_t::device_kind field.
 * - **Memcpy**: The backend is auto-detected via
 *   @ref get_device_backend().
 *
 * @see ffi.hpp            Function declarations and type definitions.
 * @see device/admin.cpp   Runtime backend detection.
 * @see device/cuda/       CUDA backend implementation.
 * @see device/hip/        HIP backend implementation.
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
 * @brief Internal template that dispatches buffer allocation to a
 *        specific backend.
 *
 * @details
 * Allocates a backend-specific buffer descriptor via
 * `std::make_unique`, calls the backend allocator, and transfers
 * ownership to @p dbuf->device_buf_ptr via `buf.release()`.
 *
 * @tparam BufKind      Backend buffer type (`CudaBuffer_t` or
 *                      `HipBuffer_t`).
 * @tparam StatusKind   Backend status type.
 * @tparam func_kind    Backend allocator function pointer.
 * @tparam DeviceKind   The `DeviceKind_t` value for this backend.
 *
 * @param[in]  bytes  Requested allocation size.
 * @param[in]  pinned Whether to allocate page-locked memory.
 * @param[in]  align  Alignment in bytes.
 * @param[out] dbuf   Device buffer descriptor to populate.
 * @param[out] dstatus  Status to populate on error.
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
 * @brief Allocate a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to @ref device_reserve_dispatch with the correct
 * backend types based on @p kind.  Compile-time gated via
 * `NOVA_HAS_CUDA` / `NOVA_HAS_HIP`.
 *
 * @param[in]  bytes  Requested allocation size.
 * @param[out] out    Buffer descriptor to populate.
 * @param[in]  pinned Whether to allocate page-locked memory.
 * @param[in]  align  Alignment in bytes.
 * @param[in]  kind   Target backend (CUDA or HIP).
 *
 * @return Status with `code == 0` on success.
 *
 * @pre  @p bytes > 0.
 * @post On success, @p out->device_buf_ptr owns a backend
 *       descriptor.
 *
 * @see device_release()  Frees the buffer.
 * @see device_resize()   Resizes the buffer.
 */
DeviceStatus_t device_reserve(std::size_t bytes, DeviceBuffer_t *out_buf,
                              bool pinned, std::size_t align,
                              DeviceKind_t kind) {
  DeviceStatus_t dstatus = {};
  if (kind == DeviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    device_reserve_dispatch<CudaBuffer_t, CudaStatus_t, cuda_reserve,
                            DeviceKind_t::DeviceCUDA>(bytes, pinned, align,
                                                      out_buf, &dstatus);
#else
    dstatus.code = -1;
    dstatus.message = "CUDA backend was not built";
#endif
  } else if (kind == DeviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
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
 * @brief Free a GPU or pinned-host memory buffer.
 *
 * @details
 * Casts @p buf->device_buf_ptr to the backend type, calls the
 * backend release, and zeroes @p buf on success.
 *
 * @param[in,out] buf  Buffer descriptor to free.
 *
 * @return Status with `code == 0` on success.
 *
 * @post On success, @p buf is zeroed.
 *
 * @see device_reserve()  Allocates the buffer freed here.
 */
DeviceStatus_t device_release(DeviceBuffer_t *buf) {
  DeviceStatus_t dstatus = {};
  if (buf->device_kind == DeviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
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
#ifdef NOVA_HAS_HIP
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
 * @brief Resize a GPU or pinned-host memory buffer.
 *
 * @details
 * Casts @p buf->device_buf_ptr to the backend type and calls the
 * backend resize.  On success, updates @p buf->ptr and
 * @p buf->bytes.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     align     Alignment in bytes.
 *
 * @return Status with `code == 0` on success.
 *
 * @post On success, @p buf->ptr and @p buf->bytes reflect the
 *       new allocation.
 *
 * @warning On failure the original buffer may be in an
 *          inconsistent state.
 *
 * @see device_reserve()  Initial allocation.
 * @see device_release()  Explicit deallocation.
 */
DeviceStatus_t device_resize(DeviceBuffer_t *buf, std::size_t new_bytes,
                             std::size_t align) {
  DeviceStatus_t dstatus = {};
  if (buf->device_kind == DeviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    auto *backend_buf = static_cast<CudaBuffer_t *>(buf->device_buf_ptr);
    CudaStatus_t cstatus = cuda_resize(backend_buf, new_bytes, align);
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
#ifdef NOVA_HAS_HIP
    auto *backend_buf = static_cast<HipBuffer_t *>(buf->device_buf_ptr);
    HipStatus_t hstatus = hip_resize(backend_buf, new_bytes, align);
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
 * @brief Copy memory between host and device.
 *
 * @details
 * Auto-detects the backend via @ref get_device_backend() and
 * dispatches to the appropriate backend memcpy.
 *
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  is_pinned Whether the host-side pointer is
 *                      page-locked.
 * @param[in]  kind      Copy direction.
 * @param[in]  bytes     Number of bytes.
 *
 * @return Status with `code == 0` on success.
 *
 * @see device_memcpy_c()  C-callable variant.
 */
DeviceStatus_t device_memcpy(const void *src, void *dst, bool is_pinned,
                             DeviceMemcpyKind kind, std::size_t bytes) {
  DeviceStatus_t status = {};
  DeviceKind_t device = get_device_backend();

  if (device == DeviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
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
#ifdef NOVA_HAS_HIP
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
 * @brief C-callable wrapper for device memcpy.
 *
 * @details
 * Same as @ref device_memcpy but accepts a @ref TransferKind
 * (from `ncore/cpp_ffi.h`) instead of @ref DeviceMemcpyKind,
 * casting between the two.
 *
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  is_pinned Whether the host-side pointer is
 *                      page-locked.
 * @param[in]  kind      Copy direction (@ref TransferKind).
 * @param[in]  bytes     Number of bytes.
 *
 * @return Status with `code == 0` on success.
 *
 * @see device_memcpy()  C++ variant.
 */
DeviceStatus device_memcpy_c(const void *src, void *dst, bool is_pinned,
                             TransferKind kind, size_t bytes) {
  DeviceStatus status = {};
  DeviceKind_t device = get_device_backend();

  if (device == DeviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
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
#ifdef NOVA_HAS_HIP
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
