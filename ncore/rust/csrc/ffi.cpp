/**
 * @file ffi.cpp
 * @brief Device-agnostic FFI dispatch implementation.
 *
 * @details
 * Implements the top-level `extern "C"` functions declared in
 * `ffi.hpp`.  Each function dispatches to the correct backend
 * (CUDA or HIP) based on the @ref deviceKind_t tag stored in
 * the buffer descriptor or detected at runtime.
 *
 * Backend availability is gated at two levels:
 * 1. **Compile-time** — `NOVA_HAS_CUDA` / `NOVA_HAS_HIP`
 *    preprocessor macros control whether backend code is
 *    compiled.
 * 2. **Runtime** — @ref getDeviceBackend() probes CUDA and
 *    HIP runtime availability.
 *
 * ## Dispatch Strategy
 *
 * - **Allocation** (`deviceReserve`) — The caller specifies the
 *   target backend via the @p kind parameter.
 * - **Release and resize** — The backend is read from the buffer
 *   descriptor's @ref deviceBuffer_t::deviceKind field.
 * - **Memcpy** — The backend is auto-detected via
 *   @ref getDeviceBackend().
 *
 * @see ffi.hpp             Function declarations and type definitions.
 * @see device/admin.cpp    Runtime backend detection.
 * @see ncore/native/cuda/  CUDA backend implementation.
 * @see ncore/native/hip/   HIP backend implementation.
 */

#include "ffi.hpp"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ncore/ffi/cpp_ffi.h>
#ifdef NOVA_HAS_CUDA
#include "CudaAllocator.hpp"
#include "CudaIO.hpp"
#endif
#ifdef NOVA_HAS_HIP
#include "HipAllocator.hpp"
#include "HipIO.hpp"
#endif

namespace {
/**
 * @brief Internal template that dispatches buffer allocation to a
 *        specific backend.
 *
 * @details
 * Allocates a backend-specific buffer descriptor via
 * `std::make_unique`, calls the backend allocator, and transfers
 * ownership to @p dbuf->deviceBufPtr via `buf.release()`.
 *
 * @tparam BufKind      Backend buffer type (`cudaBuffer_t` or
 *                      `hipBuffer_t`).
 * @tparam StatusKind   Backend status type.
 * @tparam funcKind    Backend allocator function pointer.
 * @tparam DeviceKind   The `deviceKind_t` value for this backend.
 *
 * @param[in]  bytes  Requested allocation size.
 * @param[in]  pinned Whether to allocate page-locked memory.
 * @param[in]  align  Alignment in bytes.
 * @param[out] dbuf   Device buffer descriptor to populate.
 * @param[out] dstatus  Status to populate on error.
 */
template <typename BufKind, typename StatusKind, auto funcKind, auto DeviceKind>
constexpr void deviceReserveDispatch(std::size_t bytes, bool pinned,
                                     std::size_t align, deviceBuffer_t *dbuf,
                                     deviceStatus_t *dstatus) {
  auto buf = std::make_unique<BufKind>();
  StatusKind status = funcKind(bytes, align, pinned, buf.get());

  dstatus->code = status.code;
  dstatus->message = status.msg;

  if (status.code != 0) {
    dbuf->ptr = nullptr;
    dbuf->bytes = 0;
    dbuf->isPinned = false;
    dbuf->deviceKind = deviceKind_t::DeviceNull;
    dbuf->deviceBufPtr = nullptr;
    return;
  }

  dbuf->ptr = buf->ptr;
  dbuf->bytes = buf->bytes;
  dbuf->isPinned = buf->isPinned;
  dbuf->deviceKind = DeviceKind;
  dbuf->deviceBufPtr = buf.release();
}
} // namespace

/**
 * @brief Allocate a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to @ref deviceReserveDispatch with the correct
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
 * @post On success, @p out->deviceBufPtr owns a backend
 *       descriptor.
 *
 * @see deviceRelease()  Frees the buffer.
 * @see deviceResize()   Resizes the buffer.
 */
deviceStatus_t deviceReserve(std::size_t bytes, deviceBuffer_t *out_buf,
                             bool pinned, std::size_t align,
                             deviceKind_t kind) {
  deviceStatus_t dstatus = {};
  if (kind == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    deviceReserveDispatch<cudaBuffer_t, cudaStatus_t, cudaReserve,
                          deviceKind_t::DeviceCUDA>(bytes, pinned, align,
                                                    out_buf, &dstatus);
#else
    dstatus.code = -1;
    dstatus.message = "CUDA backend was not built";
#endif
  } else if (kind == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    deviceReserveDispatch<hipBuffer_t, hipStatus_t, hipReserve,
                          deviceKind_t::DeviceHIP>(bytes, pinned, align,
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
 * Casts @p buf->deviceBufPtr to the backend type, calls the
 * backend release, and zeroes @p buf on success.
 *
 * @param[in,out] buf  Buffer descriptor to free.
 *
 * @return Status with `code == 0` on success.
 *
 * @post On success, @p buf is zeroed.
 *
 * @see deviceReserve()  Allocates the buffer freed here.
 */
deviceStatus_t deviceRelease(deviceBuffer_t *buf) {
  deviceStatus_t dstatus = {};
  if (buf->deviceKind == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    auto backendBuf = std::unique_ptr<cudaBuffer_t>(
        static_cast<cudaBuffer_t *>(buf->deviceBufPtr));
    cudaStatus_t cstatus = cudaRelease(backendBuf.get());
    dstatus.code = cstatus.code;
    dstatus.message = cstatus.msg;
#else
    dstatus.code = -1;
    dstatus.message = "CUDA backend was not built";
#endif
  } else if (buf->deviceKind == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    auto backendBuf = std::unique_ptr<hipBuffer_t>(
        static_cast<hipBuffer_t *>(buf->deviceBufPtr));
    hipStatus_t hstatus = hipRelease(backendBuf.get());
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
    *buf = deviceBuffer_t{};
  }

  return dstatus;
}

/**
 * @brief Resize a GPU or pinned-host memory buffer.
 *
 * @details
 * Casts @p buf->deviceBufPtr to the backend type and calls the
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
 * @see deviceReserve()  Initial allocation.
 * @see deviceRelease()  Explicit deallocation.
 */
deviceStatus_t deviceResize(deviceBuffer_t *buf, std::size_t new_bytes,
                            std::size_t align) {
  deviceStatus_t dstatus = {};
  if (buf->deviceKind == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    auto *backendBuf = static_cast<cudaBuffer_t *>(buf->deviceBufPtr);
    cudaStatus_t cstatus = cudaResize(backendBuf, new_bytes, align);
    dstatus.code = cstatus.code;
    dstatus.message = cstatus.msg;
    if (dstatus.code == 0) {
      buf->ptr = backendBuf->ptr;
      buf->bytes = backendBuf->bytes;
    }
#else
    dstatus.code = -1;
    dstatus.message = "CUDA backend was not built";
#endif
  } else if (buf->deviceKind == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    auto *backendBuf = static_cast<hipBuffer_t *>(buf->deviceBufPtr);
    hipStatus_t hstatus = hipResize(backendBuf, new_bytes, align);
    dstatus.code = hstatus.code;
    dstatus.message = hstatus.msg;
    if (dstatus.code == 0) {
      buf->ptr = backendBuf->ptr;
      buf->bytes = backendBuf->bytes;
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
 * Auto-detects the backend via @ref getDeviceBackend() and
 * dispatches to the appropriate backend memcpy.
 *
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  kind      Copy direction.
 * @param[in]  bytes     Number of bytes.
 *
 * @return Status with `code == 0` on success.
 *
 * @see device_transfer_c()  C-callable variant.
 */
deviceStatus_t deviceTransfer(const void *src, void *dst, DeviceMemcpyKind kind,
                              std::size_t bytes) {
  deviceStatus_t status = {};
  deviceKind_t device = getDeviceBackend();

  if (device == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    cudaStatus_t cstatus = cudaTransfer(bytes, kind, src, dst);
    status.code = cstatus.code;
    status.message = cstatus.msg;
    return status;
#else
    status.code = -1;
    status.message = "CUDA backend was not built";
    return status;
#endif
  }
  if (device == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    hipStatus_t hstatus = hipTransfer(bytes, kind, src, dst);
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
 * Same as @ref deviceTransfer but accepts a @ref TransferKind
 * (from `ncore/cpp_ffi.h`) instead of @ref DeviceMemcpyKind,
 * casting between the two.
 *
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  kind      Copy direction (@ref TransferKind).
 * @param[in]  bytes     Number of bytes.
 *
 * @return Status with `code == 0` on success.
 *
 * @see deviceTransfer()  C++ variant.
 */
DeviceStatus device_transfer_c(const void *src, void *dst, TransferKind kind,
                               size_t bytes) {
  DeviceStatus status = {};
  deviceKind_t device = getDeviceBackend();

  if (device == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    cudaStatus_t cstatus =
        cudaTransfer(bytes, static_cast<DeviceMemcpyKind>(kind), src, dst);
    status.code = cstatus.code;
    status.message = cstatus.msg;
    return status;
#else
    status.code = -1;
    status.message = "CUDA backend was not built";
    return status;
#endif
  }
  if (device == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    hipStatus_t hstatus =
        hipTransfer(bytes, static_cast<DeviceMemcpyKind>(kind), src, dst);
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
