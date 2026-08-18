/**
 * @file ffi.cpp
 * @brief Device-agnostic FFI dispatch implementation.
 *
 * @details
 * Implements the top-level @c extern "C" functions declared in
 * @c ffi.hpp.  Each function dispatches to the correct backend
 * (CUDA or HIP) based on the @ref deviceKind_t tag stored in
 * the buffer descriptor or detected at runtime.
 *
 * Backend availability is gated at two levels:
 * @li 1. Compile-time — @c NOVA_HAS_CUDA / @c NOVA_HAS_HIP
 *    preprocessor macros control whether backend code is
 *    compiled.
 * @li 2. Runtime — @ref getDeviceBackend() probes CUDA and
 *    HIP runtime availability.
 *
 * @section dispatch-strategy Dispatch Strategy
 *
 * @li Allocation (@c deviceReserve) — The caller specifies the
 *   target backend via the @p kind parameter.
 * @li Release and resize — The backend is read from the buffer
 *   descriptor's @ref deviceBuffer_t::deviceKind field.
 * @li Transfer — The backend is auto-detected via
 *   @ref getDeviceBackend().
 *
 * @see ffi.hpp             Function declarations and type definitions.
 * @see admin.cpp           Runtime backend detection.
 * @see ncore/native/cuda/  CUDA backend implementation.
 * @see ncore/native/hip/   HIP backend implementation.
 */

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

#include <ncore/core/device.h>
#include <ncore/core/status.h>

#ifdef NOVA_HAS_CUDA
#include "CudaAllocator.hpp"
#include "CudaIO.hpp"
#endif
#ifdef NOVA_HAS_HIP
#include "HipAllocator.hpp"
#include "HipIO.hpp"
#endif

#include "ffi.hpp"

namespace {
/**
 * @brief Internal template that dispatches buffer allocation to a
 *        specific backend.
 *
 * @details
 * Allocates a backend-specific buffer descriptor via
 * @c std::make_unique, calls the backend allocator, and transfers
 * ownership to @p dbuf->deviceBufPtr via @c buf.release().
 *
 * @tparam BufKind      Backend buffer type (@c cudaBuffer_t or
 *                      @c hipBuffer_t).
 * @tparam funcKind     Backend allocator function pointer.
 * @tparam DeviceKind   The @c deviceKind_t value for this backend.
 *
 * @param[in]  bytes  Requested allocation size.
 * @param[in]  pinned Whether to allocate page-locked memory.
 * @param[out] dbuf   Device buffer descriptor to populate.
 * @param[out] status  Status to populate on error.
 */
template <typename BufKind, auto funcKind, auto DeviceKind>
constexpr void deviceReserveDispatch(std::size_t bytes, bool pinned,
                                     deviceBuffer_t *dbuf,
                                     novaStatus_t *status) {
  auto buf = std::unique_ptr<BufKind>(new (std::nothrow) BufKind());
  if (!buf) {
    *status = {.err = novaOutOfMemory,
               .message = "Unable to allocate a device buffer descriptor"};
    *dbuf = deviceBuffer_t{};
    return;
  }
  novaStatus_t dstatus = funcKind(bytes, pinned, buf.get());

  status->err = dstatus.err;
  status->message = dstatus.message;

  if (dstatus.err != novaSuccess) {
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

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
constexpr DeviceMemcpyKind
mapCTransferKind2DeviceMemcpyKind(TransferKind kind) noexcept {
  switch (kind) {
  case deviceMemcpyHostToDevice:
    return DeviceMemcpyKind::deviceMemcpyHostToDevice;
  case deviceMemcpyDeviceToHost:
    return DeviceMemcpyKind::deviceMemcpyDeviceToHost;
  default:
    return DeviceMemcpyKind::deviceMemcpyDeviceToDevice;
  }
}
#endif

} // namespace

/**
 * @brief Allocate a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to @ref deviceReserveDispatch with the correct
 * backend types based on @p kind.  Compile-time gated via
 * @c NOVA_HAS_CUDA / @c NOVA_HAS_HIP.
 *
 * @param[in]  bytes   Requested allocation size.
 * @param[out] out_buf Buffer descriptor to populate.
 * @param[in]  pinned  Whether to allocate page-locked memory.
 * @param[in]  kind    Target backend (CUDA or HIP).
 *
 * @return Status with @c err == novaSuccess on success.
 *
 * @pre  @p bytes > 0.
 * @post On success, @p out_buf->deviceBufPtr owns a backend
 *       descriptor.
 *
 * @see deviceRelease()  Frees the buffer.
 * @see deviceResize()   Resizes the buffer.
 */
novaStatus_t deviceReserve(std::size_t bytes, deviceBuffer_t *out_buf,
                           bool pinned, deviceKind_t kind) {
  if (out_buf == nullptr) {
    return {.err = novaInvalidPointer,
            .message = "deviceReserve requires a valid output buffer"};
  }
  novaStatus_t status = {};
  if (kind == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    deviceReserveDispatch<cudaBuffer_t, cudaReserve, deviceKind_t::DeviceCUDA>(
        bytes, pinned, out_buf, &status);
#else
    status.err = novaBackendNotCompiled;
    status.message = "CUDA support is not available in this build";
#endif
  } else if (kind == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    deviceReserveDispatch<hipBuffer_t, hipReserve, deviceKind_t::DeviceHIP>(
        bytes, pinned, out_buf, &status);
#else
    status.err = novaBackendNotCompiled;
    status.message = "HIP support is not available in this build";
#endif
  } else {
    status.err = novaDeviceNotAvailable;
    status.message = "The requested device kind is not recognized";
  }
  return status;
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
 * @return Status with @c err == novaSuccess on success.
 *
 * @post On success, @p buf is zeroed.
 *
 * @see deviceReserve()  Allocates the buffer freed here.
 */
novaStatus_t deviceRelease(deviceBuffer_t *buf) {
  if (buf == nullptr) {
    return {.err = novaInvalidPointer,
            .message = "deviceRelease requires a valid buffer"};
  }
  novaStatus_t status = {};
  if (buf->deviceKind == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    auto backendBuf = std::unique_ptr<cudaBuffer_t>(
        static_cast<cudaBuffer_t *>(buf->deviceBufPtr));
    novaStatus_t cstatus = cudaRelease(backendBuf.get());
    status.err = cstatus.err;
    status.message = cstatus.message;
#else
    status.err = novaBackendNotCompiled;
    status.message = "CUDA support is not available in this build";
#endif
  } else if (buf->deviceKind == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    auto backendBuf = std::unique_ptr<hipBuffer_t>(
        static_cast<hipBuffer_t *>(buf->deviceBufPtr));
    novaStatus_t hstatus = hipRelease(backendBuf.get());
    status.err = hstatus.err;
    status.message = hstatus.message;
#else
    status.err = novaBackendNotCompiled;
    status.message = "HIP support is not available in this build";
#endif
  } else {
    status.err = novaDeviceNotAvailable;
    status.message = "This buffer belongs to an unrecognized GPU backend";
  }

  if (status.err == novaSuccess) {
    *buf = deviceBuffer_t{};
  }

  return status;
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
 *
 * @return Status with @c err == novaSuccess on success.
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
novaStatus_t deviceResize(deviceBuffer_t *buf, std::size_t new_bytes) {
  if (buf == nullptr) {
    return {.err = novaInvalidPointer,
            .message = "deviceResize requires a valid buffer"};
  }
  novaStatus_t status = {};
  if (buf->deviceKind == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    auto *backendBuf = static_cast<cudaBuffer_t *>(buf->deviceBufPtr);
    novaStatus_t cstatus = cudaResize(backendBuf, new_bytes);
    status.err = cstatus.err;
    status.message = cstatus.message;
    if (status.err == novaSuccess) {
      buf->ptr = backendBuf->ptr;
      buf->bytes = backendBuf->bytes;
    }
#else
    status.err = novaBackendNotCompiled;
    status.message = "CUDA support is not available in this build";
#endif
  } else if (buf->deviceKind == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    auto *backendBuf = static_cast<hipBuffer_t *>(buf->deviceBufPtr);
    novaStatus_t hstatus = hipResize(backendBuf, new_bytes);
    status.err = hstatus.err;
    status.message = hstatus.message;
    if (status.err == novaSuccess) {
      buf->ptr = backendBuf->ptr;
      buf->bytes = backendBuf->bytes;
    }
#else
    status.err = novaBackendNotCompiled;
    status.message = "HIP support is not available in this build";
#endif
  } else {
    status.err = novaDeviceNotAvailable;
    status.message = "This buffer belongs to an unrecognized GPU backend";
  }
  return status;
}

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Auto-detects the active GPU backend via @ref getDeviceBackend()
 * and dispatches to the appropriate backend memcpy function.  The
 * copy direction is specified by @p kind.
 *
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  kind      Copy direction (@ref TransferKind).
 * @param[in]  bytes     Number of bytes to copy.
 *
 * @return @ref novaStatus_t with @c err == novaSuccess on success.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src
 *       and @p dst.
 */
novaStatus_t deviceTransfer(const void *src, void *dst, TransferKind kind,
                            size_t bytes) {
  novaStatus_t status = {};
  deviceKind_t device = getDeviceBackend();

  if (device == deviceKind_t::DeviceCUDA) {
#ifdef NOVA_HAS_CUDA
    novaStatus_t cstatus =
        cudaTransfer(bytes, mapCTransferKind2DeviceMemcpyKind(kind), src, dst);
    status.err = cstatus.err;
    status.message = cstatus.message;
    return status;
#else
    status.err = novaBackendNotCompiled;
    status.message = "CUDA support is not available in this build";
    return status;
#endif
  }
  if (device == deviceKind_t::DeviceHIP) {
#ifdef NOVA_HAS_HIP
    novaStatus_t hstatus =
        hipTransfer(bytes, mapCTransferKind2DeviceMemcpyKind(kind), src, dst);
    status.err = hstatus.err;
    status.message = hstatus.message;
    return status;
#else
    status.err = novaBackendNotCompiled;
    status.message = "HIP support is not available in this build";
    return status;
#endif
  }
  status.err = novaDeviceNotAvailable;
  status.message = "No GPU backend was detected on this system";
  return status;
}
