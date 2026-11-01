/**
 * @file ffi.cpp
 * @brief Device-agnostic FFI dispatch implementation.
 *
 * @details
 * Implements the top-level @c extern "C" functions declared in
 * @c ffi.hpp.  Each function resolves the backend through @ref opsFor
 * and forwards the request to the matching operation.
 *
 * Backend availability is gated at two levels: compile-time
 * (@c NOVA_HAS_CUDA / @c NOVA_HAS_HIP preprocessor macros) and
 * runtime (@ref getDeviceBackend() probes CUDA and HIP runtime
 * availability).  Dispatch strategy per operation: allocation
 * (@c deviceReserve) takes the target backend from the caller's
 * @p kind parameter; release and resize read the backend from the
 * buffer descriptor's @ref deviceBuffer_t::deviceKind field;
 * transfer auto-detects the backend via @ref getDeviceBackend().
 *
 * Every compiled backend contributes one @ref BackendOps table whose
 * entries are dispatch variants specialized to that backend.  A
 * variant owns the backend-specific casts, so no public entry point
 * names a backend type.
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
#include "cuda/memory/CudaAllocator.hpp"
#include "cuda/memory/CudaIO.hpp"
#endif
#ifdef NOVA_HAS_HIP
#include "hip/memory/HipAllocator.hpp"
#include "hip/memory/HipIO.hpp"
#endif

#include <ncore/memory/ffi.hpp>

namespace {
/**
 * @brief Dispatches a buffer allocation to one backend.
 *
 * @details
 * Allocates a backend-specific buffer descriptor with a nothrow
 * allocation, calls the backend allocator on the selected device, and
 * transfers ownership to @p dbuf->deviceBufPtr.
 *
 * @tparam BufKind         Backend buffer type (@c cudaBuffer_t or
 *                         @c hipBuffer_t).
 * @tparam StreamKind      Backend stream type (@c cudaStream_t or
 *                         @c hipStream_t).
 * @tparam funcKind        Backend allocator function.
 * @tparam currentDeviceId Backend query for the active device.
 * @tparam DeviceKind      The @c deviceKind_t value for this backend.
 *
 * @param[in]  bytes      Requested allocation size.
 * @param[in]  pinned     Whether to allocate page-locked memory.
 * @param[out] dbuf       Device buffer descriptor to populate.
 * @param[out] device_id  Receives the selected runtime device.
 * @param[in]  stream     Caller stream forwarded to the backend.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success.
 */
template <typename BufKind, typename StreamKind, auto funcKind,
          auto currentDeviceId, auto DeviceKind>
novaStatus_t deviceReserveDispatch(std::size_t bytes, bool pinned,
                                   deviceBuffer_t *dbuf,
                                   std::int32_t *device_id, void *stream) {
  auto buf = std::unique_ptr<BufKind>(new (std::nothrow) BufKind());
  if (!buf) {
    *dbuf = deviceBuffer_t{};
    *device_id = -1;
    return {.err = novaOutOfMemory,
            .message = "Unable to allocate a device buffer descriptor.\n"};
  }

  std::int32_t selectedDevice = -1;
  novaStatus_t status = currentDeviceId(&selectedDevice);
  if (status.err == novaSuccess) {
    status = funcKind(selectedDevice, bytes, pinned, buf.get(),
                      static_cast<StreamKind>(stream));
  }
  if (status.err != novaSuccess) {
    *dbuf = deviceBuffer_t{};
    *device_id = -1;
    return status;
  }

  dbuf->ptr = buf->ptr;
  dbuf->bytes = buf->bytes;
  dbuf->isPinned = buf->isPinned;
  dbuf->deviceKind = DeviceKind;
  dbuf->deviceBufPtr = buf.release();
  *device_id = selectedDevice;
  return status;
}

/**
 * @brief Dispatches a buffer free to one backend.
 *
 * @details
 * Casts the opaque descriptor to the backend type and calls the
 * backend release.  The descriptor is deleted once the backend accepts
 * the free, so the submitted free cannot be retried against the same
 * pointer.
 *
 * @tparam BufKind    Backend buffer type (@c cudaBuffer_t or
 *                    @c hipBuffer_t).
 * @tparam StreamKind Backend stream type (@c cudaStream_t or
 *                    @c hipStream_t).
 * @tparam funcKind   Backend release function.
 *
 * @param[in,out] buf       Buffer descriptor to free.
 * @param[in]     device_id Runtime device ordinal that owns the buffer.
 * @param[in]     stream    Caller stream forwarded to the backend.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success.
 */
template <typename BufKind, typename StreamKind, auto funcKind>
novaStatus_t deviceReleaseDispatch(deviceBuffer_t *buf, std::int32_t device_id,
                                   void *stream) {
  auto *backendBuf = static_cast<BufKind *>(buf->deviceBufPtr);
  const novaStatus_t status =
      funcKind(device_id, backendBuf, static_cast<StreamKind>(stream));
  if (status.err != novaSuccess) {
    // A backend that already handed the memory back zeroes the descriptor
    // before failing. There is nothing left to retry, so the descriptor is
    // released here instead of passing a dead handle to the caller.
    if (backendBuf != nullptr && backendBuf->ptr == nullptr) {
      delete backendBuf;
      *buf = deviceBuffer_t{};
    }
    return status;
  }
  delete backendBuf;
  *buf = deviceBuffer_t{};
  return status;
}

/**
 * @brief Dispatches a buffer resize to one backend.
 *
 * @details
 * Casts the opaque descriptor to the backend type and calls the
 * backend resize.  The device pointer and size are copied back to
 * @p buf because the backend may relocate the allocation.
 *
 * @tparam BufKind    Backend buffer type (@c cudaBuffer_t or
 *                    @c hipBuffer_t).
 * @tparam StreamKind Backend stream type (@c cudaStream_t or
 *                    @c hipStream_t).
 * @tparam funcKind   Backend resize function.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     device_id Runtime device ordinal that owns the buffer.
 * @param[in]     stream    Caller stream forwarded to the backend.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success.
 */
template <typename BufKind, typename StreamKind, auto funcKind>
novaStatus_t deviceResizeDispatch(deviceBuffer_t *buf, std::size_t new_bytes,
                                  std::int32_t device_id, void *stream) {
  auto *backendBuf = static_cast<BufKind *>(buf->deviceBufPtr);
  const novaStatus_t status = funcKind(device_id, backendBuf, new_bytes,
                                       static_cast<StreamKind>(stream));
  if (backendBuf != nullptr && backendBuf->ptr != nullptr) {
    buf->ptr = backendBuf->ptr;
    buf->bytes = backendBuf->bytes;
  }
  return status;
}

/**
 * @brief Dispatches a memory transfer to one backend.
 *
 * @details
 * Casts the opaque stream handle to the backend type and calls the
 * backend transfer.
 *
 * @tparam StreamKind Backend stream type (@c cudaStream_t or
 *                    @c hipStream_t).
 * @tparam funcKind   Backend transfer function.
 *
 * @param[in]  bytes Number of bytes to copy.
 * @param[in]  kind  Copy direction (@ref DeviceMemcpyKind).
 * @param[in]  src   Source pointer.
 * @param[out] dst   Destination pointer.
 * @param[in]  stream Caller stream forwarded to the backend.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success.
 */
template <typename StreamKind, auto funcKind>
novaStatus_t deviceTransferDispatch(std::size_t bytes, DeviceMemcpyKind kind,
                                    const void *src, void *dst, void *stream) {
  return funcKind(bytes, kind, src, dst, static_cast<StreamKind>(stream));
}

/**
 * @brief Operations reachable through the device-agnostic FFI.
 *
 * @details
 * One instance per compiled backend.  Each entry is a dispatch
 * variant specialized to that backend, so the public entry points
 * never name a backend type.
 */
struct BackendOps {
  novaStatus_t (*reserve)(std::size_t bytes, bool pinned,
                          deviceBuffer_t *out_buf, std::int32_t *device_id,
                          void *stream); ///< Allocates a buffer descriptor.
  novaStatus_t (*release)(deviceBuffer_t *buf, std::int32_t device_id,
                          void *stream); ///< Frees a buffer descriptor.
  novaStatus_t (*resize)(deviceBuffer_t *buf, std::size_t new_bytes,
                         std::int32_t device_id, void *stream); ///< Relocates a
                                                                 ///< buffer to a
                                                                 ///< new size.
  novaStatus_t (*transfer)(std::size_t bytes, DeviceMemcpyKind kind,
                           const void *src, void *dst, void *stream); ///< Copies
                                                                        ///< memory
                                                                        ///< between
                                                                        ///< host and
                                                                        ///< device.
  novaStatus_t (*empty_cache)(std::int32_t device_id,
                              std::size_t min_bytes_to_keep); ///< Trims the
                                                              ///< reclaimable
                                                              ///< backing.
  novaStatus_t (*pool_stats)(std::int32_t device_id,
                             DeviceMemoryPoolStats *out); ///< Reads the pool
                                                         ///< counters.
  novaStatus_t (*pool_set_release_threshold)(std::int32_t device_id,
                                             std::uint64_t bytes); ///< Sets the
                                                                         ///< cache
                                                                         ///< release
                                                                         ///< threshold.
  novaStatus_t (*pool_destroy)(std::int32_t device_id); ///< Destroys an idle
                                                         ///< pool.
};

#ifdef NOVA_HAS_CUDA
const BackendOps kCudaOps{
    .reserve =
        &deviceReserveDispatch<cudaBuffer_t, cudaStream_t, cudaReserve,
                               cudaCurrentDeviceId, deviceKind_t::DeviceCUDA>,
    .release = &deviceReleaseDispatch<cudaBuffer_t, cudaStream_t, cudaRelease>,
    .resize = &deviceResizeDispatch<cudaBuffer_t, cudaStream_t, cudaResize>,
    .transfer = &deviceTransferDispatch<cudaStream_t, cudaTransfer>,
    .empty_cache = &cudaEmptyCache,
    .pool_stats = &cudaPoolStats,
    .pool_set_release_threshold = &cudaPoolSetReleaseThreshold,
    .pool_destroy = &cudaPoolDestroy,
};
#endif

#ifdef NOVA_HAS_HIP
const BackendOps kHipOps{
    .reserve =
        &deviceReserveDispatch<hipBuffer_t, hipStream_t, hipReserve,
                               hipCurrentDeviceId, deviceKind_t::DeviceHIP>,
    .release = &deviceReleaseDispatch<hipBuffer_t, hipStream_t, hipRelease>,
    .resize = &deviceResizeDispatch<hipBuffer_t, hipStream_t, hipResize>,
    .transfer = &deviceTransferDispatch<hipStream_t, hipTransfer>,
    .empty_cache = &hipEmptyCache,
    .pool_stats = &hipPoolStats,
    .pool_set_release_threshold = &hipPoolSetReleaseThreshold,
    .pool_destroy = &hipPoolDestroy,
};
#endif

/**
 * @brief Returns the operations of the backend behind @p kind.
 *
 * @details
 * Returns a null pointer when the build does not compile that
 * backend.  The caller then reports @ref backendUnavailable().
 *
 * @param[in] kind Backend to resolve.
 *
 * @return The backend operations, or null when the backend is absent.
 */
const BackendOps *opsFor(deviceKind_t kind) {
#ifdef NOVA_HAS_CUDA
  if (kind == deviceKind_t::DeviceCUDA) {
    return &kCudaOps;
  }
#endif
#ifdef NOVA_HAS_HIP
  if (kind == deviceKind_t::DeviceHIP) {
    return &kHipOps;
  }
#endif
  return nullptr;
}

/**
 * @brief Reports why a backend cannot serve an operation.
 *
 * @details
 * A GPU backend that this build does not compile reports
 * @c novaBackendNotCompiled, a system with no GPU reports
 * @c novaDeviceNotAvailable, and an unrecognized kind reports
 * @c novaInvalidDevice.
 *
 * @param[in] kind Backend that was requested.
 *
 * @return @ref novaStatus_t describing the missing backend.
 */
novaStatus_t backendUnavailable(deviceKind_t kind) {
  if (kind == deviceKind_t::DeviceCUDA) {
    return {.err = novaBackendNotCompiled,
            .message = "CUDA support is not available in this build.\n"};
  }
  if (kind == deviceKind_t::DeviceHIP) {
    return {.err = novaBackendNotCompiled,
            .message = "HIP support is not available in this build.\n"};
  }
  if (kind == deviceKind_t::DeviceNull) {
    return {.err = novaDeviceNotAvailable,
            .message = "No GPU backend was detected on this system.\n"};
  }
  return {.err = novaInvalidDevice,
          .message = "The requested device kind is not recognized.\n"};
}

/**
 * @brief Maps a core transfer direction to the backend copy direction.
 *
 * @details
 * The core enumeration and the backend enumeration share their
 * numeric values, so only the two directions the core names need an
 * explicit mapping.  Every other value maps to device-to-device.
 *
 * @param[in] kind Core transfer direction (@ref TransferKind).
 *
 * @return The matching @ref DeviceMemcpyKind.
 */
constexpr DeviceMemcpyKind mapCTransfKind2DevMemcpyKind(TransferKind kind) {
  switch (kind) {
  case deviceMemcpyHostToDevice:
    return DeviceMemcpyKind::deviceMemcpyHostToDevice;
  case deviceMemcpyDeviceToHost:
    return DeviceMemcpyKind::deviceMemcpyDeviceToHost;
  default:
    return DeviceMemcpyKind::deviceMemcpyDeviceToDevice;
  }
}

} // namespace

/**
 * @brief Allocate a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to the backend behind @p kind.  Compile-time gated via
 * @c NOVA_HAS_CUDA / @c NOVA_HAS_HIP.  A backend that this build does
 * not compile reports @c novaBackendNotCompiled, and a system with no
 * GPU reports @c novaDeviceNotAvailable.
 *
 * @param[in]  bytes   Requested allocation size.
 * @param[out] out_buf Buffer descriptor to populate.
 * @param[in]  pinned  Whether to allocate page-locked memory.
 * @param[in]  kind    Target backend (CUDA or HIP).
 * @param[out] device_id Receives the device selected for the allocation.
 * @param[in]  stream   Optional stream forwarded to the backend.
 *
 * @return Status with @c err == novaSuccess on success.
 *
 * @pre  @p bytes > 0.
 * @post On success, @p out_buf->deviceBufPtr owns a backend
 *       descriptor.
 *
 * @see deviceRelease()  Frees the buffer.
 * @see deviceResize()   Resizes the buffer.
 * @see cudaReserve()    CUDA backend entry point.
 * @see hipReserve()     HIP backend entry point.
 */
novaStatus_t deviceReserve(std::size_t bytes, deviceBuffer_t *out_buf,
                           bool pinned, deviceKind_t kind,
                           std::int32_t *device_id, void *stream) {
  if (out_buf == nullptr) {
    return {.err = novaInvalidPointer,
            .message = "deviceReserve requires a valid output buffer.\n"};
  }
  if (device_id == nullptr) {
    *out_buf = deviceBuffer_t{};
    return {.err = novaInvalidPointer,
            .message = "deviceReserve requires a device ID output.\n"};
  }
  const BackendOps *ops = opsFor(kind);
  if (ops == nullptr) {
    *out_buf = deviceBuffer_t{};
    *device_id = -1;
    return backendUnavailable(kind);
  }
  return ops->reserve(bytes, pinned, out_buf, device_id, stream);
}

/**
 * @brief Free a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to the backend recorded in @p buf->deviceKind.  Once the
 * backend accepts the free the descriptor is zeroed. A backend
 * rejection before acceptance leaves @p buf intact.  A backend that
 * this build does not compile reports @c novaBackendNotCompiled.
 *
 * @param[in,out] buf       Buffer descriptor to free.
 * @param[in]     device_id Runtime device ordinal that owns the buffer.
 * @param[in]     stream    Optional stream that orders the free.
 *
 * @return Status with @c err == novaSuccess on success.
 *
 * @post Once the backend accepts the free, @p buf is zeroed.
 *
 * @see deviceReserve()  Allocates the buffer freed here.
 * @see cudaRelease()    CUDA backend entry point.
 * @see hipRelease()     HIP backend entry point.
 */
novaStatus_t deviceRelease(deviceBuffer_t *buf, std::int32_t device_id,
                           void *stream) {
  if (buf == nullptr) {
    return {.err = novaInvalidPointer,
            .message = "deviceRelease requires a valid buffer.\n"};
  }
  const BackendOps *ops = opsFor(buf->deviceKind);
  if (ops == nullptr) {
    return backendUnavailable(buf->deviceKind);
  }
  return ops->release(buf, device_id, stream);
}

/**
 * @brief Resize a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to the backend recorded in @p buf->deviceKind.  On
 * success, updates @p buf->ptr and @p buf->bytes.  A backend that
 * this build does not compile reports @c novaBackendNotCompiled.
 *
 * @param[in,out] buf       Buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     device_id Runtime device ordinal that owns the buffer.
 * @param[in]     stream    Optional stream ordering allocation, copy and free.
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
 * @see cudaResize()     CUDA backend entry point.
 * @see hipResize()      HIP backend entry point.
 */
novaStatus_t deviceResize(deviceBuffer_t *buf, std::size_t new_bytes,
                          std::int32_t device_id, void *stream) {
  if (buf == nullptr) {
    return {.err = novaInvalidPointer,
            .message = "deviceResize requires a valid buffer.\n"};
  }
  const BackendOps *ops = opsFor(buf->deviceKind);
  if (ops == nullptr) {
    return backendUnavailable(buf->deviceKind);
  }
  return ops->resize(buf, new_bytes, device_id, stream);
}

/**
 * @brief Copy memory between host and device (or device to device).
 *
 * @details
 * Auto-detects the active GPU backend via @ref getDeviceBackend()
 * and dispatches to the appropriate backend memcpy function.  The
 * copy direction is specified by @p kind.  A backend that this build
 * does not compile reports @c novaBackendNotCompiled, and a system
 * with no GPU reports @c novaDeviceNotAvailable.
 *
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  kind      Copy direction (@ref TransferKind).
 * @param[in]  bytes     Number of bytes to copy.
 * @param[in]  stream    Optional stream used to order the transfer.
 *
 * @return @ref novaStatus_t with @c err == novaSuccess on success.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src
 *       and @p dst.
 *
 * @see cudaTransfer()  CUDA backend entry point.
 * @see hipTransfer()   HIP backend entry point.
 */
novaStatus_t deviceTransfer(const void *src, void *dst, TransferKind kind,
                            size_t bytes, void *stream) {

  if (kind != deviceMemcpyHostToDevice && kind != deviceMemcpyDeviceToHost &&
      kind != deviceMemcpyDeviceToDevice) {
    return {.err = novaInvalidTransfDirection,
            .message = "Unknown transfer direction.\n"};
  }

  const deviceKind_t device = getDeviceBackend();
  const BackendOps *ops = opsFor(device);
  if (ops == nullptr) {
    return backendUnavailable(device);
  }
  return ops->transfer(bytes, mapCTransfKind2DevMemcpyKind(kind), src, dst,
                       stream);
}

/**
 * @brief Trims reclaimable backing from one backend device pool.
 *
 * @details
 * Dispatches to the backend behind @p kind.  The backend drains its
 * engine stream before the trim, because a free still queued there
 * keeps its block counted as used and the pool would not return it.
 * A backend that this build does not compile reports
 * @c novaBackendNotCompiled.
 *
 * @param[in] kind              Backend (CUDA or HIP).
 * @param[in] device_id         Runtime device ordinal owning the pool.
 * @param[in] min_bytes_to_keep Minimum resident bytes the pool retains.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success.
 *
 * @see cudaEmptyCache()  CUDA backend entry point.
 * @see hipEmptyCache()   HIP backend entry point.
 */
novaStatus_t deviceEmptyCache(deviceKind_t kind, std::int32_t device_id,
                              std::size_t min_bytes_to_keep) {
  const BackendOps *ops = opsFor(kind);
  if (ops == nullptr) {
    return backendUnavailable(kind);
  }
  return ops->empty_cache(device_id, min_bytes_to_keep);
}

/**
 * @brief Reads runtime statistics for one backend device pool.
 *
 * @details
 * Dispatches to the backend behind @p kind.  The output slot is
 * validated before the backend is reached, so a null pointer never
 * reaches the runtime query.  A backend that this build does not
 * compile reports @c novaBackendNotCompiled.
 *
 * @param[in]  kind      Backend (CUDA or HIP).
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[out] out       Receives the per-device pool counters.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success.
 *
 * @see DeviceMemoryPoolStats  Output counter structure.
 * @see cudaPoolStats()        CUDA backend entry point.
 * @see hipPoolStats()         HIP backend entry point.
 */
novaStatus_t devicePoolStats(deviceKind_t kind, std::int32_t device_id,
                             DeviceMemoryPoolStats *out) {
  if (out == nullptr) {
    return {.err = novaInvalidPointer,
            .message = "devicePoolStats requires an output buffer.\n"};
  }
  const BackendOps *ops = opsFor(kind);
  if (ops == nullptr) {
    return backendUnavailable(kind);
  }
  return ops->pool_stats(device_id, out);
}

/**
 * @brief Sets the release threshold of one backend device pool.
 *
 * @details
 * Dispatches to the backend behind @p kind.  The threshold is the
 * byte count the pool may hold resident before it returns cached
 * blocks to the runtime.  A device without native memory pools
 * reports @c novaMemoryPoolsNotNativelySupported.  A backend that
 * this build does not compile reports @c novaBackendNotCompiled.
 *
 * @param[in] kind      Backend (CUDA or HIP).
 * @param[in] device_id Runtime device ordinal owning the pool.
 * @param[in] bytes     New release threshold in bytes.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success;
 *         @c novaMemoryPoolsNotNativelySupported when pools are unsupported.
 *
 * @see cudaPoolSetReleaseThreshold()  CUDA backend entry point.
 * @see hipPoolSetReleaseThreshold()   HIP backend entry point.
 */
novaStatus_t devicePoolSetReleaseThreshold(deviceKind_t kind,
                                           std::int32_t device_id,
                                           std::uint64_t bytes) {
  const BackendOps *ops = opsFor(kind);
  if (ops == nullptr) {
    return backendUnavailable(kind);
  }
  return ops->pool_set_release_threshold(device_id, bytes);
}

/**
 * @brief Destroys one idle backend device pool.
 *
 * @details
 * Dispatches to the backend behind @p kind.  The pool is destroyed
 * only when no memory is currently in use; a busy pool reports
 * @c novaResourceBusy.  A device without native memory pools treats
 * the call as a no-op and reports success, and its pool slot stays
 * usable.  Otherwise, after success the pool slot is marked destroyed
 * and rejects later operations.  A backend that this build does not
 * compile reports @c novaBackendNotCompiled.
 *
 * @param[in] kind      Backend (CUDA or HIP).
 * @param[in] device_id Runtime device ordinal owning the pool.
 *
 * @return @ref novaStatus_t; @c novaSuccess on success;
 *         @c novaResourceBusy when the pool still holds memory.
 *
 * @see cudaPoolDestroy()  CUDA backend entry point.
 * @see hipPoolDestroy()   HIP backend entry point.
 */
novaStatus_t devicePoolDestroy(deviceKind_t kind, std::int32_t device_id) {
  const BackendOps *ops = opsFor(kind);
  if (ops == nullptr) {
    return backendUnavailable(kind);
  }
  return ops->pool_destroy(device_id);
}
