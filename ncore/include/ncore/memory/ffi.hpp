/**
 * @file ffi.hpp
 * @brief Device-agnostic FFI layer for GPU memory management.
 *
 * @details
 * Defines the top-level C API that the Rust FFI layer
 * (@c ffi/cpp/bindings.rs) calls into for GPU memory operations.
 * All functions are @c extern "C" to be callable from both Rust
 * and pure-C translation units.
 *
 * The implementation dispatches to the correct backend (CUDA or
 * HIP) based on the @ref deviceKind_t tag passed by the caller
 * or detected at runtime.
 *
 * @see ffi.cpp              Implementation of the dispatch layer.
 * @see admin.hpp            deviceKind_t enum and runtime probe.
 * @see deviceTransfer()     C-callable memcpy wrapper (defined in ffi.cpp).
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <ncore/core/device.h>
#include <ncore/core/status.h>

#include <ncore/memory/admin.hpp>

/**
 * @struct deviceBuffer_t
 * @brief Opaque descriptor for a GPU or pinned-host memory buffer.
 *
 * @details
 * The currency type passed between the Rust FFI
 * layer and the C++ backend.  The @ref deviceBufPtr member
 * holds a pointer to a backend-specific descriptor
 * (@c cudaBuffer_t* or @c hipBuffer_t*) that is allocated by
 * @ref deviceReserve and freed by @ref deviceRelease.
 *
 * The @ref deviceKind member defaults to @ref deviceKind_t::DeviceNull.
 * It is set to the correct backend by @ref deviceReserve during
 * allocation; callers must not assume a backend before that point.
 *
 * Callers must not interpret @ref deviceBufPtr directly; it is
 * an opaque handle managed exclusively by the backend.
 */
struct deviceBuffer_t {
  void *ptr = nullptr;   ///< Pointer to device or pinned-host memory.
  std::size_t bytes = 0; ///< Usable size in bytes.
  bool isPinned = false; ///< Page-locked host memory flag.
  deviceKind_t deviceKind = deviceKind_t::DeviceNull; ///< Active backend.
  void *deviceBufPtr = nullptr; ///< Opaque backend-specific descriptor.
};

/**
 * @brief Per-device runtime memory-pool counters.
 *
 * @details
 * Snapshot returned by @ref devicePoolStats(). Values are read from
 * the runtime's pool attributes at query time. High-water marks are
 * cumulative since the pool was created; this layer never clears them,
 * so a caller comparing successive snapshots must account for the
 * monotonic baseline.
 */
struct DeviceMemoryPoolStats {
  std::int32_t device_id;          ///< Runtime device ordinal.
  bool native_pool_supported;      ///< @c true when the device exposes a
                                   ///< native memory pool.
  std::uint64_t reserved_current;  ///< Bytes currently reserved by the pool.
  std::uint64_t reserved_high;     ///< High-water reserved bytes.
  std::uint64_t used_current;      ///< Bytes currently in use.
  std::uint64_t used_high;         ///< High-water used bytes.
  std::uint64_t release_threshold; ///< Configured release threshold.
};

/**
 * @enum DeviceMemcpyKind
 * @brief Memory copy direction for inter-device transfers.
 */
enum class DeviceMemcpyKind : std::int8_t {
  deviceMemcpyHostToDevice = 1,  ///< Host → Device copy.
  deviceMemcpyDeviceToHost = 2,  ///< Device → Host copy.
  deviceMemcpyDeviceToDevice = 3 ///< Device → Device copy.
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

extern "C" {

/**
 * @brief Allocate a GPU or pinned-host memory buffer.
 *
 * @details
 * Dispatches to the appropriate backend allocator based on
 * @p kind.  The allocated buffer is described by @p out_buf,
 * which takes ownership of the backend-specific descriptor.
 * A backend that this build does not compile reports
 * @c novaBackendNotCompiled, and a system with no GPU reports
 * @c novaDeviceNotAvailable.
 *
 * @param[in]  bytes   Requested allocation size in bytes.
 * @param[out] out_buf Receives the buffer descriptor on success.
 * @param[in]  pinned  If @c true, allocate page-locked host memory.
 * @param[in]  kind    Target backend (CUDA or HIP).
 * @param[out] device_id Receives the runtime device ordinal selected at
 *                       reservation time.
 * @param[in]  stream   Optional caller stream forwarded to the backend;
 *                       null selects the device engine stream.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out_buf must point to a valid @ref deviceBuffer_t.
 * @post On success, @p out_buf->deviceBufPtr owns a backend
 *       descriptor that must be freed via @ref deviceRelease.
 * @post On failure, @p out_buf is reset and no ownership is transferred
 *       to the caller.
 *
 * @see deviceRelease()  Frees the buffer allocated here.
 * @see deviceResize()   Resizes an existing buffer.
 * @see cudaReserve()    CUDA backend entry point.
 * @see hipReserve()     HIP backend entry point.
 */
novaStatus_t deviceReserve(std::size_t bytes, deviceBuffer_t *out_buf,
                           bool pinned, deviceKind_t kind,
                           std::int32_t *device_id, void *stream = nullptr);

/**
 * @brief Free a GPU or pinned-host memory buffer.
 *
 * @details
 * Casts @p buf->deviceBufPtr to the backend-specific type and calls the
 * backend release function. The descriptor is cleared after the backend
 * accepts the free.  A backend that this build does not compile reports
 * @c novaBackendNotCompiled.
 *
 * @param[in,out] buf  Pointer to the buffer descriptor to free.
 * @param[in]     device_id Runtime device ordinal that owns the buffer.
 * @param[in]     stream Caller stream that orders the free, or null to
 *                        select the device engine stream.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p buf must point to a valid @ref deviceBuffer_t whose
 *       @ref deviceBufPtr was returned by @ref deviceReserve.
 * @post Once the backend accepts the free, @p buf is zeroed, so the
 *       submitted free cannot be retried against the same pointer.
 * @post When the backend rejects the free before accepting it, @p buf keeps
 *       its pointer and the caller decides whether to retry or abandon it.
 *
 * @see deviceReserve()  Allocates the buffer freed here.
 * @see cudaRelease()    CUDA backend entry point.
 * @see hipRelease()     HIP backend entry point.
 */
novaStatus_t deviceRelease(deviceBuffer_t *buf, std::int32_t device_id,
                           void *stream = nullptr);

/**
 * @brief Resize an existing GPU or pinned-host memory buffer.
 *
 * @details
 * Casts @p buf->deviceBufPtr to the backend-specific type and
 * calls the backend resize function.  On success, the @ref ptr
 * and @ref bytes members of @p buf are updated.  A backend that this
 * build does not compile reports @c novaBackendNotCompiled.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to
 *                          resize.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     device_id Runtime device ordinal that owns the buffer.
 * @param[in]     stream Caller stream for the allocation, copy and free,
 *                       or null to select the device engine stream.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p buf must point to a valid @ref deviceBuffer_t whose
 *       @ref deviceBufPtr was returned by @ref deviceReserve.
 * @post On success, @p buf->ptr and @p buf->bytes reflect the
 *       new allocation.
 *
 * @warning A backend-specific failure may leave the original buffer in an
 *          unusable or backend-defined state. Do not issue another operation
 *          on it until the backend's status contract has been evaluated.
 *
 * @see deviceReserve()  Initial allocation.
 * @see deviceRelease()  Explicit deallocation.
 * @see cudaResize()     CUDA backend entry point.
 * @see hipResize()      HIP backend entry point.
 */
novaStatus_t deviceResize(deviceBuffer_t *buf, std::size_t new_bytes,
                          std::int32_t device_id, void *stream = nullptr);
/**
 * @brief Perform a memory transfer between the host and a GPU
 *        device.
 *
 * @details
 * Dispatches to the active GPU backend (CUDA or HIP) determined at
 * run time by @ref getDeviceBackend().  The @p kind selects the
 * copy direction and must match the actual memory types of
 * @p src and @p dst.  A backend that this build does not compile
 * reports @c novaBackendNotCompiled, and a system with no GPU reports
 * @c novaDeviceNotAvailable.
 *
 * @param[in]  src   Source pointer.
 * @param[out] dst   Destination pointer.
 * @param[in]  kind  Copy direction (@ref TransferKind).
 * @param[in]  bytes Number of bytes to copy.
 * @param[in]  stream Caller stream for ordering with surrounding work,
 *                 or null to select the device engine stream.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @pre  @p src and @p dst must point to valid memory regions of
 *       at least @p bytes.
 * @pre  @p kind must match the actual memory types of @p src and
 *       @p dst.
 *
 * @see deviceReserve()  Allocates a GPU buffer.
 * @see deviceRelease()  Frees a GPU buffer.
 * @see cudaTransfer()  CUDA backend entry point.
 * @see hipTransfer()   HIP backend entry point.
 */
novaStatus_t deviceTransfer(const void *src, void *dst, TransferKind kind,
                            size_t bytes, void *stream = nullptr);

/**
 * @brief Trims reclaimable backing from one backend device pool.
 *
 * @details
 * Dispatches to the CUDA or HIP backend based on @p kind.  When the
 * device does not support native memory pools the call is a no-op and
 * reports success.  Otherwise it trims the pool to at least
 * @p min_bytes_to_keep of resident bytes.  A backend that this build
 * does not compile reports @c novaBackendNotCompiled.
 *
 * @param[in]  kind                  Backend (CUDA or HIP).
 * @param[in]  device_id             Runtime device ordinal owning the pool.
 * @param[in]  min_bytes_to_keep     Minimum resident bytes to retain.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success.
 *
 * @see cudaEmptyCache()  CUDA backend entry point.
 * @see hipEmptyCache()   HIP backend entry point.
 */
novaStatus_t deviceEmptyCache(deviceKind_t kind, std::int32_t device_id,
                              std::size_t min_bytes_to_keep = 0);
/**
 * @brief Reads runtime statistics for one backend device pool.
 *
 * @details
 * Dispatches to the CUDA or HIP backend based on @p kind.  When the
 * device does not support native memory pools the call reports
 * @c novaMemoryPoolsNotNativelySupported with the counters zeroed.
 * A backend that this build does not compile reports
 * @c novaBackendNotCompiled.
 *
 * @param[in]  kind   Backend (CUDA or HIP).
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[out] out   Receives the per-device pool counters.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success;
 *         @c novaMemoryPoolsNotNativelySupported when pools are unsupported.
 *
 * @see DeviceMemoryPoolStats  Output counter structure.
 * @see cudaPoolStats()        CUDA backend entry point.
 * @see hipPoolStats()         HIP backend entry point.
 */
novaStatus_t devicePoolStats(deviceKind_t kind, std::int32_t device_id,
                             DeviceMemoryPoolStats *out);
/**
 * @brief Sets the release threshold for one backend device pool.
 *
 * @details
 * Dispatches to the CUDA or HIP backend based on @p kind.  The threshold
 * controls how many bytes the pool keeps resident before returning them
 * to the runtime.  When the device does not support native memory pools
 * the call reports @c novaMemoryPoolsNotNativelySupported.  A backend
 * that this build does not compile reports @c novaBackendNotCompiled.
 *
 * @param[in]  kind   Backend (CUDA or HIP).
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[in]  bytes   New release threshold in bytes.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success;
 *         @c novaMemoryPoolsNotNativelySupported when pools are unsupported.
 *
 * @see cudaPoolSetReleaseThreshold()  CUDA backend entry point.
 * @see hipPoolSetReleaseThreshold()   HIP backend entry point.
 */
novaStatus_t devicePoolSetReleaseThreshold(deviceKind_t kind,
                                           std::int32_t device_id,
                                           std::uint64_t bytes);
/**
 * @brief Destroys one idle backend device pool.
 *
 * @details
 * Dispatches to the CUDA or HIP backend based on @p kind.  The pool is
 * only destroyed when no memory is currently in use; otherwise the call
 * reports @c novaResourceBusy.  When the device does not support native
 * memory pools the call is a no-op and reports success, and the pool
 * slot stays usable.  Otherwise, after success the pool slot is marked
 * destroyed and rejects later operations.  A backend that this build
 * does not compile reports @c novaBackendNotCompiled.
 *
 * @param[in]  kind      Backend (CUDA or HIP).
 * @param[in]  device_id Runtime device ordinal owning the pool.
 *
 * @return @ref novaStatus_t with @c err = novaSuccess on success;
 *         @c novaResourceBusy when the pool still holds memory.
 *
 * @see cudaPoolDestroy()  CUDA backend entry point.
 * @see hipPoolDestroy()   HIP backend entry point.
 */
novaStatus_t devicePoolDestroy(deviceKind_t kind, std::int32_t device_id);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
