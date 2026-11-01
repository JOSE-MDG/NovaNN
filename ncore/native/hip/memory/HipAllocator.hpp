/**
 * @file HipAllocator.hpp
 * @brief HIP memory allocation types and operations.
 *
 * @details
 * Declares the @ref hipBuffer_t descriptor and the three core allocation
 * functions (@ref hipReserve, @ref hipRelease, @ref hipResize) that
 * manage HIP device and pinned-host memory.  All functions return a
 * @ref novaStatus_t (see @ref status.h).
 *
 * Device memory uses the selected device's NovaNN pool when supported,
 * and direct synchronous allocation otherwise. Pinned host memory stays
 * outside device pools. A null stream selects the device engine stream;
 * a caller stream enqueues without synchronization and the caller owns
 * ordering.
 *
 * This header is the HIP counterpart of @c CudaAllocator.hpp with an
 * identical API surface; the dispatch layer in @c ffi.cpp selects
 * between CUDA and HIP at runtime.
 *
 * @see HipAllocator.cpp  Implementation of the allocation functions.
 * @see HipIO.hpp         HIP data transfer functions.
 * @see common.hh         Shared error-mapping and stream helpers.
 * @see ffi.hpp           Device-agnostic FFI layer that wraps these.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <ncore/core/status.h>

#include <ncore/headeronly/stream_types.h>

/**
 * @struct hipBuffer_t
 * @brief Descriptor for a HIP-allocated memory region.
 *
 * @details
 * Tracks the raw pointer, usable size, allocation type
 * (device or pinned-host), and the stream the allocation ran on.
 * This struct is owned by the
 * @ref deviceBuffer_t in the FFI layer and must not be freed
 * directly; use @ref hipRelease instead.
 */
struct hipBuffer_t {
  void *ptr = nullptr;                ///< Device or pinned-host pointer.
  std::size_t bytes = 0;              ///< Usable size in bytes.
  bool isPinned = false;              ///< Page-locked host memory flag.
  hipStream_t streamOrigin = nullptr; ///< Stream the allocation ran on.
};

struct DeviceMemoryPoolStats;

/**
 * @brief Sentinel value representing a successful HIP operation.
 * @var HIP_OK
 */
const inline novaStatus_t HIP_OK{
    .err = novaSuccess, .message = nova_get_error_msg(novaSuccess, nullptr)};

/**
 * @brief Read the runtime ordinal of the currently selected HIP device.
 *
 * @param[out] device_id Receives the 0-based runtime ordinal. Must not be
 *                       null.
 *
 * @return @ref HIP_OK on success, or an error status when @p device_id is
 *         null or the runtime call fails.
 *
 * @pre  @p device_id must point to writable @c std::int32_t storage.
 */
novaStatus_t hipCurrentDeviceId(std::int32_t *device_id);

/**
 * @brief Allocate a HIP memory buffer.
 *
 * @details
 * Device allocations use @c hipMallocFromPoolAsync with the pool selected
 * by @p device_id, on the device engine stream when @p stream is null.
 * Devices without native pools use synchronous @c hipMalloc. Pinned
 * allocations use @c hipHostMalloc.
 *
 * @param[in]  device_id Runtime device ordinal selecting the pool.
 * @param[in]  bytes     Requested allocation size in bytes.
 * @param[in]  pinned    If @c true, allocate page-locked host memory.
 * @param[out] out       Receives the buffer descriptor on success.
 * @param[in]  stream    Caller stream for chaining, or null to select
 *                       the device engine stream.
 *
 * @return @ref HIP_OK on success, or an error status with a
 *         descriptive message.
 *
 * @pre  @p bytes must be greater than zero.
 * @pre  @p out must not be null.
 * @post On success with @p stream == null, @p out->ptr points to a
 *       valid HIP memory region of at least @p bytes usable from
 *       any stream.
 * @post On success with @p stream != null, @p out->ptr is valid only
 *       in @p stream order (ordered after this call).
 *
 * @warning With @p stream != null there is no synchronization: any
 *          access outside @p stream order is undefined behavior.
 *
 * @see hipRelease()  Frees a buffer allocated by this function.
 * @see hipResize()   Resizes an existing buffer.
 */
novaStatus_t hipReserve(std::int32_t device_id, std::size_t bytes, bool pinned,
                        hipBuffer_t *out, hipStream_t stream = nullptr);

/**
 * @brief Free a HIP memory buffer previously allocated by
 *        @ref hipReserve.
 *
 * @details
 * Device memory is freed through the pool selected by @p device_id, on
 * the device engine stream when @p stream is null. Pinned memory uses
 * @c hipFreeHost.
 *
 * @param[in,out] buf    Pointer to the buffer descriptor to free.
 *                       Must not be null, and @p buf->ptr must be
 *                       valid.
 * @param[in]     device_id Runtime device ordinal owning the buffer.
 * @param[in]     stream Caller stream for chaining, or null to select
 *                       the device engine stream.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @pre  @p buf must point to a valid @ref hipBuffer_t whose
 *       @ref ptr member was returned by @ref hipReserve.
 * @pre  Every prior access to @p buf->ptr must be ordered before this
 *       call in the stream that owns the buffer.
 * @post On success, @p buf is zeroed (ptr = nullptr, bytes = 0,
 *       isPinned = false).
 * @post A free submitted on a stream other than the one that owns the
 *       buffer is rejected; cross-stream use requires an explicit event
 *       dependency.
 *
 * @note Safe to call with a null @p buf or null @p buf->ptr;
 *       returns a non-zero status without crashing.
 *
 * @warning With @p stream != null there is no synchronization: the
 *          memory must not be accessed after this call in any stream
 *          unless reordered with events.
 *
 * @see hipReserve()  Allocates the buffer freed here.
 */
novaStatus_t hipRelease(std::int32_t device_id, hipBuffer_t *buf,
                        hipStream_t stream = nullptr);

/**
 * @brief Resize an existing HIP memory buffer.
 *
 * @details
 * Allocates replacement storage, copies the preserved prefix, then frees the
 * old buffer. Device operations use the selected pool and @p stream.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to
 *                          resize.  Must not be null.
 * @param[in]     new_bytes New size in bytes.
 * @param[in]     device_id Runtime device ordinal owning the buffer.
 * @param[in]     stream    Caller stream for chaining, or null to
 *                          select the device engine stream. The
 *                          alloc, copy, and free all run on this
 *                          stream.
 *
 * @return @ref HIP_OK on success, or an error status.
 *
 * @pre  @p buf must point to a valid @ref hipBuffer_t whose
 *       @ref ptr member was returned by @ref hipReserve.
 * @post On success, @p buf->ptr and @p buf->bytes are updated to
 *       reflect the new allocation.
 *
 * @warning On failure the original buffer may already be freed
 *          (e.g., if the copy succeeded but the free failed).
 *          Do not use the old @p buf->ptr after a failed resize.
 *
 * @see hipReserve()  Initial allocation.
 * @see hipRelease()  Explicit deallocation.
 */
novaStatus_t hipResize(std::int32_t device_id, hipBuffer_t *buf,
                       std::size_t new_bytes, hipStream_t stream = nullptr);

/**
 * @brief Trim reclaimable backing from one device pool.
 *
 * @details
 * Calls @c hipMemPoolTrimTo so the pool releases cached blocks down to at
 * least @p min_bytes_to_keep resident bytes. Devices without native pools
 * treat the call as a no-op and report success.
 *
 * @param[in] device_id           Runtime device ordinal owning the pool.
 * @param[in] min_bytes_to_keep   Minimum resident bytes to retain.
 *
 * @return @ref HIP_OK on success, or an error status from the runtime.
 */
novaStatus_t hipEmptyCache(std::int32_t device_id,
                           std::size_t min_bytes_to_keep = 0);

/**
 * @brief Read runtime statistics for one device pool.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[out] out       Receives the counters. Must not be null; zeroed
 *                       before the query.
 *
 * @return @ref HIP_OK on success, @ref novaInvalidPointer when @p out is
 *         null, or @ref novaMemoryPoolsNotNativelySupported when the device
 *         has no native pool.
 */
novaStatus_t hipPoolStats(std::int32_t device_id, DeviceMemoryPoolStats *out);

/**
 * @brief Set the caching threshold for one device pool.
 *
 * @details
 * Sets @c hipMemPoolAttrReleaseThreshold, the byte count the pool may hold
 * before returning cached memory to the driver.
 *
 * @param[in] device_id Runtime device ordinal owning the pool.
 * @param[in] bytes     New release threshold in bytes.
 *
 * @return @ref HIP_OK on success, or @ref novaMemoryPoolsNotNativelySupported
 *         when the device has no native pool.
 */
novaStatus_t hipPoolSetReleaseThreshold(std::int32_t device_id,
                                        std::uint64_t bytes);

/**
 * @brief Destroy one idle device pool.
 *
 * @details
 * The pool is destroyed only when no memory is currently in use. After
 * success the slot is marked destroyed and rejects later operations.
 * Devices without native pools treat the call as a no-op and report
 * success.
 *
 * @param[in] device_id Runtime device ordinal owning the pool.
 *
 * @return @ref HIP_OK on success, @ref novaResourceBusy when the pool still
 *         holds memory, or an error status from the runtime.
 */
novaStatus_t hipPoolDestroy(std::int32_t device_id);
