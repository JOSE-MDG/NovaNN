/**
 * @file device.h
 * @brief Public C API for device placement, backend queries, and
 *        inter-backend memory transfers.
 *
 * Device tiers:
 *
 * @li @c DEVICE_CPU - 0 - Host memory (default).
 * @li @c DEVICE_GPU - 1 - Accelerator memory (CUDA/ROCm).
 * @li @c DEVICE_META - 2 - Placeholder with no backing storage.
 *
 * META tensors carry dtype and shape metadata only; nothing is
 * allocated, which makes them useful for shape inference and graph
 * construction.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum Device_
 * @brief Target device for tensor data placement.
 *
 * Backed by a @c uint8_t underlying type (C23) to keep the footprint
 * small in frequently copied structs.
 */
typedef enum Device_ : uint8_t {
  DEVICE_CPU = 0, ///< Host memory; accessible by all CPU kernels.
  DEVICE_GPU = 1, ///< Accelerator memory; requires GPU-capable backend.
  DEVICE_META = 2 ///< Placeholder; no data buffer is allocated.
} Device_;

/**
 * @enum DeviceKind
 * @brief Runtime backend used for GPU operations.
 *
 * Sequential values so they index directly into dispatch tables.
 * Backed by @c uint8_t for the same reason as @ref Device_.
 */
typedef enum DeviceKind : uint8_t {
  CUDA_DEVICE = 0, ///< NVIDIA CUDA runtime backend.
  HIP_DEVICE = 1,  ///< AMD ROCm HIP runtime backend.
  NULL_DEVICE = 2  ///< No supported GPU backend detected.
} DeviceKind;

/**
 * @enum TransferKind
 * @brief Device-agnostic memory transfer direction.
 *
 * Mirrors the copy directions of CUDA and HIP without exposing their
 * runtime-local enums. Values match @c cudaMemcpyKind and
 * @c hipMemcpyKind, so dispatching needs no translation. The table
 * @ref transf_dispatch (in @ref device.c) maps @c (src, dst) pairs to
 * a value here.
 */
typedef enum TransferKind : uint8_t {
  deviceMemcpyHostToDevice = 1,  ///< Copy from host memory into device memory.
  deviceMemcpyDeviceToHost = 2,  ///< Copy from device memory into host memory.
  deviceMemcpyDeviceToDevice = 3 ///< Copy between two device-memory buffers.
} TransferKind;

/**
 * @brief Forward declaration; the full type lives in @c tensor.h.
 */
struct Tensor;

/** @brief Type alias for the Tensor struct. */
typedef struct Tensor Tensor;

/** @brief Pointer-to-Tensor type alias, commonly used for gradient tensors. */
typedef Tensor *TensorGrad;

/**
 * @brief Check whether a GPU backend is available.
 *
 * The first call performs the actual runtime probe and only a
 * successful probe is cached: later calls for the same kind return
 * from the cache without touching the runtime API, while a different
 * kind, or any call after a failed probe, probes again. This assumes
 * a single GPU vendor per process (CUDA _or_ HIP, never both).
 *
 * @param[in] kind     Requested backend kind.  Must be a valid
 *                     @ref DeviceKind value.
 * @param[in] verbose  If @c true, backend detection may print runtime
 *                     diagnostics to @c stdout.
 * @return @c true when the requested backend reports an available
 *         device.  @c false otherwise, including when
 *         @c NOVA_HAS_CUDA / @c NOVA_HAS_HIP are not defined.
 *
 * Thread-safe: the cache is mutex-protected and initialized through
 * @c call_once/@c InitOnceExecuteOnce.
 */
bool is_device_available(DeviceKind kind, bool verbose);

/**
 * @brief Check whether CUDA is available.
 *
 * Calls @ref is_device_available(CUDA_DEVICE, false), so no
 * diagnostics get printed.
 *
 * @return @c true when CUDA reports an available device.
 */
bool is_cuda_available();

/**
 * @brief Check whether HIP is available.
 *
 * Calls @ref is_device_available(HIP_DEVICE, false), so no
 * diagnostics get printed.
 *
 * @return @c true when HIP reports an available device.
 */
bool is_hip_available();

/**
 * @brief Transfer memory between device backends.
 *
 * Resolves the @ref TransferKind from @ref transf_dispatch for the
 * @c (src, dst) pair and forwards to @c deviceTransfer() (declared in
 * @ref ffi.hpp). The dispatch table is initialized at program startup
 * by an @c INITIALIZE(init_transf_dispatch) in @ref device.c.
 *
 * @param[in]  src       Source device placement.
 * @param[in]  dst       Target device placement.
 * @param[in]  src_buf   Source buffer, valid for at least @p bytes
 *                       in the source memory space.
 * @param[out] dst_buf   Destination buffer, valid for at least
 *                       @p bytes in the destination memory space.
 * @param[in]  bytes     Number of bytes to transfer.  Must be > 0.
 *
 * @return @ref novaStatus_t: @ref novaSuccess on success, an error
 *         status with a descriptive message otherwise.
 *
 * @note @c CPU to @c CPU is rejected with
 *       @ref novaInvalidTransfDirection; use @c memcpy() or
 *       @ref deepcopy() for host-to-host copies.
 * @note Read-only dispatch table after initialization, safe to call
 *       from multiple threads.
 */
NCORE_HOST_DEVICE novaStatus_t transfer_to(Device_ src, Device_ dst,
                                           const void *src_buf, void *dst_buf,
                                           size_t bytes);

/**
 * @brief Return the active device id (CUDA or HIP).
 *
 * The id is a 0-based index into the device list of the active
 * runtime; currently only the first device (id @c 0) is supported by
 * the detection layer.
 *
 * @return Active device id, or @c -1 when no GPU device is available
 *         or detection has not yet been performed. Meaningful only
 *         after one of the availability functions returned @c true.
 */
int get_device_id();

/**
 * @brief Print detailed or concise device information to stdout.
 *
 * With @p verbose false a two-line summary is printed per device, true
 * prints the full property block (warp size, threads limits, driver
 * versions...). Safe to call before any availability check. If the
 * requested backend is unavailable, nothing is printed.
 *
 * @param[in] kind     Backend to query.  Must be @c CUDA_DEVICE or
 *                     @c HIP_DEVICE.  @c NULL_DEVICE is a no-op.
 * @param[in] verbose  If @c true, print the detailed block.
 * @return novaStatus_t the result of the operation. On success, set to
 *                     @ref novaSuccess.  On failure, set to the
 *                     appropriate error code.
 */
novaStatus_t print_device_info(DeviceKind kind, bool verbose);

/**
 * @brief Return the cached device kind from the last detection.
 *
 * @return The detected @ref DeviceKind, or @c NULL_DEVICE if no
 *         detection has occurred or no GPU was found.
 */
DeviceKind get_detected_device_kind();

/**
 * @brief Check whether device detection has already run.
 *
 * @return @c true after any of the availability functions completed
 *         at least once.
 */
bool is_device_detection_done();

#ifdef __cplusplus
}
#endif
