/**
 * @file device.h
 * @brief Public C API for device placement, backend queries, and
 *        inter-backend memory transfers.
 *
 * @details
 * This header defines the foundational types and functions that the
 * rest of the NovaNN codebase uses to interact with heterogeneous
 * compute devices.  It is the single include point for:
 *
 * - **Device enumeration**: The @ref Device enum classifies where a
 *   tensor's backing storage lives (CPU, GPU, or META).
 * - **Backend identification**: The @ref DeviceKind enum identifies
 *   which GPU runtime (CUDA, HIP, or none) is in use.
 * - **Transfer direction**: The @ref TransferKind enum encodes copy
 *   directions without exposing runtime-specific types.
 * - **Result reporting**: The @ref DeviceStatus struct carries an
 *   error code and human-readable message from memory operations.
 * - **Backend probes**: @ref is_device_available(), @ref
 *   is_cuda_available(), and @ref is_hip_available() query runtime
 *   availability at run time.
 * - **Memory transfers**: @ref transfer_to() dispatches inter-device
 *   copies through the correct backend at run time.
 * - **Device identification**: @ref get_device_id() returns the
 *   active device id for the detected backend.
 *
 * ## Device Tiers
 */
// clang-format off
/**
 * | Device      | Value | Description                         |
 * |-------------|-------|-------------------------------------|
 * | `DEVICE_CPU`  | 0   | Host memory (default).              |
 * | `DEVICE_GPU`  | 1   | Accelerator memory (CUDA/ROCm).     |
 * | `DEVICE_META` | 2   | Placeholder with no backing storage.|
 */
// clang-format on
/**
 * META tensors are used for shape inference and graph construction
 * without allocating actual data buffers.  They carry dtype and shape
 * metadata but no memory.
 *
 * ## Usage Example
 *
 * @code{.cpp}
 * #include <ncore/device.h>
 *
 * // Check if any GPU backend is available.
 * if (is_device_available(CUDA_DEVICE, true)) {
 *     std::cout << "CUDA device found" << "\n";
 * }
 *
 * // Transfer a buffer from host to device.
 * DeviceStatus status = transfer_to(DEVICE_CPU, DEVICE_GPU, src, dst, true, n);
 * if (status.code != 0) {
 *     std::cout << "Transfer failed: " << status.message << "\n";
 * }
 * @endcode
 *
 * @see tensor.h      Tensor structure embedding a @ref Device field.
 * @see cpp_ffi.h     C-callable device-memory copy wrapper used by
 *                    @ref transfer_to().
 * @see cuda_device.c CUDA-specific detection implementation.
 * @see hip_device.c  HIP-specific detection implementation.
 */

#pragma once

#include <ncore/macros.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum Device
 * @brief Target device for tensor data placement.
 *
 * @details
 * Determines where a tensor's backing storage is allocated and which
 * compute kernels can operate on it.  Every tensor in the NovaNN
 * framework carries a @ref Device field that specifies its memory
 * location.
 *
 * The three tiers form a strict hierarchy:
 * - `DEVICE_CPU` is the default and universally accessible.
 * - `DEVICE_GPU` requires a capable runtime (CUDA or HIP) and is
 *   where accelerated computation happens.
 * - `DEVICE_META` carries metadata only — no data buffer is allocated.
 *   This is useful for shape inference, graph construction, and
 *   lazy evaluation patterns where the actual memory location is
 *   determined later.
 *
 * @note This enum is packed (`ATTR(packed)`) to minimise its
 *       footprint in structs that are serialised or copied frequently.
 *
 * @see DeviceKind     Identifies which GPU runtime is in use.
 * @see DeviceStatus   Result type for device memory operations.
 */
typedef enum ATTR(packed) {
  DEVICE_CPU = 0, ///< Host memory; accessible by all CPU kernels.
  DEVICE_GPU = 1, ///< Accelerator memory; requires GPU-capable backend.
  DEVICE_META = 2 ///< Placeholder; no data buffer is allocated.
} Device;

/**
 * @enum DeviceKind
 * @brief Runtime backend used for GPU operations.
 *
 * @details
 * Identifies which GPU compute runtime is available at run time.
 * This enum is used by @ref is_device_available() to probe specific
 * backends and by internal dispatch logic to select the correct
 * runtime API calls.
 *
 * The values are sequential integers starting from 0, which allows
 * them to be used as array indices in dispatch tables (see
 * @ref device.c).
 *
 * @note This enum is packed (`ATTR(packed)`) to minimise its
 *       footprint in structs that are serialised or copied frequently.
 *
 * @see Device          Target device for tensor data placement.
 * @see is_device_available()
 * @see is_cuda_available()
 * @see is_hip_available()
 */
typedef enum ATTR(packed) {
  CUDA_DEVICE = 0, ///< NVIDIA CUDA runtime backend.
  HIP_DEVICE = 1,  ///< AMD ROCm HIP runtime backend.
  NULL_DEVICE = 2  ///< No supported GPU backend detected.
} DeviceKind;

/**
 * @enum TransferKind
 * @brief Device-agnostic memory transfer direction.
 *
 * @details
 * Mirrors the copy directions supported by CUDA and HIP without
 * exposing runtime-specific enums in the public C API.  The values
 * are chosen to match the `cudaMemcpyKind` and `hipMemcpyKind`
 * enums, so no translation is needed when dispatching to the
 * underlying runtime.
 *
 * The dispatch table @ref transf_dispatch (in @ref device.c) maps
 * pairs of `(src Device, dst Device)` to a `TransferKind` value,
 * which is then passed to the C-callable `device_memcpy_c()` wrapper.
 *
 * @note This enum is packed (`ATTR(packed)`) to minimise its
 *       footprint in structs that are serialised or copied frequently.
 *
 * @see DeviceStatus   Result type for device memory operations.
 * @see transfer_to()  High-level transfer function using this enum.
 * @see transf_dispatch  Lookup table mapping device pairs to directions.
 */
typedef enum ATTR(packed) {
  deviceMemcpyHostToDevice = 1,  ///< Copy from host memory into device memory.
  deviceMemcpyDeviceToHost = 2,  ///< Copy from device memory into host memory.
  deviceMemcpyDeviceToDevice = 3 ///< Copy between two device-memory buffers.
} TransferKind;

/**
 * @struct DeviceStatus
 * @brief Result type returned by device memory operations.
 *
 * @details
 * A lightweight status struct that carries both a numeric error code
 * and a human-readable message.  A `code` of `0` indicates success;
 * any non-zero value indicates a failure.  The `message` field points
 * to a static string (not dynamically allocated) that describes the
 * error.
 *
 * Callers should check `code` first and only access `message` when
 * `code` is non-zero.  The `message` pointer is guaranteed to be
 * valid for the lifetime of the program (it points to a string
 * literal or a static buffer).
 *
 * @var DeviceStatus::code
 * @brief Zero on success, a positive error code on failure.
 *
 * @var DeviceStatus::message
 * @brief Human-readable error description.  Do not free this pointer.
 *
 * @see transfer_to()  Returns a DeviceStatus.
 * @see device_memcpy_c()  Low-level wrapper that returns a DeviceStatus.
 */
typedef struct {
  int code;            ///< Zero on success, positive error code on failure.
  const char *message; ///< Human-readable error description.
} DeviceStatus;

/**
 * @struct Tensor
 * @brief Forward declaration of the Tensor type.
 *
 * @details
 * The Tensor struct is defined in `tensor.h` and carries a @ref Device
 * field indicating where its data resides.  This forward declaration
 * allows @ref TensorGrad (used for gradient tensors) to be defined
 * without pulling in the full tensor header.
 */
struct Tensor;

/** @brief Type alias for the Tensor struct. */
typedef struct Tensor Tensor;

/** @brief Pointer-to-Tensor type alias, commonly used for gradient tensors. */
typedef Tensor *TensorGrad;

/**
 * @brief Check whether a GPU backend is available.
 *
 * @details
 * Dispatches to the backend-specific detection function based on
 * @p kind:
 * - `CUDA_DEVICE` → @ref is_cuda_device_available() (from
 *   @ref cuda_device.c).
 * - `HIP_DEVICE` → @ref is_hip_device_available() (from
 *   @ref hip_device.c).
 * - `NULL_DEVICE` or any other value → returns `false`.
 *
 * When the corresponding backend macro (`NOVA_HAS_CUDA` or
 * `NOVA_HAS_HIP`) is not defined, the function returns `false`
 * without querying the runtime.
 *
 * ## One-shot caching
 *
 * The first call to this function performs the actual runtime probe
 * and caches the result.  Subsequent calls — regardless of the
 * requested @p kind — return immediately from the cache without
 * touching the runtime API.  This design assumes a single GPU
 * vendor per process (CUDA _or_ HIP, never both), which is the
 * standard constraint in deep-learning workloads.
 *
 * @param[in] kind     Requested backend kind.  Must be a valid
 *                     @ref DeviceKind value.
 * @param[in] verbose  If `true`, backend detection may print runtime
 *                     diagnostics to `stdout`.  Pass `false` for
 *                     silent operation.
 *
 * @return `true` when the requested backend reports an available
 *         device.  `false` otherwise.
 *
 * @note Thread-safe.  Delegates to thread-safe backend detection
 *       functions.  The cached result is protected by a mutex and
 *       a `call_once` initialisation guard.
 *
 * @see is_cuda_available()   Convenience wrapper for `CUDA_DEVICE`.
 * @see is_hip_available()    Convenience wrapper for `HIP_DEVICE`.
 * @see get_detected_device_kind()  Returns the cached backend.
 * @see was_device_detection_done()  Checks if detection ran.
 * @see DeviceKind            Enum identifying backends.
 */
bool is_device_available(DeviceKind kind, bool verbose);

/**
 * @brief Check whether CUDA is available.
 *
 * @details
 * Convenience wrapper that calls @ref is_device_available() with
 * `CUDA_DEVICE`.  Equivalent to:
 * @code{.cpp}
 * is_device_available(CUDA_DEVICE, false);
 * @endcode
 *
 * @return `true` when CUDA reports an available device.  `false` if
 *         CUDA is unavailable or `NOVA_HAS_CUDA` is not defined.
 *
 * @note Thread-safe.  Does not print diagnostics (verbose is `false`).
 *
 * @see is_device_available()
 * @see is_hip_available()
 * @see DeviceKind
 */
bool is_cuda_available(void);

/**
 * @brief Check whether HIP is available.
 *
 * @details
 * Convenience wrapper that calls @ref is_device_available() with
 * `HIP_DEVICE`.  Equivalent to:
 * @code{.c}
 * is_device_available(HIP_DEVICE, false);
 * @endcode
 *
 * @return `true` when HIP reports an available device.  `false` if
 *         HIP is unavailable or `NOVA_HAS_HIP` is not defined.
 *
 * @note Thread-safe.  Does not print diagnostics (verbose is `false`).
 *
 * @see is_device_available()
 * @see is_cuda_available()
 * @see DeviceKind
 */
bool is_hip_available(void);

/**
 * @brief Transfer memory between device backends.
 *
 * @details
 * High-level memory transfer function that routes the copy through the
 * correct backend at run time.  The function:
 * 1. Looks up the @ref TransferKind from @ref transf_dispatch using
 *    the `(src, dst)` pair as indices.
 * 2. Forwards the request to `device_memcpy_c()` (declared in
 *    @ref cpp_ffi.h) with the resolved transfer kind.
 *
 * The dispatch table is initialised at program startup by a
 * `__attribute__((constructor))` function in @ref device.c, so it is
 * always ready when this function is called.
 *
 * @param[in]  src       Source device placement.  Determines the
 *                       source memory space.
 * @param[in]  dst       Target device placement.  Determines the
 *                       destination memory space.
 * @param[in]  src_buf   Pointer to the source buffer.  Must be valid
 *                       for at least @p bytes bytes in the source
 *                       memory space.
 * @param[out] dst_buf   Pointer to the destination buffer.  Must be
 *                       valid for at least @p bytes bytes in the
 *                       destination memory space.
 * @param[in]  is_pinned Whether the host-side buffer is
 *                       pinned/page-locked.  This affects whether the
 *                       runtime uses synchronous or asynchronous
 *                       transfer.
 * @param[in]  bytes     Number of bytes to transfer.  Must be > 0.
 *
 * @return @ref DeviceStatus with `code` 0 on success, or an error
 *         status with a descriptive `message` on failure.
 *
 * @pre  Both @p src_buf and @p dst_buf must point to valid memory
 *       regions of at least @p bytes.
 * @pre  @p bytes must be greater than zero.
 * @post On success, @p dst_buf contains a copy of @p src_buf.
 * @post On failure, the source and destination buffers are unchanged.
 *
 * @warning If @p src and @p dst are both `DEVICE_CPU`, the dispatch
 *          table entry is `0` (zero-initialised but invalid), which
 *          may cause undefined behaviour.  Use `memcpy()` or
 *          @ref deepcopy() for host-to-host copies.
 *
 * @note Thread-safe.  The dispatch table is read-only after
 *       initialisation, and `device_memcpy_c()` is expected to be
 *       thread-safe.
 *
 * @see device_memcpy_c()  Low-level C-callable copy wrapper.
 * @see transf_dispatch    Lookup table mapping device pairs to
 *                         transfer directions.
 * @see TransferKind       Enum encoding copy directions.
 */
DeviceStatus transfer_to(Device src, Device dst, const void *src_buf,
                         void *dst_buf, bool is_pinned, size_t bytes);

/**
 * @brief Return the active device id (CUDA or HIP).
 *
 * @details
 * Queries the backend-specific detection modules to determine which
 * GPU runtime is active and returns its device id.  The function
 * checks CUDA first, then HIP, and returns `-1` if neither is
 * available.
 *
 * The returned id is a 0-based index into the device list of the
 * active runtime.  Currently only the first device (id `0`) is
 * supported by the detection layer.
 *
 * @return Active device id (0-based), or `-1` when no GPU device is
 *         available or detection has not yet been performed.
 *
 * @note The return value is only meaningful after at least one of
 *       @ref is_cuda_available() or @ref is_hip_available() has
 *       returned `true`.
 *
 * @see is_cuda_available()
 * @see is_hip_available()
 * @see get_cuda_device_id()  CUDA-specific device id accessor.
 * @see get_hip_device_id()   HIP-specific device id accessor.
 */
int get_device_id(void);

/**
 * @brief Print detailed or concise device information to stdout.
 *
 * @details
 * Queries the specified backend for device 0 properties and prints
 * them to stdout using ANSI colour codes.
 *
 * When @p verbose is `false`, a concise two-line summary is printed
 * per device:
 * @code
 * -- [CUDA] Device 0: NVIDIA GeForce RTX 5070 | Compute 12.0 | 12.0 GiB | 48
 * SMs
 * -- [CUDA] Driver v13.3 | Runtime v13.3
 * @endcode
 *
 * When @p verbose is `true`, a detailed multi-line block is printed
 * with name, compute capability, memory, SMs, warp size, thread
 * limits, and driver/runtime versions.
 *
 * This function can be called at any time — it does not require a
 * prior call to @ref is_device_available().  If the requested backend
 * is unavailable, the function silently returns without printing.
 *
 * @param[in] kind     Backend to query.  Must be `CUDA_DEVICE` or
 *                     `HIP_DEVICE`.  `NULL_DEVICE` is a no-op.
 * @param[in] verbose  If `true`, print the detailed block.  If
 *                     `false`, print the concise summary.
 *
 * @note Thread-safe.  Delegates to backend-specific print functions.
 *
 * @see is_device_available()
 * @see is_cuda_available()
 * @see is_hip_available()
 */
void print_device_info(DeviceKind kind, bool verbose);

/**
 * @brief Return the cached device kind from the last detection.
 *
 * @details
 * After @ref is_device_available() has been called at least once,
 * this function returns the @ref DeviceKind that was detected
 * (e.g., `CUDA_DEVICE` or `HIP_DEVICE`).  If detection has not
 * been performed yet, or if no GPU was found, returns
 * `NULL_DEVICE`.
 *
 * The returned value is the cached result of the one-shot
 * detection performed by @ref is_device_available().
 *
 * @return The detected @ref DeviceKind, or `NULL_DEVICE` if no
 *         detection has occurred or no GPU was found.
 *
 * @see is_device_available()
 * @see was_device_detection_done()
 * @see DeviceKind
 */
DeviceKind get_detected_device_kind(void);

/**
 * @brief Check whether device detection has already been performed.
 *
 * @details
 * Returns `true` after the first call to @ref is_device_available()
 * (or its convenience wrappers @ref is_cuda_available() /
 * @ref is_hip_available()) has completed.  Useful for guarding
 * one-time initialisation that depends on the detection result.
 *
 * @return `true` if detection has been performed at least once,
 *         `false` otherwise.
 *
 * @see is_device_available()
 * @see get_detected_device_kind()
 */
bool was_device_detection_done(void);

#ifdef __cplusplus
}
#endif
