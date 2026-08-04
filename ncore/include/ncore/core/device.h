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
 * @li Device enumeration: The @ref Device_ enum classifies where a
 *   tensor's backing storage lives (CPU, GPU, or META).
 * @li Backend identification: The @ref DeviceKind enum identifies
 *   which GPU runtime (CUDA, HIP, or none) is in use.
 * @li Transfer direction: The @ref TransferKind enum encodes copy
 *   directions without exposing runtime-specific types.
 * @li Result reporting: The @ref novaStatus_t struct carries an
 *   error code and human-readable message from memory operations.
 * @li Backend probes: @ref is_device_available(), @ref
 *   is_cuda_available(), and @ref is_hip_available() query runtime
 *   availability at run time.
 * @li Memory transfers: @ref transfer_to() dispatches inter-device
 *   copies through the correct backend at run time.
 * @li Device identification: @ref get_device_id() returns the
 *   active device id for the detected backend.
 *
 * @section device-tiers Device Tiers
 *
 * @li @c DEVICE_CPU — 0 — Host memory (default).
 * @li @c DEVICE_GPU — 1 — Accelerator memory (CUDA/ROCm).
 * @li @c DEVICE_META — 2 — Placeholder with no backing storage.
 *
 * META tensors are used for shape inference and graph construction
 * without allocating actual data buffers.  They carry dtype and shape
 * metadata but no memory.
 *
 * @section usage-example Usage Example
 *
 * @code{.cpp}
 * #include <ncore/core/device.h>
 *
 * // Check if any GPU backend is available.
 * if (is_device_available(CUDA_DEVICE, true)) {
 *     std::cout << "CUDA device found" << "\n";
 * }
 *
 * // Transfer a buffer from host to device.
 * novaStatus_t status = transfer_to(DEVICE_CPU, DEVICE_GPU, src, dst, n);
 * if (status.err != 0) {
 *     std::cout << "Transfer failed: " << status.message << "\n";
 * }
 * @endcode
 *
 * @see tensor.h      Tensor structure embedding a @ref Device_ field.
 * @see ffi.hpp       C-callable device-memory copy wrapper used by
 *                    @ref transfer_to().
 * @see DetectCudaDevice.cpp CUDA-specific detection implementation.
 * @see DetectHipDevice.cpp  HIP-specific detection implementation.
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
 * @details
 * Determines where a tensor's backing storage is allocated and which
 * compute kernels can operate on it.  Every tensor in the NovaNN
 * framework carries a @ref Device_ field that specifies its memory
 * location.
 *
 * The three tiers form a strict hierarchy:
 * @li @c DEVICE_CPU is the default and universally accessible.
 * @li @c DEVICE_GPU requires a capable runtime (CUDA or HIP) and is
 *   where accelerated computation happens.
 * @li @c DEVICE_META carries metadata only — no data buffer is allocated.
 *   This is useful for shape inference, graph construction, and
 *   lazy evaluation patterns where the actual memory location is
 *   determined later.
 *
 * @note This enum uses a @c uint8_t underlying type (C23) to minimise its
 *       footprint in structs that are serialised or copied frequently.
 *
 * @see DeviceKind     Identifies which GPU runtime is in use.
 * @see novaStatus_t   Result type for device memory operations.
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
 * @note This enum uses a @c uint8_t underlying type (C23) to minimise its
 *       footprint in structs that are serialised or copied frequently.
 *
 * @see Device_          Target device for tensor data placement.
 * @see is_device_available()
 * @see is_cuda_available()
 * @see is_hip_available()
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
 * @details
 * Mirrors the copy directions supported by CUDA and HIP without
 * exposing runtime-specific enums in the public C API.  The values
 * are chosen to match the @c cudaMemcpyKind and @c hipMemcpyKind
 * enums, so no translation is needed when dispatching to the
 * underlying runtime.
 *
 * The dispatch table @ref transf_dispatch (in @ref device.c) maps
 * pairs of @c (src Device_, dst Device_) to a @c TransferKind value,
 * which is then passed to the C-callable @c deviceTransfer() wrapper.
 *
 * @note This enum uses a @c uint8_t underlying type (C23) to minimise its
 *       footprint in structs that are serialised or copied frequently.
 *
 * @see novaStatus_t   Result type for device memory operations.
 * @see transfer_to()  High-level transfer function using this enum.
 * @see transf_dispatch  Lookup table mapping device pairs to directions.
 */
typedef enum TransferKind : uint8_t {
  deviceMemcpyHostToDevice = 1,  ///< Copy from host memory into device memory.
  deviceMemcpyDeviceToHost = 2,  ///< Copy from device memory into host memory.
  deviceMemcpyDeviceToDevice = 3 ///< Copy between two device-memory buffers.
} TransferKind;

/**
 * @struct Tensor
 * @brief Forward declaration of the Tensor type.
 *
 * @details
 * The Tensor struct is defined in @c tensor.h and carries a @ref Device_
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
 * @li @c CUDA_DEVICE — @ref isCudaDeviceAvailable() from the native
 *   CUDA backend.
 * @li @c HIP_DEVICE — @ref isHipDeviceAvailable() from the native
 *   HIP backend.
 * @li @c NULL_DEVICE or any other value — returns @c false.
 *
 * When the corresponding backend macro (@c NOVA_HAS_CUDA or
 * @c NOVA_HAS_HIP) is not defined, the function returns @c false
 * without querying the runtime.
 *
 * @section one-shot-caching One-shot caching
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
 * @param[in] verbose  If @c true, backend detection may print runtime
 *                     diagnostics to @c stdout.  Pass @c false for
 *                     silent operation.
 *
 * @return @c true when the requested backend reports an available
 *         device.  @c false otherwise.
 *
 * @note Thread-safe.  Delegates to thread-safe backend detection
 *       functions.  The cached result is protected by a mutex and
 *       a @c call_once/@c InitOnceExecuteOnce initialisation guard.
 *
 * @see is_cuda_available()   Convenience wrapper for @c CUDA_DEVICE.
 * @see is_hip_available()    Convenience wrapper for @c HIP_DEVICE.
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
 * @c CUDA_DEVICE.  Equivalent to:
 * @code{.cpp}
 * is_device_available(CUDA_DEVICE, false);
 * @endcode
 *
 * @return @c true when CUDA reports an available device.  @c false if
 *         CUDA is unavailable or @c NOVA_HAS_CUDA is not defined.
 *
 * @note Does not print diagnostics (verbose is @c false).
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
 * @c HIP_DEVICE.  Equivalent to:
 * @code{.c}
 * is_device_available(HIP_DEVICE, false);
 * @endcode
 *
 * @return @c true when HIP reports an available device.  @c false if
 *         HIP is unavailable or @c NOVA_HAS_HIP is not defined.
 *
 * @note Does not print diagnostics (verbose is @c false).
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
 * @li 1. Looks up the @ref TransferKind from @ref transf_dispatch using
 *    the @c (src, dst) pair as indices.
 * @li 2. Forwards the request to @c deviceTransfer() (declared in
 *    @ref ffi.hpp) with the resolved transfer kind.
 *
 * The dispatch table is initialised at program startup by an
 * @c INITIALIZE(init_transf_dispatch) function in @ref device.c, so it is
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
 * @param[in]  bytes     Number of bytes to transfer.  Must be > 0.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or an error status with a descriptive @c message on
 *         failure.
 *
 * @pre  Both @p src_buf and @p dst_buf must point to valid memory
 *       regions of at least @p bytes.
 * @pre  @p bytes must be greater than zero.
 * @post On success, @p dst_buf contains a copy of @p src_buf.
 * @post On failure, the source and destination buffers are unchanged.
 *
 * @warning If @p src and @p dst are both @c DEVICE_CPU, the dispatch
 *          table entry is @c 0 (zero-initialised but invalid), which
 *          may cause undefined behaviour.  Use @c memcpy() or
 *          @ref deepcopy() for host-to-host copies.
 *
 * @note Thread-safe.  The dispatch table is read-only after
 *       initialisation, and @c deviceTransfer() is expected to be
 *       thread-safe.
 *
 * @see deviceTransfer()  Low-level C-callable copy wrapper.
 * @see transf_dispatch    Lookup table mapping device pairs to
 *                         transfer directions.
 * @see TransferKind       Enum encoding copy directions.
 */
novaStatus_t transfer_to(Device_ src, Device_ dst, const void *src_buf,
                         void *dst_buf, size_t bytes);

/**
 * @brief Return the active device id (CUDA or HIP).
 *
 * @details
 * Queries the backend-specific detection modules to determine which
 * GPU runtime is active and returns its device id.  The function
 * checks CUDA first, then HIP, and returns @c -1 if neither is
 * available.
 *
 * The returned id is a 0-based index into the device list of the
 * active runtime.  Currently only the first device (id @c 0) is
 * supported by the detection layer.
 *
 * @return Active device id (0-based), or @c -1 when no GPU device is
 *         available or detection has not yet been performed.
 *
 * @note The return value is only meaningful after at least one of
 *       @ref is_cuda_available() or @ref is_hip_available() has
 *       returned @c true.
 *
 * @see is_cuda_available()
 * @see is_hip_available()
 * @see getCudaDeviceId()  CUDA-specific device id accessor.
 * @see getHipDeviceId()   HIP-specific device id accessor.
 */
int get_device_id(void);

/**
 * @brief Print detailed or concise device information to stdout.
 *
 * @details
 * Queries the specified backend for device 0 properties and prints
 * them to stdout using ANSI colour codes.
 *
 * When @p verbose is @c false, a concise two-line summary is printed
 * per device:
 * @code
 * [CUDA] Device 0: NVIDIA GeForce RTX 5070 | Compute 12.0 | 12.0 GiB | 48 SMs
 * [CUDA] Driver v13.3 | Runtime v13.3
 * @endcode
 *
 * When @p verbose is @c true, a detailed multi-line block is printed
 * with name, compute capability, memory, SMs, warp size, thread
 * limits, and driver/runtime versions.
 *
 * This function can be called at any time — it does not require a
 * prior call to @ref is_device_available().  If the requested backend
 * is unavailable, the function silently returns without printing.
 *
 * @param[in] kind     Backend to query.  Must be @c CUDA_DEVICE or
 *                     @c HIP_DEVICE.  @c NULL_DEVICE is a no-op.
 * @param[in] verbose  If @c true, print the detailed block.  If
 *                     @c false, print the concise summary.
 * @return novaStatus_t the result of the operation. On success, set to
 *                     @ref novaSuccess.  On failure, set to the
 *                     appropriate error code.
 *
 * @see is_device_available()
 * @see is_cuda_available()
 * @see is_hip_available()
 */
novaStatus_t print_device_info(DeviceKind kind, bool verbose);

/**
 * @brief Return the cached device kind from the last detection.
 *
 * @details
 * After @ref is_device_available() has been called at least once,
 * this function returns the @ref DeviceKind that was detected
 * (e.g., @c CUDA_DEVICE or @c HIP_DEVICE).  If detection has not
 * been performed yet, or if no GPU was found, returns
 * @c NULL_DEVICE.
 *
 * The returned value is the cached result of the one-shot
 * detection performed by @ref is_device_available().
 *
 * @return The detected @ref DeviceKind, or @c NULL_DEVICE if no
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
 * Returns @c true after the first call to @ref is_device_available()
 * (or its convenience wrappers @ref is_cuda_available() /
 * @ref is_hip_available()) has completed.  Useful for guarding
 * one-time initialisation that depends on the detection result.
 *
 * @return @c true if detection has been performed at least once,
 *         @c false otherwise.
 *
 * @see is_device_available()
 * @see get_detected_device_kind()
 */
bool was_device_detection_done(void);

#ifdef __cplusplus
}
#endif
