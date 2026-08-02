/**
 * @file backend.h
 * @brief Backend enumeration and runtime backend selection API for the
 *        NovaNN compute layer.
 *
 * @details
 * This header defines the abstract @ref Backend enumeration and the public
 * functions required to query, delegate, and inspect hardware acceleration
 * backends at runtime.  The NovaNN runtime supports multiple vendor-specific
 * compute libraries, and this module is the single entry point for backend
 * management.
 *
 * @section architecture Architecture
 *
 * The runtime follows a strategy pattern for backend selection:
 *
 * @li 1. At process startup, the application calls @ref is_backend_available()
 *    to probe which backends are compiled in and have matching hardware.
 * @li 2. The application then calls @ref delegate_execution() to set the
 *    preferred backend for upcoming tensor operations.
 * @li 3. When the tensor dispatcher processes a computation, it opts for the
 *    preferred backend (if available), falling back to another option
 *    otherwise.
 * @li 4. The preferred backend can be queried at any time via
 *    @ref get_current_running_backend().
 *
 * @section backend-catalogue Backend Catalogue
 *
 * @li @c CUDA — NVIDIA cuBLAS / cuDNN — NVIDIA GPUs
 * @li @c HIP — AMD rocBLAS / MIOpen — AMD GPUs
 * @li @c CPU — Scalar / SIMD fallback — Any CPU
 * @li @c Meta — No-op (shape inference) — N/A
 * @li @c Miopen — AMD MIOpen — AMD GPUs (convolution)
 * @li @c OneDNN — Intel oneDNN / MKL-DNN — Intel/AMD CPUs / GPUs
 * @li @c Generic — Scalar / SIMD fallback — Any platform
 *
 * @section thread-safety Thread Safety
 *
 * @li @ref get_current_running_backend() returns a pointer to a process-lifetime
 *   object; it is safe to call from any thread.
 * @li @ref delegate_execution() is safe to call from any thread.  Each call
 *   updates the preferred backend for subsequent dispatch; concurrent calls
 *   may race on the preference, but no undefined behaviour occurs.
 * @li @ref is_backend_available() is a pure query with no side effects and is
 *   safe to call from any thread.
 *
 * @note This is a C header.  When included from C++ code, the declarations
 *       are wrapped in @c extern "C" for ABI compatibility.
 *
 * @see simd.h   CPU SIMD capability detection (orthogonal to backend
 * selection).
 * @see device.h Low-level device enumeration and memory transfers.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum Backend
 * @brief Identifies the hardware/software compute backend used for kernel
 *        execution in the NovaNN runtime.
 *
 * @details
 * Each enumerator maps to a vendor-provided acceleration library that
 * handles matrix multiplication, convolution, and other tensor operations.
 * The runtime selects an appropriate backend at startup based on compiled-in
 * support and available hardware.
 *
 * The @c Generic backend serves as a portable fallback that uses scalar or
 * SIMD-optimised kernels and is always available.  The @c Meta backend
 * performs no computation and exists solely for shape inference and graph
 * construction — it never allocates device memory or launches kernels.
 *
 * @note This enum uses a @c uint8_t underlying type (C23).  The enumerator
 *       values are explicit to guarantee a stable ABI across compilation
 *       units.  Do not renumber or reorder existing entries; append new
 *       backends after @c Meta.
 *
 * @see delegate_execution()
 * @see is_backend_available()
 */
typedef enum Backend : uint8_t {
  CUDA,    ///< NVIDIA CUDA runtime (cuBLAS, cuDNN, custom kernels).
  HIP,     ///< AMD HIP/ROCm runtime (rocBLAS, MIOpen).
  CPU,     ///< CPU-only execution with scalar or SIMD-optimised kernels.
  Meta,    ///< No-op backend; used for shape inference only.
  Miopen,  ///< AMD MIOpen library for GPU-accelerated convolutions.
  OneDNN,  ///< Intel oneDNN (MKL-DNN) for CPU/GPU fused operations.
  Generic, ///< Portable fallback using generic kernels.
} Backend;

/**
 * @brief Return a pointer to the currently preferred backend instance.
 *
 * @details
 * Returns a pointer to the process-lifetime object that represents the
 * preferred compute backend.  The pointer is valid for the entire lifetime
 * of the process and must not be freed or modified by the caller.
 *
 * If @ref delegate_execution() has not been called yet, the return value
 * is implementation-defined — typically @c nullptr or a pointer to the
 * @c Generic backend descriptor.
 *
 * This function is intended for diagnostic logging and runtime introspection,
 * not for hot-path dispatch.  Backend selection should be performed once at
 * initialisation time.
 *
 * @return Pointer to the preferred @ref Backend value.  The returned pointer
 *         is valid for the lifetime of the process and must not be freed.
 *
 * @note Thread-safe.  The returned pointer references a process-lifetime
 *       object that is never invalidated.
 *
 * @see delegate_execution()  Sets the preferred backend.
 * @see is_backend_available()  Probes backend availability.
 */
Backend *get_current_running_backend(void);

/**
 * @brief Delegate subsequent tensor operations to the specified backend.
 *
 * @details
 * Sets the preferred backend for the next tensor operation dispatch.  When
 * the tensor dispatcher processes a computation, it will opt for @p backend
 * if it is available (as reported by @ref is_backend_available()), falling
 * back to another option otherwise.
 *
 * This function does not allocate resources or "activate" the backend in a
 * hardware sense — it is a hint to the dispatcher about the caller's
 * preference.  The actual backend used for a given operation depends on
 * availability at dispatch time.
 *
 * @section behaviour Behaviour
 *
 * @li 1. Store @p backend as the preferred target for subsequent dispatch.
 * @li 2. Return @c true if the backend is available and was set as preferred.
 * @li 3. Return @c false if the backend is unavailable or @p backend is not a
 *    valid enumerator.
 *
 * @param[in] backend  The backend to prefer for upcoming tensor operations.
 *                     Must be a value defined in the @ref Backend enumeration.
 *
 * @return @c true if @p backend is available and was set as the preferred
 *         dispatch target.
 * @return @c false if the requested backend is unavailable on this platform
 *         or @p backend is not a valid enumerator.
 *
 * @pre  @p backend must be a value defined in the @ref Backend enumeration.
 * @post On success, the dispatcher will prefer @p backend for the next
 *       tensor operation when it is available.
 *
 * @note This function may be called multiple times; each call updates the
 *       preference.  The last successful call determines the backend used
 *       by the next dispatch.
 *
 * @see is_backend_available()  Check availability before delegating.
 * @see get_current_running_backend()  Query the current preferred backend.
 */
bool delegate_execution(Backend backend);

/**
 * @brief Check whether the given backend is available on this platform.
 *
 * @details
 * Probes the system to determine whether the specified backend can be used.
 * A backend is considered available when:
 *
 * @li The corresponding vendor library is linked into the process (or can be
 *   loaded dynamically).
 * @li The required hardware is detected at runtime (e.g. an NVIDIA GPU for
 *   @c CUDA).
 *
 * The @c Generic backend is always available on every platform.  The @c Meta
 * backend is always available as it performs no computation.
 *
 * This function has no side effects and does not allocate resources.
 * It is safe to call during early initialisation, before any backend has
 * been selected.
 *
 * @param[in] backend  The backend to query.  Must be a valid enumerator
 *                     from the @ref Backend enumeration.
 *
 * @return @c true if the backend is available for use on this platform.
 * @return @c false if the backend is not supported, the vendor library is
 *         missing, or the required hardware was not detected.
 *
 * @note Thread-safe and reentrant.  No global state is modified.
 *
 * @see delegate_execution()  Set the preferred backend.
 * @see get_current_running_backend()  Query the current preferred backend.
 */
bool is_backend_available(Backend backend);

#ifdef __cplusplus
}
#endif
