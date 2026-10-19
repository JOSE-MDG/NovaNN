/**
 * @file backend.h
 * @brief Backend enum and runtime backend selection for the NovaNN
 *        compute layer.
 *
 * Selection flow: probe with @ref is_backend_available(), delegate
 * with @ref delegate_execution(), then dispatch prefers that backend
 * with a fallback to another option.
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
 * @enum Backend_
 * @brief Hardware or software backend used for kernel execution.
 *
 * Enumerator values are explicit for a stable ABI across compilation
 * units. Do not renumber or reorder existing entries; append new
 * backends after @c Meta.
 */
typedef enum Backend_ : uint8_t {
  CUDA,    ///< NVIDIA CUDA runtime (cuBLAS, cuDNN, custom kernels).
  HIP,     ///< AMD HIP/ROCm runtime (rocBLAS, MIOpen).
  CPU,     ///< CPU-only execution with SIMD-optimized kernels.
  Meta,    ///< No-op backend; shape only, never allocates or launches.
  Miopen,  ///< AMD MIOpen library for GPU-accelerated convolutions.
  OneDNN,  ///< Intel oneDNN (MKL-DNN) for CPU fused operations.
  Generic, ///< Portable scalar/SIMD fallback, always available.
} Backend_;

/**
 * @brief Return a pointer to the currently preferred backend instance.
 *
 * The pointer targets a process-lifetime object, never free or
 * modify it. Before @ref delegate_execution() the value is
 * implementation-defined (typically @c nullptr or the @c Generic
 * descriptor).
 *
 * Intended for diagnostics and introspection, not for hot-path
 * dispatch; select the backend once at initialization time.
 *
 * @return Pointer to the preferred @ref Backend_ value, valid for the
 *         lifetime of the process.
 */
Backend_ *get_current_running_backend(void);

/**
 * @brief Delegate subsequent tensor operations to the specified backend.
 *
 * Only a preference: dispatch still consults availability, so an
 * unavailable backend leads to a fallback one. Safe to call multiple
 * times; the last call determines the preference.
 *
 * @param[in] backend  The backend to prefer for upcoming tensor
 *                     operations.
 *
 * @return @c true if @p backend is available and was set as the
 *         preferred dispatch target, @c false if it is unavailable on
 *         this platform or not a valid enumerator.
 */
bool delegate_execution(Backend_ backend);

/**
 * @brief Check whether the given backend is available on this platform.
 *
 * A backend is available when its vendor library is linked (or
 * loadable) and the required hardware is detected. @c Generic and
 * @c Meta are always available. Pure query with no side effects, safe
 * during early initialization.
 *
 * @param[in] backend  The backend to query.
 * @return @c true if the backend can be used on this platform.
 */
bool is_backend_available(Backend_ backend);

#ifdef __cplusplus
}
#endif
