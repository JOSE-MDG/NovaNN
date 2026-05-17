/**
 * @file backend.h
 * @brief Compute backend abstraction for tensor operations.
 *
 * @details
 * Defines the Backend enumeration and the API for querying and
 * selecting the active compute backend at runtime.  The backend
 * determines which kernel implementations (CUDA, ROCm, oneDNN, etc.)
 * are dispatched for tensor operations.
 *
 * ## Backend Tiers
 * | Backend | Value | Description                              |
 * |---------|-------|------------------------------------------|
 * | CUDA    | 0     | NVIDIA GPU via CUDA runtime              |
 * | Rocm    | 1     | AMD GPU via ROCm/HIP                     |
 * | oneDNN  | 2     | Intel oneAPI Deep Neural Network Library |
 * | Generic | 3     | Portable scalar/SSE fallback             |
 * | Meta    | 4     | Shape-only; no computation performed     |
 *
 * @see device.h  Device placement (CPU vs GPU vs META).
 * @see simd.h    CPU SIMD capability detection.
 */

#pragma once

#include <ncore/macros.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute backend for tensor operations.
 *
 * Selects which kernel implementation is used for arithmetic,
 * reductions, and memory operations on GPU-accelerated tensors.
 */
typedef enum ATTR(packed) {
  CUDA = 0,    ///< NVIDIA CUDA runtime (cuBLAS, cuDNN, custom kernels).
  Rocm = 1,    ///< AMD ROCm/HIP runtime (rocBLAS, MIOpen).
  oneDNN = 2,  ///< Intel oneDNN (MKL-DNN) for CPU/GPU fused ops.
  Generic = 3, ///< Portable fallback (scalar or SIMD-optimised).
  Meta = 4,    ///< No-op backend; used for shape inference only.
} Backend;

/**
 * @brief Get the currently active backend.
 *
 * Returns a pointer to a global Backend value indicating which compute
 * backend is currently selected for tensor operations.
 *
 * @return Pointer to the current running Backend.
 */
const Backend *get_current_running_backend();

/**
 * @brief Set the backend for subsequent tensor operations.
 *
 * Updates the global execution backend.  The caller is responsible for
 * ensuring that the selected backend is available on the current
 * hardware (see is_backend_available()).
 *
 * @param backend Desired compute backend.
 * @return true on success, false if the backend is unavailable.
 */
bool set_next_execution_backend_to_(Backend backend);

/**
 * @brief Check whether a backend is available on the current hardware.
 *
 * Performs a runtime probe (e.g., CUDA driver init, ROCm hipGetDeviceCount)
 * to determine if the requested backend can be used.
 *
 * @param backend Backend to probe.
 * @return true if the backend is available, false otherwise.
 */
bool is_backend_available(Backend backend);

#ifdef __cplusplus
}
#endif
