/**
 * @file device.h
 * @brief Device abstraction for tensor placement and migration.
 *
 * @details
 * Defines the Device enumeration and the API for querying the current
 * global device, moving tensors and gradients between devices, and
 * inspecting per-tensor device placement.
 *
 * ## Device Tiers
 * | Device     | Value | Description                        |
 * |------------|-------|------------------------------------|
 * | DEVICE_CPU | 0     | Host memory (default)              |
 * | DEVICE_GPU | 1     | Accelerator memory (CUDA/ROCm)     |
 * | DEVICE_META| 2     | Placeholder with no backing storage|
 *
 * META tensors are used for shape inference and graph construction
 * without allocating actual data buffers.
 *
 * @see tensor.h  Tensor structure embedding a Device field.
 * @see backend.h Backend-specific GPU implementations.
 */

#pragma once

#include <ncore/macros.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Target device for tensor data placement.
 *
 * Determines where a tensor's backing storage is allocated and which
 * compute kernels can operate on it.
 */
typedef enum ATTR(packed) {
  DEVICE_CPU = 0, ///< Host memory; accessible by all CPU kernels.
  DEVICE_GPU = 1, ///< Accelerator memory; requires GPU-capable backend.
  DEVICE_META = 2 ///< Placeholder; no data buffer is allocated.
} Device;

struct Tensor;
typedef struct Tensor Tensor;
typedef Tensor *TensorGrad;

/**
 * @brief Get the current global default device.
 *
 * Returns a pointer to a thread-global Device value that controls
 * where newly created tensors are placed when no explicit device is
 * specified.
 *
 * @return Pointer to the current global Device.
 */
const Device *get_current_global_device();

/**
 * @brief Set the current global default device.
 *
 * Updates the thread-global default so that subsequent tensor creation
 * calls use the specified device unless overridden.
 *
 * @param device New default device.
 */
void set_current_gloval_device_to_(Device device);

/**
 * @brief Query the device on which a tensor currently resides.
 *
 * @param ten Tensor to inspect.
 * @return The Device associated with the tensor's storage.
 */
Device get_current_device_from(const Tensor *ten);

/**
 * @brief Move a tensor's data buffer to a different device.
 *
 * Allocates new storage on the target device, copies the existing data,
 * and updates the tensor's device field.  If the tensor is already on
 * the target device this is a no-op.
 *
 * @param device Target device.
 * @param ten    Tensor to move.
 * @return Status code (0 on success, non-zero on failure).
 */
int move_tensor_to_(Device device, Tensor *ten);

/**
 * @brief Move a gradient tensor's data buffer to a different device.
 *
 * Same semantics as move_tensor_to_() but operates on a TensorGrad.
 *
 * @param device Target device.
 * @param grad   Gradient tensor to move.
 * @return Status code (0 on success, non-zero on failure).
 */
int move_grad_to_(Device device, TensorGrad grad);

#ifdef __cplusplus
}
#endif
