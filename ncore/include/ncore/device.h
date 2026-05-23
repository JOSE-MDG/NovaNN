/**
 * @file device.h
 * @brief Device abstraction for tensor placement and backend transfers.
 *
 * @details
 * Defines the Device enumeration and the API for querying the current
 * compute backend, checking runtime availability, and dispatching memory
 * transfers through CUDA or HIP.
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

/**
 * @brief Runtime backend used for GPU operations.
 *
 * @var CUDA_DEVICE NVIDIA CUDA runtime backend.
 * @var HIP_DEVICE AMD ROCm HIP runtime backend.
 * @var NULL_DEVICE No supported GPU backend detected.
 */
typedef enum ATTR(packed) {
  CUDA_DEVICE = 0,
  HIP_DEVICE = 1,
  NULL_DEVICE = 2
} DeviceKind;

/**
 * @brief Device-agnostic memory transfer direction.
 *
 * Mirrors the copy directions supported by CUDA and HIP without exposing
 * runtime-specific enums in the public C API.
 *
 * @var deviceMemcpyHostToDevice Copy from host memory into device memory.
 * @var deviceMemcpyDeviceToHost Copy from device memory into host memory.
 * @var deviceMemcpyDeviceToDevice Copy between two device-memory buffers.
 */
typedef enum ATTR(packed) {
  deviceMemcpyHostToDevice = 1,
  deviceMemcpyDeviceToHost = 2,
  deviceMemcpyDeviceToDevice = 3
} TransferKind;

struct Tensor;
typedef struct Tensor Tensor;
typedef Tensor *TensorGrad;

/**
 * @brief Check whether a GPU backend is available.
 *
 * @param kind Requested backend kind.
 * @param verbose If true, backend detection may print runtime diagnostics.
 * @return true when the requested backend is available, false otherwise.
 */
bool is_device_available(DeviceKind kind, bool verbose);

/**
 * @brief Check whether CUDA is available.
 *
 * @return true when CUDA reports an available device.
 */
bool is_cuda_available(void);

/**
 * @brief Check whether HIP is available.
 *
 * @return true when HIP reports an available device.
 */
bool is_hip_available(void);

/**
 * @brief Transfer a memory buffer to the target device backend.
 *
 * @param targe_device Target tensor-placement device.
 * @param kind Transfer direction.
 * @param src_buf Source buffer.
 * @param dst_buf Destination buffer.
 * @return 0 on success, or a backend-specific error code.
 */
int transfer_to(Device targe_device, TransferKind kind, const void *src_buf,
                void *dst_buf);

/**
 * @brief Probe HIP runtime availability.
 * @param log If true, print runtime error details.
 * @return true when a HIP device is available.
 */
extern bool is_hip_device_available(bool log);

/**
 * @brief Probe HIP runtime unavailability.
 * @param log If true, print runtime error details.
 * @return true when no HIP device is available.
 */
extern bool is_hip_device_not_available(bool log);

/**
 * @brief Probe CUDA runtime unavailability.
 * @param log If true, print runtime error details.
 * @return true when no CUDA device is available.
 */
extern bool is_cuda_device_not_available(bool log);

/**
 * @brief Probe CUDA runtime availability.
 * @param log If true, print runtime error details.
 * @return true when a CUDA device is available.
 */
extern bool is_cuda_device_available(bool log);

/**
 * @brief Return the active HIP device id.
 * @return Device id, or -1 when HIP is unavailable.
 */
extern int get_hip_device_id(void);

/**
 * @brief Return the active CUDA device id.
 * @return Device id, or -1 when CUDA is unavailable.
 */
extern int get_cuda_device_id(void);

#ifdef __cplusplus
}
#endif
