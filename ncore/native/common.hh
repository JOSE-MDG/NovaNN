/**
 * @file common.hh
 * @brief Shared CUDA and HIP runtime helpers for the memory backends.
 *
 * Private to the native backends. The two scopes expose the same
 * error-mapping and device helpers, and exactly one is active per
 * build. Device work never runs on the null stream: rocclr cancels
 * pending null-stream kernels when a stream is created, so every
 * backend operation resolves the device engine stream instead.
 *
 * @see CudaIO.cpp
 * @see CudaAllocator.cpp
 * @see HipIO.cpp
 * @see HipAllocator.cpp
 */

#pragma once

#if defined(NOVA_HAS_CUDA) && __has_include(<cuda_runtime_api.h>)

#include <cuda_runtime_api.h>

#include <mutex>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/memory/ffi.hpp>

#include "cuda/DetectCudaDevice.hpp"

/**
 * @brief Map a CUDA error code to a @ref novaError_t.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return The corresponding Nova error category.
 */
static inline novaError_t mapError(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return novaSuccess;
  case cudaErrorInvalidValue:
    return novaInvalidValue;
  case cudaErrorInvalidDevice:
    return novaInvalidDeviceId;
  case cudaErrorMemoryAllocation:
    return novaOutOfMemory;
  case cudaErrorNotSupported:
    return novaNotImplemented;
  case cudaErrorExternalDevice:
    return novaExternalDeviceError;
  case cudaErrorInvalidResourceHandle:
    return novaInvalidResourceHandle;
  default:
    return novaNotImplemented;
  }
}

/**
 * @brief Map a CUDA runtime error to a @ref novaStatus_t.
 *
 * @details
 * Converts @c cudaError_t codes returned by @c cudaMemcpyAsync
 * and @c cudaStreamSynchronize into the project-standard
 * @ref novaStatus_t format.  Each error code is mapped to a
 * unique @ref novaError_t category, and the human-readable
 * error string is selected from the Nova status table via
 * @ref nova_get_error_msg.
 *
 * The mapped codes are: @c cudaSuccess → @ref novaSuccess,
 * @c cudaErrorInvalidValue → @ref novaInvalidValue,
 * @c cudaErrorInvalidDevice → @ref novaInvalidDeviceId,
 * @c cudaErrorMemoryAllocation → @ref novaOutOfMemory,
 * @c cudaErrorNotSupported → @ref novaNotImplemented,
 * @c cudaErrorExternalDevice → @ref novaExternalDeviceError,
 * @c cudaErrorInvalidMemcpyDirection → @ref novaInvalidTransfDirection,
 * @c cudaErrorInvalidResourceHandle → @ref novaInvalidResourceHandle,
 * and every other error → @ref novaNotImplemented.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return @ref novaStatus_t with the mapped @ref novaError_t code and
 *         the message string obtained from @ref nova_get_error_msg.
 */
static inline novaStatus_t mapError2Status(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return {.err = novaSuccess,
            .message = nova_get_error_msg(novaSuccess, nullptr)};
  case cudaErrorInvalidValue:
    return {.err = novaInvalidValue,
            .message = nova_get_error_msg(novaInvalidValue, nullptr)};
  case cudaErrorMemoryAllocation:
    return {.err = novaOutOfMemory,
            .message = nova_get_error_msg(novaOutOfMemory, nullptr)};
  case cudaErrorNotSupported:
    return {.err = novaNotImplemented,
            .message = nova_get_error_msg(novaNotImplemented, nullptr)};
  case cudaErrorInvalidDevice:
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  case cudaErrorExternalDevice:
    return {.err = novaExternalDeviceError,
            .message = nova_get_error_msg(novaExternalDeviceError, nullptr)};
  case cudaErrorInvalidMemcpyDirection:
    return {.err = novaInvalidTransfDirection,
            .message = nova_get_error_msg(novaInvalidTransfDirection, nullptr)};
  case cudaErrorInvalidResourceHandle:
    return {.err = novaInvalidResourceHandle,
            .message = nova_get_error_msg(novaInvalidResourceHandle, nullptr)};
  default:
    return {.err = novaNotImplemented,
            .message = nova_get_error_msg(novaNotImplemented, nullptr)};
  }
}

/**
 * @brief Convert @ref DeviceMemcpyKind to @c cudaMemcpyKind.
 *
 * @details
 * Maps the backend-agnostic @ref DeviceMemcpyKind enum to the
 * CUDA-specific @c cudaMemcpyKind enum used by @c cudaMemcpyAsync.
 * The mapping is: @c deviceMemcpyHostToDevice →
 * @c cudaMemcpyHostToDevice, @c deviceMemcpyDeviceToHost →
 * @c cudaMemcpyDeviceToHost, @c deviceMemcpyDeviceToDevice →
 * @c cudaMemcpyDeviceToDevice, default → @c cudaMemcpyDefault.
 *
 * @param[in] kind  The device-agnostic copy direction.
 *
 * @return The corresponding CUDA memcpy kind.
 */
static inline cudaMemcpyKind mapMemcpyKind(DeviceMemcpyKind kind) {
  switch (kind) {
  case DeviceMemcpyKind::deviceMemcpyHostToDevice:
    return cudaMemcpyHostToDevice;
  case DeviceMemcpyKind::deviceMemcpyDeviceToHost:
    return cudaMemcpyDeviceToHost;
  case DeviceMemcpyKind::deviceMemcpyDeviceToDevice:
    return cudaMemcpyDeviceToDevice;
  default:
    return cudaMemcpyDefault;
  }
}

/**
 * @brief Validates a device ordinal against the compiled and runtime limits.
 *
 * @details
 * Rejects ordinals below zero or at or above
 * @ref NCORE_COMPILE_TIME_MAX_GPUS before querying the runtime, then
 * rejects ordinals the runtime does not expose.
 *
 * @param[in] device_id Runtime device ordinal to validate.
 *
 * @return @ref novaStatus_t; @c novaInvalidDeviceId when the ordinal is out
 *         of range, or the mapped runtime error when the probe fails.
 */
static inline novaStatus_t validateDeviceId(std::int32_t device_id) {
  if (device_id < 0 || device_id >= NCORE_COMPILE_TIME_MAX_GPUS) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  int deviceCount = 0;
  const cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    return mapError2Status(err);
  }
  if (device_id >= deviceCount) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/**
 * @brief Runs an operation with the backend device set to @p device_id.
 *
 * @details
 * Saves the current device, switches to @p device_id if different,
 * runs @p operation, then restores the previous device.  If the switch
 * back fails and the operation itself succeeded, the restore error is
 * reported instead so transient state loss is not masked.
 *
 * @tparam Operation Callable taking no arguments and returning
 *                   @ref novaStatus_t.
 * @param[in]  device_id Target runtime device ordinal.
 * @param[in]  operation Work to execute on @p device_id.
 *
 * @return Status from @p operation, or the device-switch/restore error.
 */
template <typename Operation>
static inline novaStatus_t withDeviceId(std::int32_t device_id,
                                        Operation &&operation) {
  int previousDevice = -1;
  cudaError_t err = cudaGetDevice(&previousDevice);
  if (err != cudaSuccess) {
    return mapError2Status(err);
  }
  const bool changed = previousDevice != device_id;
  if (changed) {
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      return mapError2Status(err);
    }
  }

  novaStatus_t status = operation();
  if (changed) {
    const cudaError_t restoreError = cudaSetDevice(previousDevice);
    if (restoreError != cudaSuccess && status.err == novaSuccess) {
      status = mapError2Status(restoreError);
    }
  }
  return status;
}

/**
 * @brief Validates that @p stream belongs to @p device_id.
 *
 * @details
 * A null stream is always accepted (it denotes the device engine
 * stream).  A non-null stream is queried for its device via
 * @c cudaStreamGetDevice and rejected when it does not match.
 *
 * @param[in]  device_id Expected runtime device ordinal.
 * @param[in]  stream    Stream to validate, or null.
 *
 * @return @ref novaStatus_t; @c novaInvalidDeviceId when the stream
 *         belongs to a different device.
 */
static inline novaStatus_t validateStreamDevice(std::int32_t device_id,
                                                cudaStream_t stream) {
  if (stream == nullptr) {
    return {.err = novaSuccess,
            .message = nova_get_error_msg(novaSuccess, nullptr)};
  }
  int streamDevice = -1;
  const cudaError_t err = cudaStreamGetDevice(stream, &streamDevice);
  if (err != cudaSuccess) {
    return mapError2Status(err);
  }
  if (streamDevice != device_id) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/**
 * @brief Reports whether the backend device supports native memory pools.
 *
 * @details
 * Queries the device attribute for memory-pool support.  When the query
 * fails the error is mapped through @ref mapError2Status and reported;
 * the device is then treated as pool-less by the caller.
 *
 * @param[in]     device_id Runtime device ordinal.
 * @param[in,out] supported Receives @c true when pools are supported.
 *
 * @return @ref novaStatus_t; @c novaInvalidPointer when @p supported is
 *         null.
 */
static inline novaStatus_t supportMemoryPool(std::int32_t device_id,
                                             bool *supported) {
  if (supported == nullptr) {
    return {.err = novaInvalidPointer,
            .message = nova_get_error_msg(novaInvalidPointer, nullptr)};
  }
  int value = 0;
  const cudaError_t err = cudaDeviceGetAttribute(
      &value, cudaDevAttrMemoryPoolsSupported, device_id);
  if (err != cudaSuccess) {
    return mapError2Status(err);
  }
  *supported = value != 0;
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

#elif defined(NOVA_HAS_HIP) && __has_include(<hip/hip_runtime_api.h>)

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

#include <mutex>

#include <ncore/core/status.h>
#include <ncore/memory/ffi.hpp>

#include "hip/DetectHipDevice.hpp"

/**
 * @brief Map a HIP error code to a @ref novaError_t.
 *
 * @details
 * HIP exposes no @c hipErrorExternalDevice equivalent; unsupported or
 * otherwise unclassified failures map to @ref novaNotImplemented.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return The corresponding Nova error category.
 */
static inline novaError_t mapError(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return novaSuccess;
  case hipErrorInvalidValue:
    return novaInvalidValue;
  case hipErrorInvalidDevice:
    return novaInvalidDeviceId;
  case hipErrorOutOfMemory:
    return novaOutOfMemory;
  case hipErrorNotSupported:
    return novaNotImplemented;
  case hipErrorInvalidDevicePointer:
    return novaInvalidPointer;
  case hipErrorInvalidResourceHandle:
    return novaInvalidResourceHandle;
  default:
    return novaNotImplemented;
  }
}

/**
 * @brief Map a HIP runtime error to a @ref novaStatus_t.
 *
 * @details
 * Converts @c hipError_t codes returned by @c hipMemcpyAsync
 * and @c hipStreamSynchronize into the project-standard
 * @ref novaStatus_t format.  Each error code is mapped to a
 * unique @ref novaError_t category, and the human-readable
 * error string is selected from the Nova status table via
 * @ref nova_get_error_msg.
 *
 * HIP exposes no @c hipErrorExternalDevice equivalent, so the
 * mapped codes are: @c hipSuccess → @ref novaSuccess,
 * @c hipErrorInvalidValue → @ref novaInvalidValue,
 * @c hipErrorInvalidDevice → @ref novaInvalidDeviceId,
 * @c hipErrorOutOfMemory → @ref novaOutOfMemory,
 * @c hipErrorNotSupported → @ref novaNotImplemented,
 * @c hipErrorInvalidMemcpyDirection → @ref novaInvalidTransfDirection,
 * @c hipErrorInvalidResourceHandle → @ref novaInvalidResourceHandle,
 * and every other error → @ref novaNotImplemented.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return @ref novaStatus_t with the mapped @ref novaError_t code and
 *         the message string obtained from @ref nova_get_error_msg.
 */
static inline novaStatus_t mapError2Status(hipError_t err) {
  switch (err) {
  case hipSuccess:
    return {.err = novaSuccess,
            .message = nova_get_error_msg(novaSuccess, nullptr)};
  case hipErrorInvalidValue:
    return {.err = novaInvalidValue,
            .message = nova_get_error_msg(novaInvalidValue, nullptr)};
  case hipErrorOutOfMemory:
    return {.err = novaOutOfMemory,
            .message = nova_get_error_msg(novaOutOfMemory, nullptr)};
  case hipErrorNotSupported:
    return {.err = novaNotImplemented,
            .message = nova_get_error_msg(novaNotImplemented, nullptr)};
  case hipErrorInvalidDevice:
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  case hipErrorInvalidMemcpyDirection:
    return {.err = novaInvalidTransfDirection,
            .message = nova_get_error_msg(novaInvalidTransfDirection, nullptr)};
  case hipErrorInvalidResourceHandle:
    return {.err = novaInvalidResourceHandle,
            .message = nova_get_error_msg(novaInvalidResourceHandle, nullptr)};
  default:
    return {.err = novaNotImplemented,
            .message = nova_get_error_msg(novaNotImplemented, nullptr)};
  }
}

/**
 * @brief Convert @ref DeviceMemcpyKind to @c hipMemcpyKind.
 *
 * @details
 * Maps the backend-agnostic @ref DeviceMemcpyKind enum to the
 * HIP-specific @c hipMemcpyKind enum used by @c hipMemcpyAsync.
 * The mapping is: @c deviceMemcpyHostToDevice →
 * @c hipMemcpyHostToDevice, @c deviceMemcpyDeviceToHost →
 * @c hipMemcpyDeviceToHost, @c deviceMemcpyDeviceToDevice →
 * @c hipMemcpyDeviceToDevice, default → @c hipMemcpyDefault.
 *
 * @param[in] kind  The device-agnostic copy direction.
 *
 * @return The corresponding HIP memcpy kind.
 */
static inline hipMemcpyKind mapMemcpyKind(DeviceMemcpyKind kind) {
  switch (kind) {
  case DeviceMemcpyKind::deviceMemcpyHostToDevice:
    return hipMemcpyHostToDevice;
  case DeviceMemcpyKind::deviceMemcpyDeviceToHost:
    return hipMemcpyDeviceToHost;
  case DeviceMemcpyKind::deviceMemcpyDeviceToDevice:
    return hipMemcpyDeviceToDevice;
  default:
    return hipMemcpyDefault;
  }
}

/**
 * @brief Validates a device ordinal against the compiled and runtime limits.
 *
 * @details
 * Rejects ordinals below zero or at or above
 * @ref NCORE_COMPILE_TIME_MAX_GPUS before querying the runtime, then
 * rejects ordinals the runtime does not expose.
 *
 * @param[in] device_id Runtime device ordinal to validate.
 *
 * @return @ref novaStatus_t; @c novaInvalidDeviceId when the ordinal is out
 *         of range, or the mapped runtime error when the probe fails.
 */
static inline novaStatus_t validateDeviceId(std::int32_t device_id) {
  if (device_id < 0 || device_id >= NCORE_COMPILE_TIME_MAX_GPUS) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  int deviceCount = 0;
  const hipError_t err = hipGetDeviceCount(&deviceCount);
  if (err != hipSuccess) {
    return mapError2Status(err);
  }
  if (device_id >= deviceCount) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/**
 * @brief Runs an operation with the backend device set to @p device_id.
 *
 * @details
 * Saves the current device, switches to @p device_id if different,
 * runs @p operation, then restores the previous device.  If the switch
 * back fails and the operation itself succeeded, the restore error is
 * reported instead so transient state loss is not masked.
 *
 * @tparam Operation Callable taking no arguments and returning
 *                   @ref novaStatus_t.
 * @param[in]  device_id Target runtime device ordinal.
 * @param[in]  operation Work to execute on @p device_id.
 *
 * @return Status from @p operation, or the device-switch/restore error.
 */
template <typename Operation>
static inline novaStatus_t withDeviceId(std::int32_t device_id,
                                        Operation &&operation) {
  int previousDevice = -1;
  hipError_t err = hipGetDevice(&previousDevice);
  if (err != hipSuccess) {
    return mapError2Status(err);
  }
  const bool changed = previousDevice != device_id;
  if (changed) {
    err = hipSetDevice(device_id);
    if (err != hipSuccess) {
      return mapError2Status(err);
    }
  }

  novaStatus_t status = operation();
  if (changed) {
    const hipError_t restoreError = hipSetDevice(previousDevice);
    if (restoreError != hipSuccess && status.err == novaSuccess) {
      status = mapError2Status(restoreError);
    }
  }
  return status;
}

/**
 * @brief Validates that @p stream belongs to @p device_id.
 *
 * @details
 * A null stream is always accepted (it denotes the device engine
 * stream).  A non-null stream is queried for its device and
 * rejected when it does not match.
 *
 * @param[in]  device_id Expected runtime device ordinal.
 * @param[in]  stream    Stream to validate, or null.
 *
 * @return @ref novaStatus_t; @c novaInvalidDeviceId when the stream
 *         belongs to a different device.
 */
static inline novaStatus_t validateStreamDevice(std::int32_t device_id,
                                                hipStream_t stream) {
  if (stream == nullptr) {
    return {.err = novaSuccess,
            .message = nova_get_error_msg(novaSuccess, nullptr)};
  }
  hipDevice_t streamDevice{};
  const hipError_t err = hipStreamGetDevice(stream, &streamDevice);
  if (err != hipSuccess) {
    return mapError2Status(err);
  }
  if (streamDevice != device_id) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/**
 * @brief Reports whether the backend device supports native memory pools.
 *
 * @details
 * Queries the device attribute for memory-pool support.  When the query
 * fails the error is mapped through @ref mapError2Status and reported;
 * the device is then treated as pool-less by the caller.
 *
 * @param[in]     device_id Runtime device ordinal.
 * @param[in,out] supported Receives @c true when pools are supported.
 *
 * @return @ref novaStatus_t; @c novaInvalidPointer when @p supported is
 *         null.
 */
static inline novaStatus_t supportMemoryPool(std::int32_t device_id,
                                             bool *supported) {
  if (supported == nullptr) {
    return {.err = novaInvalidPointer,
            .message = nova_get_error_msg(novaInvalidPointer, nullptr)};
  }
  int value = 0;
  const hipError_t err = hipDeviceGetAttribute(
      &value, hipDeviceAttributeMemoryPoolsSupported, device_id);
  if (err != hipSuccess) {
    return mapError2Status(err);
  }
  *supported = value != 0;
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

#endif
