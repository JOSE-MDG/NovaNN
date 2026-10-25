/**
 * @file common.hh
 * @brief Shared CUDA and HIP runtime helpers for the memory backends.
 *
 * Private to the native backends. The two scopes expose the same
 * error-mapping and stream helpers, and exactly one is active per
 * build. The HIP stream is created lazily, so callers must resolve it
 * before enqueuing null-stream work: rocclr cancels pending null-stream
 * kernels at @c hipStreamCreate.
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

#include "../memory/csrc/ffi.hpp"
#include "cuda/DetectCudaDevice.hpp"
#include <ncore/core/status.h>

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
  novaStatus_t st = {};
  switch (err) {
  case cudaSuccess:
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  case cudaErrorInvalidValue:
    st.err = novaInvalidValue;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  case cudaErrorExternalDevice:
    st.err = novaExternalDeviceError;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  case cudaErrorInvalidMemcpyDirection:
    st.err = novaInvalidTransfDirection;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  case cudaErrorInvalidResourceHandle:
    st.err = novaInvalidResourceHandle;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  default:
    st.err = novaNotImplemented;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
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
 * @brief Get or create the reusable CUDA stream.
 *
 * @details
 * Returns a singleton CUDA stream created exactly once via
 * @c std::call_once and reused for all subsequent default-path
 * @ref cudaTransfer operations.  The stream is created with default
 * flags (non-blocking, no priority override).
 *
 * The stream is never explicitly destroyed.  The CUDA runtime
 * reclaims all resources on process exit.  This avoids the
 * overhead of create/destroy per transfer and eliminates the
 * risk of use-after-free in concurrent scenarios.
 *
 * @param[out] status  Receives an error status if stream creation
 *                     fails.  Unchanged on success.
 *
 * @return The singleton @c cudaStream_t, or null on failure.
 *
 * @warning A failed creation is sticky: @c std::call_once never
 *          retries, so every later call also fails until process
 *          restart.  Stream creation failure implies a broken device.
 */
static inline cudaStream_t getStream(novaStatus_t *status) {
  static cudaStream_t stream = nullptr;
  static std::once_flag flag;
  static novaStatus_t initStatus = {};
  std::call_once(flag, [] {
    const cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
      initStatus = mapError2Status(err);
    }
  });
  if (stream == nullptr) {
    *status = initStatus;
  }
  return stream;
}

/**
 * @brief Query whether the active CUDA device supports memory pools.
 *
 * @details
 * Queries the @c cudaDevAttrMemoryPoolsSupported attribute through
 * @c getCudaDeviceId().  Safe because CUDA device detection runs
 * before any device memory is allocated and the detection result is
 * saved under lock, so @c getCudaDeviceId() always returns a valid
 * id here.
 *
 * @return @c true if memory pools are supported, @c false otherwise.
 */
static inline bool supportMemoryPool() {
  static const bool supported = [] {
    int value = 0;
    const cudaError_t err = cudaDeviceGetAttribute(
        &value, cudaDevAttrMemoryPoolsSupported, getCudaDeviceId());
    return err == cudaSuccess && value != 0;
  }();
  return supported;
}

/**
 * @brief Create a CUDA stream.
 *
 * @param[out] stream  Receives the new stream handle.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
static inline bool streamCreate(cudaStream_t *stream, novaStatus_t *status) {
  const cudaError_t err = cudaStreamCreate(stream);
  if (err != cudaSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

/**
 * @brief Block until all work on @p stream has completed.
 *
 * @param[in]  stream  The stream to synchronize.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
static inline bool streamSync(cudaStream_t stream, novaStatus_t *status) {
  const cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

/**
 * @brief Destroy a CUDA stream.
 *
 * @param[in]  stream  The stream to destroy.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
static inline bool streamDestroy(cudaStream_t stream, novaStatus_t *status) {
  const cudaError_t err = cudaStreamDestroy(stream);
  if (err != cudaSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

#elif defined(NOVA_HAS_HIP) && __has_include(<hip/hip_runtime_api.h>)

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

#include <mutex>

#include "../memory/csrc/ffi.hpp"
#include "hip/DetectHipDevice.hpp"
#include <ncore/core/status.h>

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
  novaStatus_t st = {};
  switch (err) {
  case hipSuccess:
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  case hipErrorInvalidValue:
    st.err = novaInvalidValue;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  case hipErrorInvalidMemcpyDirection:
    st.err = novaInvalidTransfDirection;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  case hipErrorInvalidResourceHandle:
    st.err = novaInvalidResourceHandle;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  default:
    st.err = novaNotImplemented;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
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
 * @brief Get or create the reusable HIP stream.
 *
 * @details
 * Returns a singleton HIP stream created exactly once via
 * @c std::call_once and reused for all subsequent default-path
 * transfers.  The stream is created with default flags (non-blocking,
 * no priority override).
 *
 * The stream is never explicitly destroyed.  The HIP runtime
 * reclaims all resources on process exit.  This avoids the
 * overhead of create/destroy per transfer and eliminates the
 * risk of use-after-free in concurrent scenarios.
 *
 * @param[out] status  Receives an error status if stream creation
 *                     fails.  Unchanged on success.
 *
 * @return The singleton @c hipStream_t, or null on failure.
 *
 * @warning A failed creation is sticky: @c std::call_once never
 *          retries, so every later call also fails until process
 *          restart.  rocclr silently cancels pending null-stream
 *          work at @c hipStreamCreate, so the first call must
 *          happen before any null-stream operation is enqueued.
 */
static inline hipStream_t getStream(novaStatus_t *status) {
  static hipStream_t stream = nullptr;
  static std::once_flag flag;
  static novaStatus_t initStatus = {};
  std::call_once(flag, [] {
    const hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
      initStatus = mapError2Status(err);
    }
  });
  if (stream == nullptr) {
    *status = initStatus;
  }
  return stream;
}

/**
 * @brief Query whether the active HIP device supports memory pools.
 *
 * @details
 * Queries the @c hipDeviceAttributeMemoryPoolsSupported attribute
 * through @c getHipDeviceId().  Safe because HIP device detection
 * runs before any device memory is allocated and the detection
 * result is saved under lock, so @c getHipDeviceId() always returns
 * a valid id here.
 *
 * @return @c true if memory pools are supported, @c false otherwise.
 */
static inline bool supportMemoryPool() {
  static const bool supported = [] {
    int value = 0;
    const hipError_t err = hipDeviceGetAttribute(
        &value, hipDeviceAttributeMemoryPoolsSupported, getHipDeviceId());
    return err == hipSuccess && value != 0;
  }();
  return supported;
}

/**
 * @brief Create a HIP stream.
 *
 * @param[out] stream  Receives the new stream handle.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
static inline bool streamCreate(hipStream_t *stream, novaStatus_t *status) {
  const hipError_t err = hipStreamCreate(stream);
  if (err != hipSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

/**
 * @brief Block until all work on @p stream has completed.
 *
 * @param[in]  stream  The stream to synchronize.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
static inline bool streamSync(hipStream_t stream, novaStatus_t *status) {
  const hipError_t err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

/**
 * @brief Destroy a HIP stream.
 *
 * @param[in]  stream  The stream to destroy.
 * @param[out] status  Receives the error status on failure.
 *
 * @return @c true on success, @c false on failure.
 */
static inline bool streamDestroy(hipStream_t stream, novaStatus_t *status) {
  const hipError_t err = hipStreamDestroy(stream);
  if (err != hipSuccess) {
    status->err = mapError(err);
    status->message = nova_get_error_msg(status->err, nullptr);
    return false;
  }
  return true;
}

#endif
