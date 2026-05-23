/**
 * @file cuda_io.cpp
 * @brief Implementation of CUDA memory copy helpers.
 *
 * Implements host↔device and device↔device copy helpers using the CUDA
 * Runtime API (cudaMemcpy / cudaMemcpyAsync / cudaStream*), returning
 * CudaStatus_t for consistent error reporting.
 */

#include "cuda_io.hpp"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>

/**
 * @brief Map a CUDA runtime error to a CudaStatus_t code/message pair.
 *
 * @param err CUDA error value returned by the runtime API.
 * @return CudaStatus_t with code 0 on success, or an application-level code
 *         describing the failure class.
 */
static CudaStatus_t map_cuda_error(cudaError_t err) {
  CudaStatus_t cstatus = {};
  switch (err) {
  case cudaSuccess: {
    cstatus.msg = "ok";
    cstatus.code = 0;
    return cstatus;
  }
  case cudaErrorInvalidValue: {
    cstatus.msg = cudaGetErrorString(cudaErrorInvalidValue);
    cstatus.code = 1;
    return cstatus;
  }
  case cudaErrorInvalidMemcpyDirection: {
    cstatus.msg = cudaGetErrorString(cudaErrorInvalidMemcpyDirection);
    cstatus.code = 2;
    return cstatus;
  }
  default: {
    cstatus.msg = cudaGetErrorString(err);
    cstatus.code = -1;
    return cstatus;
  }
  }
}

/**
 * @brief Map stream creation/destruction errors to CudaStatus_t.
 *
 * @param err CUDA error value returned by cudaStreamCreate / cudaStreamDestroy.
 * @return CudaStatus_t with code 0 on success, or an application-level code.
 */
static CudaStatus_t map_cuda_stream_error(cudaError_t err) {
  CudaStatus_t cstatus = {};
  switch (err) {
  case cudaSuccess: {
    cstatus.msg = "ok";
    cstatus.code = 0;
    return cstatus;
  }
  case cudaErrorInvalidValue: {
    cstatus.msg = cudaGetErrorString(cudaErrorInvalidValue);
    cstatus.code = 1;
    return cstatus;
  }
  case cudaErrorExternalDevice: {
    cstatus.msg = cudaGetErrorString(cudaErrorExternalDevice);
    cstatus.code = 2;
    return cstatus;
  }
  default: {
    cstatus.msg = cudaGetErrorString(err);
    cstatus.code = -1;
    return cstatus;
  }
  }
}

/**
 * @brief Map stream synchronization errors to CudaStatus_t.
 *
 * @param err CUDA error value returned by cudaStreamSynchronize.
 * @return CudaStatus_t with code 0 on success, or an application-level code.
 */
static CudaStatus_t map_cuda_sync_error(cudaError_t err) {
  CudaStatus_t cstatus = {};
  switch (err) {
  case cudaSuccess: {
    cstatus.msg = "ok";
    cstatus.code = 0;
    return cstatus;
  }
  case cudaErrorInvalidResourceHandle: {
    cstatus.msg = cudaGetErrorString(cudaErrorInvalidResourceHandle);
    cstatus.code = 1;
    return cstatus;
  }
  default: {
    cstatus.msg = cudaGetErrorString(err);
    cstatus.code = -1;
    return cstatus;
  }
  }
}

/**
 * @brief Convert the device-agnostic copy kind to CUDA's runtime enum.
 */
static cudaMemcpyKind map_cuda_memcpy_kind(DeviceMemcpyKind kind) {
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
 * @brief Synchronous host-to-device copy wrapper.
 */
CudaStatus_t cuda_memcpy_host2device(std::size_t bytes, const void *src,
                                     void *dst) {
  cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  return map_cuda_error(err);
}

/**
 * @brief Host-to-device copy with async path for pinned host memory.
 */
static CudaStatus_t cuda_memcpy_host2device_async(std::size_t bytes,
                                                  cudaStream_t stream,
                                                  const void *src, void *dst,
                                                  bool pinned) {
  if (pinned) {
    cudaError_t err =
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return map_cuda_error(err);
  }
  cudaError_t err = cudaStreamDestroy(stream);
  stream = nullptr;
  cuda_memcpy_host2device(bytes, src, dst);
  return map_cuda_error(err);
}

/**
 * @brief Synchronous device-to-host copy wrapper.
 */
CudaStatus_t cuda_memcpy_device2host(std::size_t bytes, const void *src,
                                     void *dst) {
  cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
  return map_cuda_error(err);
}

/**
 * @brief Device-to-host copy with async path for pinned host memory.
 */
static CudaStatus_t cuda_memcpy_device2host_async(std::size_t bytes,
                                                  cudaStream_t stream,
                                                  const void *src, void *dst,
                                                  bool pinned) {

  if (pinned) {
    cudaError_t err =
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, nullptr);
    map_cuda_error(err);
  }
  cudaError_t err = cudaStreamDestroy(stream);
  stream = nullptr;
  cuda_memcpy_device2host(bytes, src, dst);
  return map_cuda_error(err);
}

/**
 * @brief Async device-to-device copy wrapper.
 */
static CudaStatus_t cuda_memcpy_device2device(std::size_t bytes,
                                              cudaStream_t stream,
                                              const void *src, void *dst) {
  cudaError_t err =
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
  return map_cuda_error(err);
}

/**
 * @brief Public host-to-device copy helper with optional async transfer.
 */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyHostToDevice, src,
                     dst, pinned);
}

/**
 * @brief Public device-to-host copy helper with optional async transfer.
 */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToHost, src,
                     dst, pinned);
}

/**
 * @brief Public device-to-device copy helper.
 */
CudaStatus_t cuda_memcpy_device2device(std::size_t bytes, const void *src,
                                       void *dst) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToDevice, src,
                     dst, false);
}

/**
 * @brief High-level copy helper that creates and synchronizes a stream.
 */
CudaStatus_t cuda_memcpy(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst, bool is_pinned) {

  CudaStatus_t status = {};
  cudaStream_t stream = nullptr;
  cudaError_t stream_err = cudaStreamCreate(&stream);
  if (stream_err != cudaSuccess) {
    return map_cuda_stream_error(stream_err);
  }
  switch (map_cuda_memcpy_kind(kind)) {
  case cudaMemcpyHostToDevice:
    status =
        cuda_memcpy_host2device_async(bytes, stream, src, dst, is_pinned);
    break;
  case cudaMemcpyDeviceToHost:
    status =
        cuda_memcpy_device2host_async(bytes, stream, src, dst, is_pinned);
    break;
  case cudaMemcpyDeviceToDevice:
    status = cuda_memcpy_device2device(bytes, stream, src, dst);
    break;
  default:
    break;
  }

  if (stream != nullptr) {
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
      return map_cuda_sync_error(sync_err);
    }
  }
  return status;
}

#else

/**
 * @brief Fallback host-to-device copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_host2device(std::size_t, const void *, void *) {
  return CudaStatus_t{
      .code = -1,
      .msg = "CUDA runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback host-to-device async copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t, const void *, void *,
                                           bool) {
  return CudaStatus_t{
      .code = -1,
      .msg = "CUDA runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback device-to-host copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_device2host(std::size_t, const void *, void *) {
  return CudaStatus_t{
      .code = -1,
      .msg = "CUDA runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback device-to-host async copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t, const void *, void *,
                                           bool) {
  return CudaStatus_t{
      .code = -1,
      .msg = "CUDA runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback device-to-device copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_device2device(std::size_t, const void *, void *) {
  return CudaStatus_t{
      .code = -1,
      .msg = "CUDA runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback generic copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy(std::size_t, DeviceMemcpyKind, const void *, void *,
                         bool) {
  return CudaStatus_t{
      .code = -1,
      .msg = "CUDA runtime headers not available at build/lint time"};
}

#endif
