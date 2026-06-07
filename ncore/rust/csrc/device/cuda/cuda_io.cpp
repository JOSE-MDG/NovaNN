/**
 * @file cuda_io.cpp
 * @brief CUDA data transfer implementation.
 *
 * @details
 * Implements synchronous and asynchronous memcpy operations for
 * host-to-device, device-to-host, and device-to-device copies.
 * The master dispatcher @ref cuda_memcpy owns the stream lifecycle
 * (create, copy, synchronise, destroy).
 *
 * The file is conditionally compiled behind `NOVA_HAS_CUDA` and
 * `__has_include(<cuda_runtime_api.h>)`.  When CUDA headers are
 * unavailable, stub functions returning an error status are
 * provided.
 *
 * @see cuda_io.hpp        Function declarations.
 * @see cuda_allocator.cpp CUDA memory allocation implementation.
 * @see ffi.cpp            Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)
#include "cuda_io.hpp"
#include <cuda_runtime_api.h>

/**
 * @brief Map a CUDA error to a @ref CudaStatus_t.
 *
 * @param[in] err  The CUDA error to map.
 *
 * @return Status with code `0` on success, `1` for invalid value,
 *         `2` for invalid memcpy direction, `-1` otherwise.
 */
static CudaStatus_t map_cuda_error(cudaError_t err) {
  CudaStatus_t status = {};
  switch (err) {
  case cudaSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case cudaErrorInvalidValue:
    status.code = 1;
    status.msg = cudaGetErrorString(err);
    return status;
  case cudaErrorInvalidMemcpyDirection:
    status.code = 2;
    status.msg = cudaGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = cudaGetErrorString(err);
    return status;
  }
}

/**
 * @brief Map a CUDA stream error to a @ref CudaStatus_t.
 *
 * @param[in] err  The CUDA error from a stream operation.
 *
 * @return Status with code `0` on success, `1` for invalid value,
 *         `2` for external device error, `-1` otherwise.
 */
static CudaStatus_t map_cuda_stream_error(cudaError_t err) {
  CudaStatus_t status = {};
  switch (err) {
  case cudaSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case cudaErrorInvalidValue:
    status.code = 1;
    status.msg = cudaGetErrorString(err);
    return status;
  case cudaErrorExternalDevice:
    status.code = 2;
    status.msg = cudaGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = cudaGetErrorString(err);
    return status;
  }
}

/**
 * @brief Map a CUDA synchronisation error to a @ref CudaStatus_t.
 *
 * @param[in] err  The CUDA error from stream synchronisation.
 *
 * @return Status with code `0` on success, `1` for invalid
 *         resource handle, `-1` otherwise.
 */
static CudaStatus_t map_cuda_sync_error(cudaError_t err) {
  CudaStatus_t status = {};
  switch (err) {
  case cudaSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case cudaErrorInvalidResourceHandle:
    status.code = 1;
    status.msg = cudaGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = cudaGetErrorString(err);
    return status;
  }
}

/**
 * @brief Convert @ref DeviceMemcpyKind to `cudaMemcpyKind`.
 *
 * @param[in] kind  The device-agnostic copy direction.
 *
 * @return The corresponding CUDA memcpy kind.
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
 * @brief Perform a host-to-device copy on @p stream.
 *
 * @details
 * Uses `cudaMemcpyAsync` when @p pinned is `true`,
 * `cudaMemcpy` otherwise.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  stream The CUDA stream.
 * @param[in]  src    Source host pointer.
 * @param[out] dst    Destination device pointer.
 * @param[in]  pinned Whether @p src is page-locked.
 *
 * @return Status of the copy operation.
 */
static CudaStatus_t cuda_memcpy_h2d(std::size_t bytes, cudaStream_t stream,
                                    const void *src, void *dst, bool pinned) {
  if (pinned) {
    return map_cuda_error(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
  }
  return map_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

/**
 * @brief Perform a device-to-host copy on @p stream.
 *
 * @details
 * Uses `cudaMemcpyAsync` when @p pinned is `true`,
 * `cudaMemcpy` otherwise.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  stream The CUDA stream.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination host pointer.
 * @param[in]  pinned Whether @p dst is page-locked.
 *
 * @return Status of the copy operation.
 */
static CudaStatus_t cuda_memcpy_d2h(std::size_t bytes, cudaStream_t stream,
                                    const void *src, void *dst, bool pinned) {
  if (pinned) {
    return map_cuda_error(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
  }
  return map_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

/**
 * @brief Perform a device-to-device copy on @p stream.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  stream The CUDA stream.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination device pointer.
 *
 * @return Status of the copy operation.
 */
static CudaStatus_t cuda_memcpy_d2d(std::size_t bytes, cudaStream_t stream,
                                    const void *src, void *dst) {
  return map_cuda_error(
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
}

/**
 * @brief Synchronous host-to-device copy.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source host pointer.
 * @param[out] dst    Destination device pointer.
 *
 * @return @ref CUDA_OK on success, or an error status.
 */
CudaStatus_t cuda_memcpy_host2device(std::size_t bytes, const void *src,
                                     void *dst) {
  return map_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

/**
 * @brief Asynchronous host-to-device copy.
 *
 * @details
 * Delegates to @ref cuda_memcpy with
 * @ref DeviceMemcpyKind::deviceMemcpyHostToDevice.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source host pointer.
 * @param[out] dst    Destination device pointer.
 * @param[in]  pinned Whether @p src is page-locked.
 *
 * @return @ref CUDA_OK on success, or an error status.
 */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyHostToDevice, src,
                     dst, pinned);
}

/**
 * @brief Synchronous device-to-host copy.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination host pointer.
 *
 * @return @ref CUDA_OK on success, or an error status.
 */
CudaStatus_t cuda_memcpy_device2host(std::size_t bytes, const void *src,
                                     void *dst) {
  return map_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

/**
 * @brief Asynchronous device-to-host copy.
 *
 * @details
 * Delegates to @ref cuda_memcpy with
 * @ref DeviceMemcpyKind::deviceMemcpyDeviceToHost.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination host pointer.
 * @param[in]  pinned Whether @p dst is page-locked.
 *
 * @return @ref CUDA_OK on success, or an error status.
 */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToHost, src,
                     dst, pinned);
}

/**
 * @brief Device-to-device copy.
 *
 * @details
 * Delegates to @ref cuda_memcpy with
 * @ref DeviceMemcpyKind::deviceMemcpyDeviceToDevice.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination device pointer.
 *
 * @return @ref CUDA_OK on success, or an error status.
 */
CudaStatus_t cuda_memcpy_device2device(std::size_t bytes, const void *src,
                                       void *dst) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToDevice, src,
                     dst, false);
}

/**
 * @brief Master memcpy dispatcher for CUDA.
 *
 * @details
 * Creates a temporary stream, performs the copy in the direction
 * specified by @p kind, synchronises, and destroys the stream.
 *
 * @param[in]  bytes     Number of bytes.
 * @param[in]  kind      Copy direction.
 * @param[in]  src       Source pointer.
 * @param[out] dst       Destination pointer.
 * @param[in]  is_pinned Whether the host-side pointer is
 *                      page-locked.
 *
 * @return @ref CUDA_OK on success, or an error status.
 */
CudaStatus_t cuda_memcpy(std::size_t bytes, DeviceMemcpyKind kind,
                         const void *src, void *dst, bool is_pinned) {
  cudaStream_t stream = nullptr;
  const cudaError_t stream_err = cudaStreamCreate(&stream);
  if (stream_err != cudaSuccess) {
    return map_cuda_stream_error(stream_err);
  }

  CudaStatus_t status = {};
  switch (map_cuda_memcpy_kind(kind)) {
  case cudaMemcpyHostToDevice:
    status = cuda_memcpy_h2d(bytes, stream, src, dst, is_pinned);
    break;
  case cudaMemcpyDeviceToHost:
    status = cuda_memcpy_d2h(bytes, stream, src, dst, is_pinned);
    break;
  case cudaMemcpyDeviceToDevice:
    status = cuda_memcpy_d2d(bytes, stream, src, dst);
    break;
  default:
    break;
  }

  const cudaError_t sync_err = cudaStreamSynchronize(stream);
  if (sync_err != cudaSuccess && status.code == 0) {
    status = map_cuda_sync_error(sync_err);
  }

  const cudaError_t destroy_err = cudaStreamDestroy(stream);
  if (destroy_err != cudaSuccess && status.code == 0) {
    status = map_cuda_stream_error(destroy_err);
  }

  return status;
}

#else // !__has_include(<cuda_runtime_api.h>)

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_memcpy_host2device(std::size_t, const void *, void *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t, const void *, void *,
                                           bool) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_memcpy_device2host(std::size_t, const void *, void *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t, const void *, void *,
                                           bool) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_memcpy_device2device(std::size_t, const void *, void *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/** @brief Stub: CUDA runtime headers not available. */
CudaStatus_t cuda_memcpy(std::size_t, DeviceMemcpyKind, const void *, void *,
                         bool) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

#endif
#endif /* NOVA_HAS_CUDA */
