/**
 * @file cuda_io.cpp
 * @brief Implementation of CUDA memory copy helpers.
 *
 * Implements host↔device and device↔device copy helpers using the CUDA
 * Runtime API (cudaMemcpy / cudaMemcpyAsync / cudaStream*), returning
 * CudaStatus_t for consistent error reporting.
 *
 * All transfers that involve device memory go through cuda_memcpy, which
 * owns the stream lifetime: it creates the stream, dispatches to one of the
 * internal static helpers, synchronises, and destroys the stream before
 * returning — even on error paths. The static helpers never touch stream
 * lifetime; they only issue the async API call and return its status.
 *
 * When the host buffer is not pinned, host↔device transfers fall back to the
 * synchronous cudaMemcpy variants (no stream required); cuda_memcpy still
 * synchronises and destroys the stream it created before returning.
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
 * @brief Map a CUDA stream-creation or stream-destruction error to a
 *        CudaStatus_t code/message pair.
 *
 * @param err CUDA error value returned by cudaStreamCreate /
 *            cudaStreamDestroy.
 * @return CudaStatus_t with code 0 on success, or an application-level code.
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
 * @brief Map a CUDA stream-synchronisation error to a CudaStatus_t
 *        code/message pair.
 *
 * @param err CUDA error value returned by cudaStreamSynchronize.
 * @return CudaStatus_t with code 0 on success, or an application-level code.
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
 * @brief Convert the device-agnostic copy kind to CUDA's runtime enum.
 *
 * @param kind Device-agnostic copy direction.
 * @return The corresponding cudaMemcpyKind value.
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
 * @brief Internal: issue a host-to-device copy on @p stream (pinned) or
 *        fall back to the synchronous variant (non-pinned).
 *
 * Does NOT own or modify stream lifetime. The caller (cuda_memcpy) is
 * responsible for synchronising and destroying the stream.
 *
 * @param bytes  Number of bytes to copy.
 * @param stream Stream to use for the async path.
 * @param src    Source pointer in host memory.
 * @param dst    Destination pointer in device memory.
 * @param pinned Whether @p src is page-locked.
 * @return CudaStatus_t describing success or failure of the copy call only.
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
 * @brief Internal: issue a device-to-host copy on @p stream (pinned) or
 *        fall back to the synchronous variant (non-pinned).
 *
 * Does NOT own or modify stream lifetime. The caller (cuda_memcpy) is
 * responsible for synchronising and destroying the stream.
 *
 * @param bytes  Number of bytes to copy.
 * @param stream Stream to use for the async path.
 * @param src    Source pointer in device memory.
 * @param dst    Destination pointer in host memory.
 * @param pinned Whether @p dst is page-locked.
 * @return CudaStatus_t describing success or failure of the copy call only.
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
 * @brief Internal: issue an async device-to-device copy on @p stream.
 *
 * Does NOT own or modify stream lifetime. The caller (cuda_memcpy) is
 * responsible for synchronising and destroying the stream.
 *
 * @param bytes  Number of bytes to copy.
 * @param stream Stream on which to issue the copy.
 * @param src    Source pointer in device memory.
 * @param dst    Destination pointer in device memory.
 * @return CudaStatus_t describing success or failure of the copy call only.
 */
static CudaStatus_t cuda_memcpy_d2d(std::size_t bytes, cudaStream_t stream,
                                    const void *src, void *dst) {
  return map_cuda_error(
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
}

/**
 * @brief Synchronous host-to-device copy.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in host memory.
 * @param dst   Destination pointer in device memory.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_host2device(std::size_t bytes, const void *src,
                                     void *dst) {
  return map_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

/**
 * @brief Host-to-device copy, async when the host buffer is pinned.
 *
 * Delegates to cuda_memcpy with the HostToDevice direction.
 *
 * @param bytes  Number of bytes to copy.
 * @param src    Source pointer in host memory.
 * @param dst    Destination pointer in device memory.
 * @param pinned Whether @p src is page-locked.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyHostToDevice, src,
                     dst, pinned);
}

/**
 * @brief Synchronous device-to-host copy.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in host memory.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_device2host(std::size_t bytes, const void *src,
                                     void *dst) {
  return map_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

/**
 * @brief Device-to-host copy, async when the host buffer is pinned.
 *
 * Delegates to cuda_memcpy with the DeviceToHost direction.
 *
 * @param bytes  Number of bytes to copy.
 * @param src    Source pointer in device memory.
 * @param dst    Destination pointer in host memory.
 * @param pinned Whether @p dst is page-locked.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t bytes, const void *src,
                                           void *dst, bool pinned) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToHost, src,
                     dst, pinned);
}

/**
 * @brief Async device-to-device copy.
 *
 * Delegates to cuda_memcpy with the DeviceToDevice direction.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in device memory.
 * @return CudaStatus_t describing success or failure.
 */
CudaStatus_t cuda_memcpy_device2device(std::size_t bytes, const void *src,
                                       void *dst) {
  return cuda_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToDevice, src,
                     dst, false);
}

/**
 * @brief Unified copy entry point — owns the full stream lifetime.
 *
 * Creates a stream, dispatches to the appropriate internal helper based on
 * @p kind, then synchronises and destroys the stream unconditionally before
 * returning.  A sync or destroy error takes priority over a copy error only
 * when the copy itself succeeded.
 *
 * @param bytes     Number of bytes to copy.
 * @param kind      Copy direction.
 * @param src       Source pointer.
 * @param dst       Destination pointer.
 * @param is_pinned Whether the host-side buffer is page-locked (only
 *                  meaningful for H2D and D2H transfers).
 * @return CudaStatus_t describing success or failure.
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

  // Synchronise unconditionally
  const cudaError_t sync_err = cudaStreamSynchronize(stream);
  if (sync_err != cudaSuccess && status.code == 0) {
    status = map_cuda_sync_error(sync_err);
  }

  // Destroy unconditionally
  const cudaError_t destroy_err = cudaStreamDestroy(stream);
  if (destroy_err != cudaSuccess && status.code == 0) {
    status = map_cuda_stream_error(destroy_err);
  }

  return status;
}

#else // !__has_include(<cuda_runtime_api.h>)

/**
 * @brief Fallback host-to-device copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_host2device(std::size_t, const void *, void *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/**
 * @brief Fallback host-to-device async copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_host2device_async(std::size_t, const void *, void *,
                                           bool) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/**
 * @brief Fallback device-to-host copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_device2host(std::size_t, const void *, void *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/**
 * @brief Fallback device-to-host async copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_device2host_async(std::size_t, const void *, void *,
                                           bool) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/**
 * @brief Fallback device-to-device copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy_device2device(std::size_t, const void *, void *) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

/**
 * @brief Fallback generic copy when CUDA headers are unavailable.
 */
CudaStatus_t cuda_memcpy(std::size_t, DeviceMemcpyKind, const void *, void *,
                         bool) {
  return CudaStatus_t{.code = -1,
                      .msg = "CUDA runtime headers not available at"
                             " build/lint time"};
}

#endif
