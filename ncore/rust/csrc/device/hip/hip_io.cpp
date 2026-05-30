/**
 * @file hip_io.cpp
 * @brief Implementation of HIP memory copy helpers.
 *
 * Mirrors the CUDA helper implementation in cuda_io.cpp, but using the HIP
 * Runtime API (hipMemcpy / hipMemcpyAsync / hipStream*).
 *
 * All transfers that involve device memory go through hip_memcpy, which
 * owns the stream lifetime: it creates the stream, dispatches to one of the
 * internal static helpers, synchronises, and destroys the stream before
 * returning — even on error paths. The static helpers never touch stream
 * lifetime; they only issue the async API call and return its status.
 *
 * When the host buffer is not pinned, host↔device transfers fall back to the
 * synchronous hipMemcpy variants (no stream required); hip_memcpy still
 * synchronises and destroys the stream it created before returning.
 */

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "hip_io.hpp"
#include <hip/hip_runtime_api.h>

/**
 * @brief Map a HIP runtime error to a HipStatus_t code/message pair.
 *
 * @param err HIP error value returned by the runtime API.
 * @return HipStatus_t with code 0 on success, or an application-level code
 *         describing the failure class.
 */
static HipStatus_t map_hip_error(hipError_t err) {
  HipStatus_t status = {};
  switch (err) {
  case hipSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case hipErrorInvalidValue:
    status.code = 1;
    status.msg = hipGetErrorString(err);
    return status;
  case hipErrorInvalidMemcpyDirection:
    status.code = 2;
    status.msg = hipGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = hipGetErrorString(err);
    return status;
  }
}

/**
 * @brief Map a HIP stream-creation or stream-destruction error to a
 *        HipStatus_t code/message pair.
 *
 * @param err HIP error value returned by hipStreamCreate / hipStreamDestroy.
 * @return HipStatus_t with code 0 on success, or an application-level code.
 */
static HipStatus_t map_hip_stream_error(hipError_t err) {
  HipStatus_t status = {};
  switch (err) {
  case hipSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case hipErrorInvalidValue:
    status.code = 1;
    status.msg = hipGetErrorString(err);
    return status;
  case hipErrorInvalidDevice:
    status.code = 2;
    status.msg = hipGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = hipGetErrorString(err);
    return status;
  }
}

/**
 * @brief Map a HIP stream-synchronisation error to a HipStatus_t
 *        code/message pair.
 *
 * @param err HIP error value returned by hipStreamSynchronize.
 * @return HipStatus_t with code 0 on success, or an application-level code.
 */
static HipStatus_t map_hip_sync_error(hipError_t err) {
  HipStatus_t status = {};
  switch (err) {
  case hipSuccess:
    status.code = 0;
    status.msg = "ok";
    return status;
  case hipErrorInvalidResourceHandle:
    status.code = 1;
    status.msg = hipGetErrorString(err);
    return status;
  default:
    status.code = -1;
    status.msg = hipGetErrorString(err);
    return status;
  }
}

/**
 * @brief Convert the device-agnostic copy kind to HIP's runtime enum.
 *
 * @param kind Device-agnostic copy direction.
 * @return The corresponding hipMemcpyKind value.
 */
static hipMemcpyKind map_hip_memcpy_kind(DeviceMemcpyKind kind) {
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
 * @brief Internal: issue a host-to-device copy on @p stream (pinned) or
 *        fall back to the synchronous variant (non-pinned).
 *
 * Does NOT own or modify stream lifetime. The caller (hip_memcpy) is
 * responsible for synchronising and destroying the stream.
 *
 * @param bytes  Number of bytes to copy.
 * @param stream Stream to use for the async path.
 * @param src    Source pointer in host memory.
 * @param dst    Destination pointer in device memory.
 * @param pinned Whether @p src is page-locked.
 * @return HipStatus_t describing success or failure of the copy call only.
 */
static HipStatus_t hip_memcpy_h2d(std::size_t bytes, hipStream_t stream,
                                  const void *src, void *dst, bool pinned) {
  if (pinned) {
    return map_hip_error(
        hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream));
  }
  return map_hip_error(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

/**
 * @brief Internal: issue a device-to-host copy on @p stream (pinned) or
 *        fall back to the synchronous variant (non-pinned).
 *
 * Does NOT own or modify stream lifetime. The caller (hip_memcpy) is
 * responsible for synchronising and destroying the stream.
 *
 * @param bytes  Number of bytes to copy.
 * @param stream Stream to use for the async path.
 * @param src    Source pointer in device memory.
 * @param dst    Destination pointer in host memory.
 * @param pinned Whether @p dst is page-locked.
 * @return HipStatus_t describing success or failure of the copy call only.
 */
static HipStatus_t hip_memcpy_d2h(std::size_t bytes, hipStream_t stream,
                                  const void *src, void *dst, bool pinned) {
  if (pinned) {
    return map_hip_error(
        hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream));
  }
  return map_hip_error(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

/**
 * @brief Internal: issue an async device-to-device copy on @p stream.
 *
 * Does NOT own or modify stream lifetime. The caller (hip_memcpy) is
 * responsible for synchronising and destroying the stream.
 *
 * @param bytes  Number of bytes to copy.
 * @param stream Stream on which to issue the copy.
 * @param src    Source pointer in device memory.
 * @param dst    Destination pointer in device memory.
 * @return HipStatus_t describing success or failure of the copy call only.
 */
static HipStatus_t hip_memcpy_d2d(std::size_t bytes, hipStream_t stream,
                                  const void *src, void *dst) {
  return map_hip_error(
      hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream));
}

/**
 * @brief Synchronous host-to-device copy.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in host memory.
 * @param dst   Destination pointer in device memory.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_host2device(std::size_t bytes, const void *src,
                                   void *dst) {
  return map_hip_error(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

/**
 * @brief Host-to-device copy, async when the host buffer is pinned.
 *
 * Delegates to hip_memcpy with the HostToDevice direction.
 *
 * @param bytes  Number of bytes to copy.
 * @param src    Source pointer in host memory.
 * @param dst    Destination pointer in device memory.
 * @param pinned Whether @p src is page-locked.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_host2device_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyHostToDevice, src, dst,
                    pinned);
}

/**
 * @brief Synchronous device-to-host copy.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in host memory.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_device2host(std::size_t bytes, const void *src,
                                   void *dst) {
  return map_hip_error(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

/**
 * @brief Device-to-host copy, async when the host buffer is pinned.
 *
 * Delegates to hip_memcpy with the DeviceToHost direction.
 *
 * @param bytes  Number of bytes to copy.
 * @param src    Source pointer in device memory.
 * @param dst    Destination pointer in host memory.
 * @param pinned Whether @p dst is page-locked.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_device2host_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToHost, src, dst,
                    pinned);
}

/**
 * @brief Async device-to-device copy.
 *
 * Delegates to hip_memcpy with the DeviceToDevice direction.
 *
 * @param bytes Number of bytes to copy.
 * @param src   Source pointer in device memory.
 * @param dst   Destination pointer in device memory.
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy_device2device(std::size_t bytes, const void *src,
                                     void *dst) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToDevice, src,
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
 * @return HipStatus_t describing success or failure.
 */
HipStatus_t hip_memcpy(std::size_t bytes, DeviceMemcpyKind kind,
                       const void *src, void *dst, bool is_pinned) {
  hipStream_t stream = nullptr;
  const hipError_t stream_err = hipStreamCreate(&stream);
  if (stream_err != hipSuccess) {
    return map_hip_stream_error(stream_err);
  }

  HipStatus_t status = {};
  switch (map_hip_memcpy_kind(kind)) {
  case hipMemcpyHostToDevice:
    status = hip_memcpy_h2d(bytes, stream, src, dst, is_pinned);
    break;
  case hipMemcpyDeviceToHost:
    status = hip_memcpy_d2h(bytes, stream, src, dst, is_pinned);
    break;
  case hipMemcpyDeviceToDevice:
    status = hip_memcpy_d2d(bytes, stream, src, dst);
    break;
  default:
    break;
  }

  // Synchronise unconditionally
  const hipError_t sync_err = hipStreamSynchronize(stream);
  if (sync_err != hipSuccess && status.code == 0) {
    status = map_hip_sync_error(sync_err);
  }

  // Destroy unconditionally
  const hipError_t destroy_err = hipStreamDestroy(stream);
  if (destroy_err != hipSuccess && status.code == 0) {
    status = map_hip_stream_error(destroy_err);
  }

  return status;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/**
 * @brief Fallback host-to-device copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_host2device(std::size_t, const void *, void *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/**
 * @brief Fallback host-to-device async copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_host2device_async(std::size_t, const void *, void *,
                                         bool) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/**
 * @brief Fallback device-to-host copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_device2host(std::size_t, const void *, void *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/**
 * @brief Fallback device-to-host async copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_device2host_async(std::size_t, const void *, void *,
                                         bool) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/**
 * @brief Fallback device-to-device copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_device2device(std::size_t, const void *, void *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/**
 * @brief Fallback generic copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy(std::size_t, DeviceMemcpyKind, const void *, void *,
                       bool) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

#endif
#endif /* NOVA_HAS_HIP */
