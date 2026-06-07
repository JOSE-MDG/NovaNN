/**
 * @file hip_io.cpp
 * @brief HIP data transfer implementation.
 *
 * @details
 * Implements synchronous and asynchronous memcpy operations for
 * host-to-device, device-to-host, and device-to-device copies.
 * The master dispatcher @ref hip_memcpy owns the stream lifecycle
 * (create, copy, synchronise, destroy).
 *
 * The file is conditionally compiled behind `NOVA_HAS_HIP` and
 * `__has_include(<hip/hip_runtime_api.h>)`.  When HIP headers are
 * unavailable, stub functions returning an error status are
 * provided.
 *
 * A `__HIP_PLATFORM_AMD__` macro is defined when neither AMD nor
 * NVIDIA platform macros are set, ensuring clangd and clang-tidy
 * can parse the HIP headers correctly.
 *
 * @see hip_io.hpp        Function declarations.
 * @see hip_allocator.cpp HIP memory allocation implementation.
 * @see ffi.cpp           Dispatch layer that calls into this file.
 */

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include "hip_io.hpp"
#include <hip/hip_runtime_api.h>

/**
 * @brief Map a HIP error to a @ref HipStatus_t.
 *
 * @param[in] err  The HIP error to map.
 *
 * @return Status with code `0` on success, `1` for invalid value,
 *         `2` for invalid memcpy direction, `-1` otherwise.
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
 * @brief Map a HIP stream error to a @ref HipStatus_t.
 *
 * @param[in] err  The HIP error from a stream operation.
 *
 * @return Status with code `0` on success, `1` for invalid value,
 *         `2` for invalid device, `-1` otherwise.
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
 * @brief Map a HIP synchronisation error to a @ref HipStatus_t.
 *
 * @param[in] err  The HIP error from stream synchronisation.
 *
 * @return Status with code `0` on success, `1` for invalid
 *         resource handle, `-1` otherwise.
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
 * @brief Convert @ref DeviceMemcpyKind to `hipMemcpyKind`.
 *
 * @param[in] kind  The device-agnostic copy direction.
 *
 * @return The corresponding HIP memcpy kind.
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
 * @brief Perform a host-to-device copy on @p stream.
 *
 * @details
 * Uses `hipMemcpyAsync` when @p pinned is `true`,
 * `hipMemcpy` otherwise.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  stream The HIP stream.
 * @param[in]  src    Source host pointer.
 * @param[out] dst    Destination device pointer.
 * @param[in]  pinned Whether @p src is page-locked.
 *
 * @return Status of the copy operation.
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
 * @brief Perform a device-to-host copy on @p stream.
 *
 * @details
 * Uses `hipMemcpyAsync` when @p pinned is `true`,
 * `hipMemcpy` otherwise.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  stream The HIP stream.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination host pointer.
 * @param[in]  pinned Whether @p dst is page-locked.
 *
 * @return Status of the copy operation.
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
 * @brief Perform a device-to-device copy on @p stream.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  stream The HIP stream.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination device pointer.
 *
 * @return Status of the copy operation.
 */
static HipStatus_t hip_memcpy_d2d(std::size_t bytes, hipStream_t stream,
                                  const void *src, void *dst) {
  return map_hip_error(
      hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream));
}

/**
 * @brief Synchronous host-to-device copy.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source host pointer.
 * @param[out] dst    Destination device pointer.
 *
 * @return @ref HIP_OK on success, or an error status.
 */
HipStatus_t hip_memcpy_host2device(std::size_t bytes, const void *src,
                                   void *dst) {
  return map_hip_error(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

/**
 * @brief Asynchronous host-to-device copy.
 *
 * @details
 * Delegates to @ref hip_memcpy with
 * @ref DeviceMemcpyKind::deviceMemcpyHostToDevice.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source host pointer.
 * @param[out] dst    Destination device pointer.
 * @param[in]  pinned Whether @p src is page-locked.
 *
 * @return @ref HIP_OK on success, or an error status.
 */
HipStatus_t hip_memcpy_host2device_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyHostToDevice, src, dst,
                    pinned);
}

/**
 * @brief Synchronous device-to-host copy.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination host pointer.
 *
 * @return @ref HIP_OK on success, or an error status.
 */
HipStatus_t hip_memcpy_device2host(std::size_t bytes, const void *src,
                                   void *dst) {
  return map_hip_error(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

/**
 * @brief Asynchronous device-to-host copy.
 *
 * @details
 * Delegates to @ref hip_memcpy with
 * @ref DeviceMemcpyKind::deviceMemcpyDeviceToHost.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination host pointer.
 * @param[in]  pinned Whether @p dst is page-locked.
 *
 * @return @ref HIP_OK on success, or an error status.
 */
HipStatus_t hip_memcpy_device2host_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToHost, src, dst,
                    pinned);
}

/**
 * @brief Device-to-device copy.
 *
 * @details
 * Delegates to @ref hip_memcpy with
 * @ref DeviceMemcpyKind::deviceMemcpyDeviceToDevice.
 *
 * @param[in]  bytes  Number of bytes.
 * @param[in]  src    Source device pointer.
 * @param[out] dst    Destination device pointer.
 *
 * @return @ref HIP_OK on success, or an error status.
 */
HipStatus_t hip_memcpy_device2device(std::size_t bytes, const void *src,
                                     void *dst) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToDevice, src,
                    dst, false);
}

/**
 * @brief Master memcpy dispatcher for HIP.
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
 * @return @ref HIP_OK on success, or an error status.
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

  const hipError_t sync_err = hipStreamSynchronize(stream);
  if (sync_err != hipSuccess && status.code == 0) {
    status = map_hip_sync_error(sync_err);
  }

  const hipError_t destroy_err = hipStreamDestroy(stream);
  if (destroy_err != hipSuccess && status.code == 0) {
    status = map_hip_stream_error(destroy_err);
  }

  return status;
}

#else // !__has_include(<hip/hip_runtime_api.h>)

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_memcpy_host2device(std::size_t, const void *, void *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_memcpy_host2device_async(std::size_t, const void *, void *,
                                         bool) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_memcpy_device2host(std::size_t, const void *, void *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_memcpy_device2host_async(std::size_t, const void *, void *,
                                         bool) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_memcpy_device2device(std::size_t, const void *, void *) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

/** @brief Stub: HIP runtime headers not available. */
HipStatus_t hip_memcpy(std::size_t, DeviceMemcpyKind, const void *, void *,
                       bool) {
  return HipStatus_t{.code = -1,
                     .msg = "HIP runtime headers not available at"
                            " build/lint time"};
}

#endif
#endif /* NOVA_HAS_HIP */
