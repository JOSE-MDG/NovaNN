/**
 * @file hip_io.cpp
 * @brief Implementation of HIP memory copy helpers.
 *
 * Mirrors the CUDA helper implementation in cuda_io.cpp, but using the HIP
 * Runtime API (hipMemcpy / hipMemcpyAsync / hipStream*).
 */

#include "hip_io.hpp"

#if __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

/**
 * @brief Map a HIP runtime error to a HipStatus_t code/message pair.
 *
 * @param err HIP error value returned by the runtime API.
 * @return HipStatus_t with code 0 on success, or an application-level code
 *         describing the failure class.
 */
static HipStatus_t map_hip_error(hipError_t err) {
  HipStatus_t hstatus = {};
  switch (err) {
  case hipSuccess: {
    hstatus.msg = "ok";
    hstatus.code = 0;
    return hstatus;
  }
  case hipErrorInvalidValue: {
    hstatus.msg = hipGetErrorString(hipErrorInvalidValue);
    hstatus.code = 1;
    return hstatus;
  }
  case hipErrorInvalidMemcpyDirection: {
    hstatus.msg = hipGetErrorString(hipErrorInvalidMemcpyDirection);
    hstatus.code = 2;
    return hstatus;
  }
  default: {
    hstatus.msg = hipGetErrorString(err);
    hstatus.code = -1;
    return hstatus;
  }
  }
}

/**
 * @brief Map stream creation/destruction errors to HipStatus_t.
 *
 * @param err HIP error value returned by hipStreamCreate / hipStreamDestroy.
 * @return HipStatus_t with code 0 on success, or an application-level code.
 */
static HipStatus_t map_hip_stream_error(hipError_t err) {
  HipStatus_t hstatus = {};
  switch (err) {
  case hipSuccess: {
    hstatus.msg = "ok";
    hstatus.code = 0;
    return hstatus;
  }
  case hipErrorInvalidValue: {
    hstatus.msg = hipGetErrorString(hipErrorInvalidValue);
    hstatus.code = 1;
    return hstatus;
  }
  case hipErrorInvalidDevice: {
    hstatus.msg = hipGetErrorString(hipErrorInvalidDevice);
    hstatus.code = 2;
    return hstatus;
  }
  default: {
    hstatus.msg = hipGetErrorString(err);
    hstatus.code = -1;
    return hstatus;
  }
  }
}

/**
 * @brief Map stream synchronization errors to HipStatus_t.
 *
 * @param err HIP error value returned by hipStreamSynchronize.
 * @return HipStatus_t with code 0 on success, or an application-level code.
 */
static HipStatus_t map_hip_sync_error(hipError_t err) {
  HipStatus_t hstatus = {};
  switch (err) {
  case hipSuccess: {
    hstatus.msg = "ok";
    hstatus.code = 0;
    return hstatus;
  }
  case hipErrorInvalidResourceHandle: {
    hstatus.msg = hipGetErrorString(hipErrorInvalidResourceHandle);
    hstatus.code = 1;
    return hstatus;
  }
  default: {
    hstatus.msg = hipGetErrorString(err);
    hstatus.code = -1;
    return hstatus;
  }
  }
}

/**
 * @brief Convert the device-agnostic copy kind to HIP's runtime enum.
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
 * @brief Synchronous host-to-device copy wrapper.
 */
HipStatus_t hip_memcpy_host2device(std::size_t bytes, const void *src,
                                   void *dst) {
  hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
  return map_hip_error(err);
}

/**
 * @brief Host-to-device copy with async path for pinned host memory.
 */
static HipStatus_t hip_memcpy_host2device_async(std::size_t bytes,
                                                hipStream_t stream,
                                                const void *src, void *dst,
                                                bool pinned) {
  if (pinned) {
    hipError_t err =
        hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
    return map_hip_error(err);
  }
  hipError_t err = hipStreamDestroy(stream);
  stream = nullptr;
  hip_memcpy_host2device(bytes, src, dst);
  return map_hip_error(err);
}

/**
 * @brief Synchronous device-to-host copy wrapper.
 */
HipStatus_t hip_memcpy_device2host(std::size_t bytes, const void *src,
                                   void *dst) {
  hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
  return map_hip_error(err);
}

/**
 * @brief Device-to-host copy with async path for pinned host memory.
 */
static HipStatus_t hip_memcpy_device2host_async(std::size_t bytes,
                                                hipStream_t stream,
                                                const void *src, void *dst,
                                                bool pinned) {

  if (pinned) {
    hipError_t err =
        hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, nullptr);
    map_hip_error(err);
  }
  hipError_t err = hipStreamDestroy(stream);
  stream = nullptr;
  hip_memcpy_device2host(bytes, src, dst);
  return map_hip_error(err);
}

/**
 * @brief Async device-to-device copy wrapper.
 */
static HipStatus_t hip_memcpy_device2device(std::size_t bytes,
                                            hipStream_t stream,
                                            const void *src, void *dst) {
  hipError_t err =
      hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);
  return map_hip_error(err);
}

/**
 * @brief Public host-to-device copy helper with optional async transfer.
 */
HipStatus_t hip_memcpy_host2device_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyHostToDevice, src, dst,
                    pinned);
}

/**
 * @brief Public device-to-host copy helper with optional async transfer.
 */
HipStatus_t hip_memcpy_device2host_async(std::size_t bytes, const void *src,
                                         void *dst, bool pinned) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToHost, src, dst,
                    pinned);
}

/**
 * @brief Public device-to-device copy helper.
 */
HipStatus_t hip_memcpy_device2device(std::size_t bytes, const void *src,
                                     void *dst) {
  return hip_memcpy(bytes, DeviceMemcpyKind::deviceMemcpyDeviceToDevice, src,
                    dst, false);
}

/**
 * @brief High-level copy helper that creates and synchronizes a stream.
 */
HipStatus_t hip_memcpy(std::size_t bytes, DeviceMemcpyKind kind,
                       const void *src, void *dst, bool is_pinned) {

  HipStatus_t status = {};
  hipStream_t stream = nullptr;
  hipError_t stream_err = hipStreamCreate(&stream);
  if (stream_err != hipSuccess) {
    return map_hip_stream_error(stream_err);
  }
  switch (map_hip_memcpy_kind(kind)) {
  case hipMemcpyHostToDevice:
    status =
        hip_memcpy_host2device_async(bytes, stream, src, dst, is_pinned);
    break;
  case hipMemcpyDeviceToHost:
    status =
        hip_memcpy_device2host_async(bytes, stream, src, dst, is_pinned);
    break;
  case hipMemcpyDeviceToDevice:
    status = hip_memcpy_device2device(bytes, stream, src, dst);
    break;
  default:
    break;
  }

  if (stream != nullptr) {
    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
      return map_hip_sync_error(sync_err);
    }
  }
  return status;
}

#else

/**
 * @brief Fallback host-to-device copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_host2device(std::size_t, const void *, void *) {
  return HipStatus_t{
      .code = -1,
      .msg = "HIP runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback host-to-device async copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_host2device_async(std::size_t, const void *, void *,
                                         bool) {
  return HipStatus_t{
      .code = -1,
      .msg = "HIP runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback device-to-host copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_device2host(std::size_t, const void *, void *) {
  return HipStatus_t{
      .code = -1,
      .msg = "HIP runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback device-to-host async copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_device2host_async(std::size_t, const void *, void *,
                                         bool) {
  return HipStatus_t{
      .code = -1,
      .msg = "HIP runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback device-to-device copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy_device2device(std::size_t, const void *, void *) {
  return HipStatus_t{
      .code = -1,
      .msg = "HIP runtime headers not available at build/lint time"};
}

/**
 * @brief Fallback generic copy when HIP headers are unavailable.
 */
HipStatus_t hip_memcpy(std::size_t, DeviceMemcpyKind, const void *, void *,
                       bool) {
  return HipStatus_t{
      .code = -1,
      .msg = "HIP runtime headers not available at build/lint time"};
}

#endif
