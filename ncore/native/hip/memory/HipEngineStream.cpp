/**
 * @file HipEngineStream.cpp
 * @brief Per-device non-blocking HIP engine stream.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#if defined(NOVA_HAS_HIP) && __has_include(<hip/hip_runtime_api.h>)
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>
#endif

#include "HipEngineStream.hpp"

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)

#include "common.hh"

std::array<HipEngineStream, NCORE_COMPILE_TIME_MAX_GPUS>
    HipEngineStream::slots_{};

novaStatus_t HipEngineStream::initialize(std::int32_t device_id) {
  const novaStatus_t status = withDeviceId(device_id, [&]() -> novaStatus_t {
    const hipError_t err =
        hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
    if (err != hipSuccess) {
      return mapError2Status(err);
    }
    return {.err = novaSuccess,
            .message = nova_get_error_msg(novaSuccess, nullptr)};
  });
  if (status.err != novaSuccess) {
    state_ = State::Failed;
    return status;
  }
  state_ = State::Ready;
  return status;
}

novaStatus_t HipEngineStream::get(std::int32_t device_id, hipStream_t *out) {
  if (out == nullptr) {
    return {.err = novaInvalidPointer,
            .message = nova_get_error_msg(novaInvalidPointer, nullptr)};
  }
  if (device_id < 0 || device_id >= NCORE_COMPILE_TIME_MAX_GPUS) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  auto &slot = slots_[static_cast<std::size_t>(device_id)];
  const std::scoped_lock lock(slot.mutex_);
  if (slot.state_ == State::Failed) {
    return {.err = novaResourceBusy,
            .message = nova_get_error_msg(novaResourceBusy, nullptr)};
  }
  if (slot.state_ == State::Uninitialized) {
    const novaStatus_t status = slot.initialize(device_id);
    if (status.err != novaSuccess) {
      return status;
    }
  }
  *out = slot.stream_;
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

novaStatus_t HipEngineStream::current(hipStream_t *out) {
  if (out == nullptr) {
    return {.err = novaInvalidPointer,
            .message = nova_get_error_msg(novaInvalidPointer, nullptr)};
  }
  int deviceId = 0;
  const hipError_t err = hipGetDevice(&deviceId);
  if (err != hipSuccess) {
    return mapError2Status(err);
  }
  return get(static_cast<std::int32_t>(deviceId), out);
}

novaStatus_t HipEngineStream::drain(std::int32_t device_id) {
  if (device_id < 0 || device_id >= NCORE_COMPILE_TIME_MAX_GPUS) {
    return {.err = novaInvalidDeviceId,
            .message = nova_get_error_msg(novaInvalidDeviceId, nullptr)};
  }
  auto &slot = slots_[static_cast<std::size_t>(device_id)];
  const std::scoped_lock lock(slot.mutex_);
  if (slot.state_ != State::Ready || slot.stream_ == nullptr) {
    return {.err = novaSuccess,
            .message = nova_get_error_msg(novaSuccess, nullptr)};
  }
  const hipError_t err = hipStreamSynchronize(slot.stream_);
  if (err != hipSuccess) {
    return mapError2Status(err);
  }
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

#else // !__has_include(<hip/hip_runtime_api.h>)

namespace {
novaStatus_t hipEngineUnavailable() {
  return {.err = novaBackendNotCompiled,
          .message = "HIP support is not available in this build.\n"};
}
} // namespace

std::array<HipEngineStream, NCORE_COMPILE_TIME_MAX_GPUS>
    HipEngineStream::slots_{};

novaStatus_t HipEngineStream::get(std::int32_t, hipStream_t *) {
  return hipEngineUnavailable();
}

novaStatus_t HipEngineStream::current(hipStream_t *) {
  return hipEngineUnavailable();
}

novaStatus_t HipEngineStream::drain(std::int32_t) {
  return hipEngineUnavailable();
}

#endif
#endif /* NOVA_HAS_HIP */
