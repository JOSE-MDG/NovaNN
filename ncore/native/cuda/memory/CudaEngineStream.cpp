/**
 * @file CudaEngineStream.cpp
 * @brief Per-device non-blocking CUDA engine stream.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#if defined(NOVA_HAS_CUDA) && __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#endif

#include "CudaEngineStream.hpp"

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)

#include "common.hh"

std::array<CudaEngineStream, NCORE_COMPILE_TIME_MAX_GPUS>
    CudaEngineStream::slots_{};

novaStatus_t CudaEngineStream::initialize(std::int32_t device_id) {
  const novaStatus_t status = withDeviceId(device_id, [&]() -> novaStatus_t {
    const cudaError_t err =
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
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

novaStatus_t CudaEngineStream::get(std::int32_t device_id, cudaStream_t *out) {
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

novaStatus_t CudaEngineStream::current(cudaStream_t *out) {
  if (out == nullptr) {
    return {.err = novaInvalidPointer,
            .message = nova_get_error_msg(novaInvalidPointer, nullptr)};
  }
  int deviceId = 0;
  const cudaError_t err = cudaGetDevice(&deviceId);
  if (err != cudaSuccess) {
    return mapError2Status(err);
  }
  return get(static_cast<std::int32_t>(deviceId), out);
}

novaStatus_t CudaEngineStream::drain(std::int32_t device_id) {
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
  const cudaError_t err = cudaStreamSynchronize(slot.stream_);
  if (err != cudaSuccess) {
    return mapError2Status(err);
  }
  return {.err = novaSuccess,
          .message = nova_get_error_msg(novaSuccess, nullptr)};
}

#else // !__has_include(<cuda_runtime_api.h>)

namespace {
novaStatus_t cudaEngineUnavailable() {
  return {.err = novaBackendNotCompiled,
          .message = "CUDA support is not available in this build.\n"};
}
} // namespace

std::array<CudaEngineStream, NCORE_COMPILE_TIME_MAX_GPUS>
    CudaEngineStream::slots_{};

novaStatus_t CudaEngineStream::get(std::int32_t, cudaStream_t *) {
  return cudaEngineUnavailable();
}

novaStatus_t CudaEngineStream::current(cudaStream_t *) {
  return cudaEngineUnavailable();
}

novaStatus_t CudaEngineStream::drain(std::int32_t) {
  return cudaEngineUnavailable();
}

#endif
#endif /* NOVA_HAS_CUDA */
