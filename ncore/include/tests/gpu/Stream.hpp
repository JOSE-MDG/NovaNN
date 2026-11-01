/**
 * @file Stream.hpp
 * @brief Raw runtime stream aliases and a scoped stream for the pool and
 *        engine-stream suites.
 */

#pragma once

#include <ncore/headeronly/stream_types.h>

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

#ifdef NOVA_HAS_CUDA
#include <cuda_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

namespace tests::gpu {

#ifdef NOVA_HAS_CUDA
using RuntimeStream = cudaStream_t;
using RuntimeError = cudaError_t;
constexpr RuntimeError kRuntimeSuccess = cudaSuccess;

/// @brief Runtime device of the engine, bypassing the NovaNN device layer.
inline RuntimeError runtimeGetDevice(int *device_id) {
  return cudaGetDevice(device_id);
}

/// @brief Creates a blocking stream the test fully controls.
inline RuntimeError runtimeStreamCreate(RuntimeStream *stream) {
  return cudaStreamCreate(stream);
}

inline RuntimeError runtimeStreamSynchronize(RuntimeStream stream) {
  return cudaStreamSynchronize(stream);
}

inline RuntimeError runtimeStreamDestroy(RuntimeStream stream) {
  return cudaStreamDestroy(stream);
}
#else
using RuntimeStream = hipStream_t;
using RuntimeError = hipError_t;
constexpr RuntimeError kRuntimeSuccess = hipSuccess;

/// @brief Runtime device of the engine, bypassing the NovaNN device layer.
inline RuntimeError runtimeGetDevice(int *device_id) {
  return hipGetDevice(device_id);
}

/// @brief Creates a blocking stream the test fully controls.
inline RuntimeError runtimeStreamCreate(RuntimeStream *stream) {
  return hipStreamCreate(stream);
}

inline RuntimeError runtimeStreamSynchronize(RuntimeStream stream) {
  return hipStreamSynchronize(stream);
}

inline RuntimeError runtimeStreamDestroy(RuntimeStream stream) {
  return hipStreamDestroy(stream);
}
#endif

/**
 * @class ScopedStream
 * @brief Runtime stream destroyed on scope exit.
 *
 * The destructor synchronizes before destroying: a free still queued on
 * the stream would otherwise dangle past the handle teardown.
 */
class ScopedStream final {
public:
  ScopedStream() = default;
  ScopedStream(const ScopedStream &) = delete;
  ScopedStream &operator=(const ScopedStream &) = delete;

  ~ScopedStream() {
    if (stream_ != nullptr) {
      (void)runtimeStreamSynchronize(stream_);
      (void)runtimeStreamDestroy(stream_);
    }
  }

  RuntimeError create() { return runtimeStreamCreate(&stream_); }
  [[nodiscard]] RuntimeStream get() const { return stream_; }
  [[nodiscard]] void *opaque() const { return static_cast<void *>(stream_); }

private:
  RuntimeStream stream_{};
};

} // namespace tests::gpu

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP
