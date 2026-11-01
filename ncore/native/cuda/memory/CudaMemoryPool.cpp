/**
 * @file CudaMemoryPool.cpp
 * @brief Per-device CUDA memory-pool implementation.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_CUDA
#if __has_include(<cuda_runtime_api.h>)

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>

#include "CudaEngineStream.hpp"
#include "CudaMemoryPool.hpp"
#include "common.hh"

std::array<CudaDevicePool, NCORE_COMPILE_TIME_MAX_GPUS>
    CudaDevicePool::pools_{};

namespace {

novaStatus_t statusFor(novaError_t error) {
  return {.err = error, .message = nova_get_error_msg(error, nullptr)};
}

novaStatus_t statusFor(cudaError_t error) { return mapError2Status(error); }

novaStatus_t statusForCrossStream() {
  return {.err = novaInvalidValue,
          .message =
              "The buffer was allocated on another stream; cross-stream use "
              "requires an explicit event dependency.\n"};
}

} // namespace

/**
 * @brief Runs an operation on the lazy per-device pool slot.
 *
 * @details
 * Validates the device and stream, then runs @p operation under the
 * pool mutex.  A destroyed slot rejects later operations.  When
 * @p initialize_pool is true and the slot is uninitialized it is
 * initialized first; failure aborts before running @p operation.
 * The active CUDA device is switched to @p device_id for the duration
 * of the call and restored afterwards.
 *
 * @tparam Operation Callable taking a @ref CudaDevicePool& and
 *                   returning @ref novaStatus_t.
 * @param[in]  device_id Runtime device ordinal.
 * @param[in]  stream    Caller stream for ordering, or null to select
 *                       the device engine stream.
 * @param[in]  initialize_pool When true, lazily initialize the pool
 *                             before running @p operation.
 * @param[in]  operation Work to execute on the pool slot.
 *
 * @return Status from validation, initialization, or @p operation.
 */
template <typename Operation>
novaStatus_t CudaDevicePool::withPool(std::int32_t device_id,
                                      cudaStream_t stream, bool initialize_pool,
                                      Operation &&operation) {
  novaStatus_t status = validateDeviceId(device_id);
  if (status.err != novaSuccess) {
    return status;
  }
  status = validateStreamDevice(device_id, stream);
  if (status.err != novaSuccess) {
    return status;
  }

  CudaDevicePool &pool = pools_[static_cast<std::size_t>(device_id)];
  return withDeviceId(device_id, [&]() {
    std::scoped_lock lock(pool.mutex_);
    if (pool.state_ == State::Destroyed) {
      return statusFor(novaInvalidResourceHandle);
    }
    if (initialize_pool && pool.state_ == State::Uninitialized) {
      const novaStatus_t initStatus = pool.initialize(device_id);
      if (initStatus.err != novaSuccess) {
        return initStatus;
      }
    }
    return std::forward<Operation>(operation)(pool);
  });
}

novaStatus_t CudaDevicePool::initialize(std::int32_t device_id) {
  bool supported = false;
  const novaStatus_t supportStatus = supportMemoryPool(device_id, &supported);
  if (supportStatus.err != novaSuccess) {
    return supportStatus;
  }
  if (!supported) {
    native_pool_supported_ = false;
    state_ = State::Ready;
    return CUDA_OK;
  }
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = device_id;

  cudaMemPool_t nativePool = nullptr;
  cudaError_t error = cudaMemPoolCreate(&nativePool, &props);
  if (error != cudaSuccess) {
    return statusFor(error);
  }

  cuuint64_t threshold = std::numeric_limits<cuuint64_t>::max();
  error = cudaMemPoolSetAttribute(nativePool, cudaMemPoolAttrReleaseThreshold,
                                  &threshold);
  if (error != cudaSuccess) {
    (void)cudaMemPoolDestroy(nativePool);
    return statusFor(error);
  }

  native_pool_ = nativePool;
  native_pool_supported_ = true;
  state_ = State::Ready;
  return CUDA_OK;
}

novaStatus_t CudaDevicePool::currentDeviceId(std::int32_t *device_id) {
  if (device_id == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  int current = -1;
  const cudaError_t error = cudaGetDevice(&current);
  if (error != cudaSuccess) {
    return statusFor(error);
  }
  const novaStatus_t status = validateDeviceId(current);
  if (status.err == novaSuccess) {
    *device_id = current;
  }
  return status;
}

novaStatus_t CudaDevicePool::reserve(std::int32_t device_id, std::size_t bytes,
                                     bool pinned, cudaBuffer_t *out,
                                     cudaStream_t stream) {
  if (out == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  if (bytes == 0) {
    return statusFor(novaInvalidValue);
  }
  *out = cudaBuffer_t{};
  return withPool(device_id, stream, !pinned, [&](CudaDevicePool &pool) {
    return pool.reserveBuffer(bytes, pinned, out, stream);
  });
}

/**
 * @brief Allocates one buffer from the pool (or directly).
 *
 * @details
 * Pinned allocations use @c cudaMallocHost and bypass the pool.
 * Device allocations on pool-capable devices use
 * @c cudaMallocFromPoolAsync.  When @p stream is null the allocation
 * runs on the device engine stream and returns without
 * synchronization.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  pinned When true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 * @param[in]  stream Caller stream for chaining, or null to select the
 *                    device engine stream.
 *
 * @return Status; @c novaSuccess on success, or a mapped CUDA error.
 */
novaStatus_t CudaDevicePool::reserveBuffer(std::size_t bytes, bool pinned,
                                           cudaBuffer_t *out,
                                           cudaStream_t stream) {
  void *pointer = nullptr;
  cudaStream_t work = stream;

  if (work == nullptr) {
    const novaStatus_t engineStatus = CudaEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }

  if (pinned) {
    const cudaError_t error = cudaMallocHost(&pointer, bytes);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
  } else if (native_pool_supported_) {

    const cudaError_t error = cudaMallocFromPoolAsync(
        &pointer, bytes, static_cast<cudaMemPool_t>(native_pool_), work);
    if (error != cudaSuccess) {
      return statusFor(error);
    }

  } else {
    const cudaError_t error = cudaMalloc(&pointer, bytes);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
  }

  out->ptr = pointer;
  out->bytes = bytes;
  out->isPinned = pinned;
  out->streamOrigin = work;
  return CUDA_OK;
}

novaStatus_t CudaDevicePool::release(std::int32_t device_id, cudaBuffer_t *buf,
                                     cudaStream_t stream) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  return withPool(device_id, stream, !buf->isPinned, [&](CudaDevicePool &pool) {
    return pool.releaseBuffer(buf, stream);
  });
}

/**
 * @brief Frees one buffer through its pool.
 *
 * @details
 * Pinned memory uses @c cudaFreeHost.  Device memory on pool-capable
 * devices is freed asynchronously via @c cudaFreeAsync.  When
 * @p stream is null the free runs on the device engine stream and
 * returns without synchronization.
 *
 * @param[in,out] buf  Pointer to the buffer descriptor to free.
 * @param[in]     stream Caller stream for chaining, or null to select the
 *                       device engine stream.
 *
 * @return Status; @c novaSuccess on success, or a mapped CUDA error.
 */
novaStatus_t CudaDevicePool::releaseBuffer(cudaBuffer_t *buf,
                                           cudaStream_t stream) {

  cudaStream_t work = stream;

  if (work == nullptr) {
    const novaStatus_t engineStatus = CudaEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }

  if (!buf->isPinned && buf->streamOrigin != work) {
    return statusForCrossStream();
  }

  if (buf->isPinned) {
    const cudaError_t error = cudaFreeHost(buf->ptr);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
  } else if (native_pool_supported_) {

    const cudaError_t error = cudaFreeAsync(buf->ptr, work);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
    *buf = cudaBuffer_t{};

    return CUDA_OK;
  } else {
    const cudaError_t error = cudaFree(buf->ptr);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
  }

  *buf = cudaBuffer_t{};
  return CUDA_OK;
}

novaStatus_t CudaDevicePool::resize(std::int32_t device_id, cudaBuffer_t *buf,
                                    std::size_t new_bytes,
                                    cudaStream_t stream) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  if (new_bytes == 0) {
    return statusFor(novaInvalidValue);
  }
  return withPool(device_id, stream, !buf->isPinned, [&](CudaDevicePool &pool) {
    return pool.resizeBuffer(buf, new_bytes, stream);
  });
}

/**
 * @brief Resizes one buffer preserving its contents.
 *
 * @details
 * Allocates replacement storage, copies the preserved prefix
 * (@c std::min(old, new)), then frees the old buffer.  All device
 * work runs on the same stream ordering.  When @p stream is null the
 * work runs on the device engine stream and returns without
 * synchronization.
 *
 * @param[in,out] buf       Pointer to the buffer descriptor to resize.
 * @param[in]     new_bytes New size in bytes (> 0).
 * @param[in]     stream    Caller stream for chaining, or null to select the
 *                          device engine stream.
 *
 * @return Status; @c novaSuccess on success, or a mapped CUDA error.
 *
 * @warning On failure the original buffer may already be freed; do not
 *          reuse its pointer after a failed resize.
 */
novaStatus_t CudaDevicePool::resizeBuffer(cudaBuffer_t *buf,
                                          std::size_t new_bytes,
                                          cudaStream_t stream) {
  const std::size_t copyBytes = std::min(buf->bytes, new_bytes);
  void *newPointer = nullptr;
  cudaStream_t work = stream;

  if (work == nullptr) {
    const novaStatus_t engineStatus = CudaEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }

  if (!buf->isPinned && buf->streamOrigin != work) {
    return statusForCrossStream();
  }

  if (buf->isPinned) {
    cudaError_t error = cudaMallocHost(&newPointer, new_bytes);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
    std::memcpy(newPointer, buf->ptr, copyBytes);
    error = cudaFreeHost(buf->ptr);
    if (error != cudaSuccess) {
      (void)cudaFreeHost(newPointer);
      return statusFor(error);
    }
  } else if (native_pool_supported_) {

    cudaMemPool_t nativePool = static_cast<cudaMemPool_t>(native_pool_);
    cudaError_t error =
        cudaMallocFromPoolAsync(&newPointer, new_bytes, nativePool, work);
    if (error != cudaSuccess) {
      return statusFor(error);
    }

    error = cudaMemcpyAsync(newPointer, buf->ptr, copyBytes,
                            cudaMemcpyDeviceToDevice, work);
    if (error != cudaSuccess) {
      const cudaError_t freeAsyncErr = cudaFreeAsync(newPointer, work);
      if (freeAsyncErr != cudaSuccess) {
        return statusFor(freeAsyncErr);
      }
      return statusFor(error);
    }

    error = cudaFreeAsync(buf->ptr, work);

    if (error != cudaSuccess) {
      const cudaError_t freeAsyncErr = cudaFreeAsync(newPointer, work);

      if (freeAsyncErr != cudaSuccess) {
        return statusFor(freeAsyncErr);
      }
      return statusFor(error);
    }

    buf->ptr = newPointer;
    buf->bytes = new_bytes;
    buf->streamOrigin = work;
    return CUDA_OK;
  } else {
    cudaError_t error = cudaMalloc(&newPointer, new_bytes);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
    error =
        cudaMemcpy(newPointer, buf->ptr, copyBytes, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
      (void)cudaFree(newPointer);
      return statusFor(error);
    }
    error = cudaFree(buf->ptr);
    if (error != cudaSuccess) {
      (void)cudaFree(newPointer);
      return statusFor(error);
    }
  }

  buf->ptr = newPointer;
  buf->bytes = new_bytes;
  buf->streamOrigin = work;
  return CUDA_OK;
}

/**
 * @brief Trims reclaimable backing from one device pool.
 *
 * @details
 * No-op when the device does not support native memory pools.
 * Otherwise the device engine stream drains first, because a free
 * still queued there keeps its block counted as used, and the pool
 * trims to at least @p min_bytes_to_keep resident bytes via
 * @c cudaMemPoolTrimTo.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[in]  min_bytes_to_keep Minimum resident bytes to retain.
 *
 * @return Status; @c novaSuccess when the pool is unsupported or trimmed.
 */
novaStatus_t CudaDevicePool::emptyCache(std::int32_t device_id,
                                        std::size_t min_bytes_to_keep) {
  return withPool(device_id, nullptr, true, [&](CudaDevicePool &pool) {
    if (!pool.native_pool_supported_) {
      return CUDA_OK;
    }
    const novaStatus_t drainStatus = CudaEngineStream::drain(device_id);
    if (drainStatus.err != novaSuccess) {
      return drainStatus;
    }
    return statusFor(cudaMemPoolTrimTo(
        static_cast<cudaMemPool_t>(pool.native_pool_), min_bytes_to_keep));
  });
}

/**
 * @brief Reads runtime statistics for one device pool.
 *
 * @details
 * Zeroes @p out and queries the pool attributes via
 * @c cudaMemPoolGetAttribute.  When the device does not support native
 * memory pools the call reports @c novaMemoryPoolsNotNativelySupported.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[out] out       Receives the per-device pool counters.
 *
 * @return Status; @c novaSuccess on success;
 *         @c novaMemoryPoolsNotNativelySupported when pools are unsupported.
 */
novaStatus_t CudaDevicePool::stats(std::int32_t device_id,
                                   DeviceMemoryPoolStats *out) {
  if (out == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  *out = DeviceMemoryPoolStats{};
  return withPool(device_id, nullptr, true, [&](CudaDevicePool &pool) {
    out->device_id = device_id;
    out->native_pool_supported = pool.native_pool_supported_;
    if (!pool.native_pool_supported_) {
      return statusFor(novaMemoryPoolsNotNativelySupported);
    }

    cudaMemPool_t nativePool = static_cast<cudaMemPool_t>(pool.native_pool_);
    cuuint64_t reservedCurrent = 0;
    cuuint64_t reservedHigh = 0;
    cuuint64_t usedCurrent = 0;
    cuuint64_t usedHigh = 0;
    cuuint64_t releaseThreshold = 0;
    cudaError_t error = cudaMemPoolGetAttribute(
        nativePool, cudaMemPoolAttrReservedMemCurrent, &reservedCurrent);
    if (error == cudaSuccess) {
      error = cudaMemPoolGetAttribute(
          nativePool, cudaMemPoolAttrReservedMemHigh, &reservedHigh);
    }
    if (error == cudaSuccess) {
      error = cudaMemPoolGetAttribute(nativePool, cudaMemPoolAttrUsedMemCurrent,
                                      &usedCurrent);
    }
    if (error == cudaSuccess) {
      error = cudaMemPoolGetAttribute(nativePool, cudaMemPoolAttrUsedMemHigh,
                                      &usedHigh);
    }
    if (error == cudaSuccess) {
      error = cudaMemPoolGetAttribute(
          nativePool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold);
    }
    if (error != cudaSuccess) {
      return statusFor(error);
    }

    out->reserved_current = reservedCurrent;
    out->reserved_high = reservedHigh;
    out->used_current = usedCurrent;
    out->used_high = usedHigh;
    out->release_threshold = releaseThreshold;
    return CUDA_OK;
  });
}

/**
 * @brief Sets the release threshold for one device pool.
 *
 * @details
 * When the device does not support native memory pools the call reports
 * @c novaMemoryPoolsNotNativelySupported.  Otherwise it sets
 * @c cudaMemPoolAttrReleaseThreshold on the pool.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[in]  bytes     New release threshold in bytes.
 *
 * @return Status; @c novaSuccess on success;
 *         @c novaMemoryPoolsNotNativelySupported when pools are unsupported.
 */
novaStatus_t CudaDevicePool::setReleaseThreshold(std::int32_t device_id,
                                                 std::uint64_t bytes) {
  return withPool(device_id, nullptr, true, [&](CudaDevicePool &pool) {
    if (!pool.native_pool_supported_) {
      return statusFor(novaMemoryPoolsNotNativelySupported);
    }
    cuuint64_t threshold = static_cast<cuuint64_t>(bytes);
    const cudaError_t error =
        cudaMemPoolSetAttribute(static_cast<cudaMemPool_t>(pool.native_pool_),
                                cudaMemPoolAttrReleaseThreshold, &threshold);
    if (error != cudaSuccess) {
      return statusFor(error);
    }
    return CUDA_OK;
  });
}

/**
 * @brief Destroys one idle device pool.
 *
 * @details
 * When the device does not support native memory pools the call is a
 * no-op and reports success.  Otherwise the device engine stream drains
 * first, because a free still queued there keeps the pool alive after the
 * used counter drops.  The pool is destroyed only when no memory is
 * currently in use; a busy pool reports @c novaResourceBusy.  After
 * success the slot is marked destroyed and rejects later operations.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 *
 * @return Status; @c novaSuccess on success; @c novaResourceBusy when
 *         the pool still holds memory.
 */
novaStatus_t CudaDevicePool::destroy(std::int32_t device_id) {
  return withPool(device_id, nullptr, true, [&](CudaDevicePool &pool) {
    if (!pool.native_pool_supported_) {
      return CUDA_OK;
    }

    const novaStatus_t drainStatus = CudaEngineStream::drain(device_id);
    if (drainStatus.err != novaSuccess) {
      return drainStatus;
    }

    cuuint64_t usedCurrent = 0;
    const cudaError_t statsError =
        cudaMemPoolGetAttribute(static_cast<cudaMemPool_t>(pool.native_pool_),
                                cudaMemPoolAttrUsedMemCurrent, &usedCurrent);
    if (statsError != cudaSuccess) {
      return statusFor(statsError);
    }
    if (usedCurrent != 0) {
      return statusFor(novaResourceBusy);
    }

    const cudaError_t error =
        cudaMemPoolDestroy(static_cast<cudaMemPool_t>(pool.native_pool_));
    if (error != cudaSuccess) {
      return statusFor(error);
    }
    pool.native_pool_ = nullptr;
    pool.state_ = State::Destroyed;
    return CUDA_OK;
  });
}

#else // !__has_include(<cuda_runtime_api.h>)

namespace {
novaStatus_t cudaPoolUnavailable() {
  return {.err = novaBackendNotCompiled,
          .message = nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}
} // namespace

novaStatus_t CudaDevicePool::currentDeviceId(std::int32_t *) {
  return cudaPoolUnavailable();
}
novaStatus_t CudaDevicePool::reserve(std::int32_t, std::size_t, bool,
                                     cudaBuffer_t *, cudaStream_t) {
  return cudaPoolUnavailable();
}
novaStatus_t CudaDevicePool::release(std::int32_t, cudaBuffer_t *,
                                     cudaStream_t) {
  return cudaPoolUnavailable();
}
novaStatus_t CudaDevicePool::resize(std::int32_t, cudaBuffer_t *, std::size_t,
                                    cudaStream_t) {
  return cudaPoolUnavailable();
}
novaStatus_t CudaDevicePool::emptyCache(std::int32_t, std::size_t) {
  return cudaPoolUnavailable();
}
novaStatus_t CudaDevicePool::stats(std::int32_t, DeviceMemoryPoolStats *) {
  return cudaPoolUnavailable();
}
novaStatus_t CudaDevicePool::setReleaseThreshold(std::int32_t, std::uint64_t) {
  return cudaPoolUnavailable();
}
novaStatus_t CudaDevicePool::destroy(std::int32_t) {
  return cudaPoolUnavailable();
}

#endif
#endif
