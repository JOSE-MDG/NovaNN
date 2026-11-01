/**
 * @file HipMemoryPool.cpp
 * @brief Per-device HIP memory-pool implementation.
 */

#include <ncore/core/status.h>

#ifdef NOVA_HAS_HIP
#if __has_include(<hip/hip_runtime_api.h>)

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>

#include "HipEngineStream.hpp"
#include "HipMemoryPool.hpp"
#include "common.hh"

std::array<HipDevicePool, NCORE_COMPILE_TIME_MAX_GPUS> HipDevicePool::pools_{};

namespace {

novaStatus_t statusFor(novaError_t error) {
  return {.err = error, .message = nova_get_error_msg(error, nullptr)};
}

novaStatus_t statusFor(hipError_t error) { return mapError2Status(error); }

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
 * The active HIP device is switched to @p device_id for the duration
 * of the call and restored afterwards.
 *
 * @tparam Operation Callable taking a @ref HipDevicePool& and
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
novaStatus_t HipDevicePool::withPool(std::int32_t device_id, hipStream_t stream,
                                     bool initialize_pool,
                                     Operation &&operation) {
  novaStatus_t status = validateDeviceId(device_id);
  if (status.err != novaSuccess) {
    return status;
  }
  status = validateStreamDevice(device_id, stream);
  if (status.err != novaSuccess) {
    return status;
  }

  HipDevicePool &pool = pools_[static_cast<std::size_t>(device_id)];
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

/**
 * @brief Initializes the per-device native pool (or falls back).
 *
 * @details
 * Probes the device for native pool support.  Unsupported devices are
 * marked ready without a pool and use synchronous allocation; supported
 * devices create a pool with the maximum release threshold so memory
 * stays resident until explicitly released.
 *
 * @param[in]  device_id Runtime device ordinal to initialize.
 *
 * @return Status; @c novaSuccess when the slot is ready.
 */
novaStatus_t HipDevicePool::initialize(std::int32_t device_id) {
  bool supported = false;
  const novaStatus_t supportStatus = supportMemoryPool(device_id, &supported);
  if (supportStatus.err != novaSuccess) {
    return supportStatus;
  }
  if (!supported) {
    native_pool_supported_ = false;
    state_ = State::Ready;
    return HIP_OK;
  }

  hipMemPoolProps properties{};
  properties.allocType = hipMemAllocationTypePinned;
  properties.handleTypes = hipMemHandleTypeNone;
  properties.location.type = hipMemLocationTypeDevice;
  properties.location.id = device_id;

  hipMemPool_t nativePool = nullptr;
  hipError_t error = hipMemPoolCreate(&nativePool, &properties);
  if (error != hipSuccess) {
    return statusFor(error);
  }

  std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
  error = hipMemPoolSetAttribute(nativePool, hipMemPoolAttrReleaseThreshold,
                                 &threshold);
  if (error != hipSuccess) {
    (void)hipMemPoolDestroy(nativePool);
    return statusFor(error);
  }

  native_pool_ = nativePool;
  native_pool_supported_ = true;
  state_ = State::Ready;
  return HIP_OK;
}

/**
 * @brief Reports the current HIP device ordinal.
 *
 * @details
 * Queries the runtime and validates the result against the compile-time
 * device limit.
 *
 * @param[out] device_id Receives the current runtime device ordinal.
 *
 * @return Status; @c novaInvalidPointer when @p device_id is null;
 *         @c novaInvalidDeviceId when the ordinal is out of range.
 */
novaStatus_t HipDevicePool::currentDeviceId(std::int32_t *device_id) {
  if (device_id == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  int current = -1;
  const hipError_t error = hipGetDevice(&current);
  if (error != hipSuccess) {
    return statusFor(error);
  }
  const novaStatus_t status = validateDeviceId(current);
  if (status.err == novaSuccess) {
    *device_id = current;
  }
  return status;
}

/**
 * @brief Allocates a device or pinned-host buffer on one device.
 *
 * @details
 * Pinned allocations bypass the pool.  Device allocations use the
 * native pool when supported, otherwise synchronous @c hipMalloc.  A
 * null stream selects the device engine stream, and a caller stream is
 * enqueued without synchronization.
 *
 * @param[in]  device_id Runtime device ordinal selecting the pool.
 * @param[in]  bytes     Requested allocation size in bytes (> 0).
 * @param[in]  pinned    When true, allocate page-locked host memory.
 * @param[out] out       Receives the buffer descriptor on success.
 * @param[in]  stream    Caller stream for chaining, or null to select the
 *                       device engine stream.
 *
 * @return Status; @c novaInvalidPointer when @p out is null;
 *         @c novaInvalidValue when @p bytes is zero.
 */
novaStatus_t HipDevicePool::reserve(std::int32_t device_id, std::size_t bytes,
                                    bool pinned, hipBuffer_t *out,
                                    hipStream_t stream) {
  if (out == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  if (bytes == 0) {
    return statusFor(novaInvalidValue);
  }
  *out = hipBuffer_t{};
  return withPool(device_id, stream, !pinned, [&](HipDevicePool &pool) {
    return pool.reserveBuffer(bytes, pinned, out, stream);
  });
}

/**
 * @brief Allocates one buffer from the pool (or directly).
 *
 * @details
 * Pinned allocations use @c hipHostMalloc and bypass the pool.
 * Device allocations on pool-capable devices use
 * @c hipMallocFromPoolAsync.  When @p stream is null the allocation is
 * enqueued on the device engine stream without synchronization.  A caller
 * stream is enqueued without synchronization.
 *
 * @param[in]  bytes  Requested allocation size in bytes.
 * @param[in]  pinned When true, allocate page-locked host memory.
 * @param[out] out    Receives the buffer descriptor on success.
 * @param[in]  stream Caller stream for chaining, or null to select
 *                    the device engine stream.
 *
 * @return Status; @c novaSuccess on success, or a mapped HIP error.
 */
novaStatus_t HipDevicePool::reserveBuffer(std::size_t bytes, bool pinned,
                                          hipBuffer_t *out,
                                          hipStream_t stream) {
  void *pointer = nullptr;
  hipStream_t work = stream;

  if (work == nullptr) {
    const novaStatus_t engineStatus = HipEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }

  if (pinned) {
    const hipError_t error =
        hipHostMalloc(&pointer, bytes, hipHostMallocDefault);
    if (error != hipSuccess) {
      return statusFor(error);
    }
  } else if (native_pool_supported_) {
    const hipError_t error = hipMallocFromPoolAsync(
        &pointer, bytes, static_cast<hipMemPool_t>(native_pool_), work);
    if (error != hipSuccess) {
      return statusFor(error);
    }

  } else {
    const hipError_t error = hipMalloc(&pointer, bytes);
    if (error != hipSuccess) {
      return statusFor(error);
    }
  }

  out->ptr = pointer;
  out->bytes = bytes;
  out->isPinned = pinned;
  out->streamOrigin = work;
  return HIP_OK;
}

novaStatus_t HipDevicePool::release(std::int32_t device_id, hipBuffer_t *buf,
                                    hipStream_t stream) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  return withPool(device_id, stream, !buf->isPinned, [&](HipDevicePool &pool) {
    return pool.releaseBuffer(buf, stream);
  });
}

/**
 * @brief Frees one buffer through its pool.
 *
 * @details
 * Pinned memory uses @c hipFreeHost.  Device memory on pool-capable
 * devices is freed asynchronously via @c hipFreeAsync.  When
 * @p stream is null the free runs on the device engine stream and
 * returns without synchronization.
 *
 * @param[in,out] buf  Pointer to the buffer descriptor to free.
 * @param[in]     stream Caller stream for chaining, or null to select the
 *                       device engine stream.
 *
 * @return Status; @c novaSuccess on success, or a mapped HIP error.
 */
novaStatus_t HipDevicePool::releaseBuffer(hipBuffer_t *buf,
                                          hipStream_t stream) {
  hipStream_t work = stream;

  if (work == nullptr) {
    const novaStatus_t engineStatus = HipEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }

  if (!buf->isPinned && buf->streamOrigin != work) {
    return statusForCrossStream();
  }

  if (buf->isPinned) {
    const hipError_t error = hipFreeHost(buf->ptr);
    if (error != hipSuccess) {
      return statusFor(error);
    }
  } else if (native_pool_supported_) {
    const hipError_t error = hipFreeAsync(buf->ptr, work);
    if (error != hipSuccess) {
      return statusFor(error);
    }
    *buf = hipBuffer_t{};

    return HIP_OK;
  } else {
    const hipError_t error = hipFree(buf->ptr);
    if (error != hipSuccess) {
      return statusFor(error);
    }
  }

  *buf = hipBuffer_t{};
  return HIP_OK;
}

novaStatus_t HipDevicePool::resize(std::int32_t device_id, hipBuffer_t *buf,
                                   std::size_t new_bytes, hipStream_t stream) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  if (new_bytes == 0) {
    return statusFor(novaInvalidValue);
  }
  return withPool(device_id, stream, !buf->isPinned, [&](HipDevicePool &pool) {
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
 * @return Status; @c novaSuccess on success, or a mapped HIP error.
 *
 * @warning On failure the original buffer may already be freed; do not
 *          reuse its pointer after a failed resize.
 */
novaStatus_t HipDevicePool::resizeBuffer(hipBuffer_t *buf,
                                         std::size_t new_bytes,
                                         hipStream_t stream) {
  const std::size_t copyBytes = std::min(buf->bytes, new_bytes);
  void *newPointer = nullptr;
  hipStream_t work = stream;

  if (work == nullptr) {
    const novaStatus_t engineStatus = HipEngineStream::current(&work);
    if (engineStatus.err != novaSuccess) {
      return engineStatus;
    }
  }
  if (!buf->isPinned && buf->streamOrigin != work) {
    return statusForCrossStream();
  }

  if (buf->isPinned) {
    hipError_t error =
        hipHostMalloc(&newPointer, new_bytes, hipHostMallocDefault);
    if (error != hipSuccess) {
      return statusFor(error);
    }
    std::memcpy(newPointer, buf->ptr, copyBytes);
    error = hipFreeHost(buf->ptr);
    if (error != hipSuccess) {
      (void)hipFreeHost(newPointer);
      return statusFor(error);
    }
  } else if (native_pool_supported_) {
    hipMemPool_t nativePool = static_cast<hipMemPool_t>(native_pool_);

    hipError_t error =
        hipMallocFromPoolAsync(&newPointer, new_bytes, nativePool, work);
    if (error != hipSuccess) {
      return statusFor(error);
    }

    error = hipMemcpyAsync(newPointer, buf->ptr, copyBytes,
                           hipMemcpyDeviceToDevice, work);
    if (error != hipSuccess) {
      const hipError_t freeAsyncErr = hipFreeAsync(newPointer, work);
      if (freeAsyncErr != hipSuccess) {
        return statusFor(freeAsyncErr);
      }
      return statusFor(error);
    }

    error = hipFreeAsync(buf->ptr, work);
    if (error != hipSuccess) {
      const hipError_t freeAsyncErr = hipFreeAsync(newPointer, work);

      if (freeAsyncErr != hipSuccess) {
        return statusFor(freeAsyncErr);
      }
      return statusFor(error);
    }

    buf->ptr = newPointer;
    buf->bytes = new_bytes;
    buf->streamOrigin = work;
    return HIP_OK;
  } else {
    hipError_t error = hipMalloc(&newPointer, new_bytes);
    if (error != hipSuccess) {
      return statusFor(error);
    }
    error = hipMemcpy(newPointer, buf->ptr, copyBytes, hipMemcpyDeviceToDevice);
    if (error != hipSuccess) {
      (void)hipFree(newPointer);
      return statusFor(error);
    }
    error = hipFree(buf->ptr);
    if (error != hipSuccess) {
      (void)hipFree(newPointer);
      return statusFor(error);
    }
  }

  buf->ptr = newPointer;
  buf->bytes = new_bytes;
  return HIP_OK;
}

/**
 * @brief Trims reclaimable backing from one device pool.
 *
 * @details
 * No-op when the device does not support native memory pools.
 * Otherwise the device engine stream drains first, because a free
 * still queued there keeps its block counted as used, and the pool
 * trims to at least @p min_bytes_to_keep resident bytes via
 * @c hipMemPoolTrimTo.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[in]  min_bytes_to_keep Minimum resident bytes to retain.
 *
 * @return Status; @c novaSuccess when the pool is unsupported or trimmed.
 */
novaStatus_t HipDevicePool::emptyCache(std::int32_t device_id,
                                       std::size_t min_bytes_to_keep) {
  return withPool(device_id, nullptr, true, [&](HipDevicePool &pool) {
    if (!pool.native_pool_supported_) {
      return HIP_OK;
    }
    const novaStatus_t drainStatus = HipEngineStream::drain(device_id);
    if (drainStatus.err != novaSuccess) {
      return drainStatus;
    }
    return statusFor(hipMemPoolTrimTo(
        static_cast<hipMemPool_t>(pool.native_pool_), min_bytes_to_keep));
  });
}

/**
 * @brief Reads runtime statistics for one device pool.
 *
 * @details
 * Zeroes @p out and queries the pool attributes via
 * @c hipMemPoolGetAttribute.  When the device does not support native
 * memory pools the call reports @c novaMemoryPoolsNotNativelySupported.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[out] out       Receives the per-device pool counters.
 *
 * @return Status; @c novaSuccess on success;
 *         @c novaMemoryPoolsNotNativelySupported when pools are unsupported.
 */
novaStatus_t HipDevicePool::stats(std::int32_t device_id,
                                  DeviceMemoryPoolStats *out) {
  if (out == nullptr) {
    return statusFor(novaInvalidPointer);
  }
  *out = DeviceMemoryPoolStats{};
  return withPool(device_id, nullptr, true, [&](HipDevicePool &pool) {
    out->device_id = device_id;
    out->native_pool_supported = pool.native_pool_supported_;
    if (!pool.native_pool_supported_) {
      return statusFor(novaMemoryPoolsNotNativelySupported);
    }

    hipMemPool_t nativePool = static_cast<hipMemPool_t>(pool.native_pool_);
    std::uint64_t reservedCurrent = 0;
    std::uint64_t reservedHigh = 0;
    std::uint64_t usedCurrent = 0;
    std::uint64_t usedHigh = 0;
    std::uint64_t releaseThreshold = 0;
    hipError_t error = hipMemPoolGetAttribute(
        nativePool, hipMemPoolAttrReservedMemCurrent, &reservedCurrent);
    if (error == hipSuccess) {
      error = hipMemPoolGetAttribute(nativePool, hipMemPoolAttrReservedMemHigh,
                                     &reservedHigh);
    }
    if (error == hipSuccess) {
      error = hipMemPoolGetAttribute(nativePool, hipMemPoolAttrUsedMemCurrent,
                                     &usedCurrent);
    }
    if (error == hipSuccess) {
      error = hipMemPoolGetAttribute(nativePool, hipMemPoolAttrUsedMemHigh,
                                     &usedHigh);
    }
    if (error == hipSuccess) {
      error = hipMemPoolGetAttribute(nativePool, hipMemPoolAttrReleaseThreshold,
                                     &releaseThreshold);
    }
    if (error != hipSuccess) {
      return statusFor(error);
    }

    out->reserved_current = reservedCurrent;
    out->reserved_high = reservedHigh;
    out->used_current = usedCurrent;
    out->used_high = usedHigh;
    out->release_threshold = releaseThreshold;
    return HIP_OK;
  });
}

/**
 * @brief Sets the release threshold for one device pool.
 *
 * @details
 * When the device does not support native memory pools the call reports
 * @c novaMemoryPoolsNotNativelySupported.  Otherwise it sets
 * @c hipMemPoolAttrReleaseThreshold on the pool.
 *
 * @param[in]  device_id Runtime device ordinal owning the pool.
 * @param[in]  bytes     New release threshold in bytes.
 *
 * @return Status; @c novaSuccess on success;
 *         @c novaMemoryPoolsNotNativelySupported when pools are unsupported.
 */
novaStatus_t HipDevicePool::setReleaseThreshold(std::int32_t device_id,
                                                std::uint64_t bytes) {
  return withPool(device_id, nullptr, true, [&](HipDevicePool &pool) {
    if (!pool.native_pool_supported_) {
      return statusFor(novaMemoryPoolsNotNativelySupported);
    }
    std::uint64_t threshold = bytes;
    const hipError_t error =
        hipMemPoolSetAttribute(static_cast<hipMemPool_t>(pool.native_pool_),
                               hipMemPoolAttrReleaseThreshold, &threshold);
    if (error != hipSuccess) {
      return statusFor(error);
    }
    return HIP_OK;
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
novaStatus_t HipDevicePool::destroy(std::int32_t device_id) {
  return withPool(device_id, nullptr, true, [&](HipDevicePool &pool) {
    if (!pool.native_pool_supported_) {
      return HIP_OK;
    }

    const novaStatus_t drainStatus = HipEngineStream::drain(device_id);
    if (drainStatus.err != novaSuccess) {
      return drainStatus;
    }

    std::uint64_t usedCurrent = 0;
    const hipError_t statsError =
        hipMemPoolGetAttribute(static_cast<hipMemPool_t>(pool.native_pool_),
                               hipMemPoolAttrUsedMemCurrent, &usedCurrent);
    if (statsError != hipSuccess) {
      return statusFor(statsError);
    }
    if (usedCurrent != 0) {
      return statusFor(novaResourceBusy);
    }

    const hipError_t error =
        hipMemPoolDestroy(static_cast<hipMemPool_t>(pool.native_pool_));
    if (error != hipSuccess) {
      return statusFor(error);
    }
    pool.native_pool_ = nullptr;
    pool.state_ = State::Destroyed;
    return HIP_OK;
  });
}

#else // !__has_include(<hip/hip_runtime_api.h>)

namespace {
novaStatus_t hipPoolUnavailable() {
  return {.err = novaBackendNotCompiled,
          .message = nova_get_error_msg(novaBackendNotCompiled, nullptr)};
}
} // namespace

novaStatus_t HipDevicePool::currentDeviceId(std::int32_t *) {
  return hipPoolUnavailable();
}
novaStatus_t HipDevicePool::reserve(std::int32_t, std::size_t, bool,
                                    hipBuffer_t *, hipStream_t) {
  return hipPoolUnavailable();
}
novaStatus_t HipDevicePool::release(std::int32_t, hipBuffer_t *, hipStream_t) {
  return hipPoolUnavailable();
}
novaStatus_t HipDevicePool::resize(std::int32_t, hipBuffer_t *, std::size_t,
                                   hipStream_t) {
  return hipPoolUnavailable();
}
novaStatus_t HipDevicePool::emptyCache(std::int32_t, std::size_t) {
  return hipPoolUnavailable();
}
novaStatus_t HipDevicePool::stats(std::int32_t, DeviceMemoryPoolStats *) {
  return hipPoolUnavailable();
}
novaStatus_t HipDevicePool::setReleaseThreshold(std::int32_t, std::uint64_t) {
  return hipPoolUnavailable();
}
novaStatus_t HipDevicePool::destroy(std::int32_t) {
  return hipPoolUnavailable();
}

#endif
#endif
