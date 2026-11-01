/**
 * @file HipMemoryPool.hpp
 * @brief Per-device HIP memory-pool operations.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#include "HipAllocator.hpp"

struct DeviceMemoryPoolStats;

/**
 * @brief Owns the lazy HIP memory pool for one runtime device.
 *
 * @details
 * One instance per device slot in @ref pools_. Initialization is
 * lazy and guarded by @ref mutex_; a destroyed slot rejects later
 * operations. Native pools use @c hipMallocFromPoolAsync; devices
 * without pool support fall back to @c hipMalloc with no caching.
 * A null stream selects the device engine stream and returns without
 * synchronizing. HIP pools are Linux-only and marked beta by ROCm.
 */
class HipDevicePool final {
public:
  static novaStatus_t currentDeviceId(std::int32_t *device_id);
  static novaStatus_t reserve(std::int32_t device_id, std::size_t bytes,
                              bool pinned, hipBuffer_t *out,
                              hipStream_t stream);
  static novaStatus_t release(std::int32_t device_id, hipBuffer_t *buf,
                              hipStream_t stream);
  static novaStatus_t resize(std::int32_t device_id, hipBuffer_t *buf,
                             std::size_t new_bytes, hipStream_t stream);
  static novaStatus_t emptyCache(std::int32_t device_id,
                                 std::size_t min_bytes_to_keep);
  static novaStatus_t stats(std::int32_t device_id, DeviceMemoryPoolStats *out);
  static novaStatus_t setReleaseThreshold(std::int32_t device_id,
                                          std::uint64_t bytes);
  static novaStatus_t destroy(std::int32_t device_id);

private:
  enum class State : std::uint8_t { Uninitialized, Ready, Destroyed };

  template <typename Operation>
  static novaStatus_t withPool(std::int32_t device_id, hipStream_t stream,
                               bool initialize_pool, Operation &&operation);

  novaStatus_t initialize(std::int32_t device_id);
  novaStatus_t reserveBuffer(std::size_t bytes, bool pinned, hipBuffer_t *out,
                             hipStream_t stream);
  novaStatus_t releaseBuffer(hipBuffer_t *buf, hipStream_t stream);
  novaStatus_t resizeBuffer(hipBuffer_t *buf, std::size_t new_bytes,
                            hipStream_t stream);

  static std::array<HipDevicePool, NCORE_COMPILE_TIME_MAX_GPUS> pools_;

  std::mutex mutex_;                   ///< Guards pool state transitions.
  State state_ = State::Uninitialized; ///< Current lifecycle state.
  bool native_pool_supported_ = false; ///< @c true when the device exposes
                                       ///< a native pool.
  void *native_pool_ = nullptr;        ///< Opaque @c hipMemPool_t handle.
};
