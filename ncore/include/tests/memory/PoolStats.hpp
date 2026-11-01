/**
 * @file PoolStats.hpp
 * @brief Pool statistics probe shared by the device-pool suites.
 */

#pragma once

#include <cstdint>

#include <ncore/core/status.h>
#include <ncore/memory/ffi.hpp>

#include <tests/memory/DeviceBuffer.hpp>

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

namespace tests::memory {

/**
 * @brief Reads the pool statistics of the device this build reserves on.
 */
inline novaStatus_t getPoolStats(std::int32_t device_id,
                                 DeviceMemoryPoolStats *stats) {
  return devicePoolStats(kTestDeviceBackend, device_id, stats);
}

} // namespace tests::memory

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP
