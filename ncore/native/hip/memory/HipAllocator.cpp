/**
 * @file HipAllocator.cpp
 * @brief HIP allocator entry points.
 */

#include "HipAllocator.hpp"
#include "HipMemoryPool.hpp"

novaStatus_t hipCurrentDeviceId(std::int32_t *device_id) {
  return HipDevicePool::currentDeviceId(device_id);
}

novaStatus_t hipReserve(std::int32_t device_id, std::size_t bytes, bool pinned,
                        hipBuffer_t *out, hipStream_t stream) {
  return HipDevicePool::reserve(device_id, bytes, pinned, out, stream);
}

novaStatus_t hipRelease(std::int32_t device_id, hipBuffer_t *buf,
                        hipStream_t stream) {
  return HipDevicePool::release(device_id, buf, stream);
}

novaStatus_t hipResize(std::int32_t device_id, hipBuffer_t *buf,
                       std::size_t new_bytes, hipStream_t stream) {
  return HipDevicePool::resize(device_id, buf, new_bytes, stream);
}

novaStatus_t hipEmptyCache(std::int32_t device_id,
                           std::size_t min_bytes_to_keep) {
  return HipDevicePool::emptyCache(device_id, min_bytes_to_keep);
}

novaStatus_t hipPoolStats(std::int32_t device_id, DeviceMemoryPoolStats *out) {
  return HipDevicePool::stats(device_id, out);
}

novaStatus_t hipPoolSetReleaseThreshold(std::int32_t device_id,
                                        std::uint64_t bytes) {
  return HipDevicePool::setReleaseThreshold(device_id, bytes);
}

novaStatus_t hipPoolDestroy(std::int32_t device_id) {
  return HipDevicePool::destroy(device_id);
}
