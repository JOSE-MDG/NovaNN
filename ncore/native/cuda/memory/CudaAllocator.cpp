/**
 * @file CudaAllocator.cpp
 * @brief CUDA allocator entry points.
 */

#include "CudaAllocator.hpp"
#include "CudaMemoryPool.hpp"

novaStatus_t cudaCurrentDeviceId(std::int32_t *device_id) {
  return CudaDevicePool::currentDeviceId(device_id);
}

novaStatus_t cudaReserve(std::int32_t device_id, std::size_t bytes, bool pinned,
                         cudaBuffer_t *out, cudaStream_t stream) {
  return CudaDevicePool::reserve(device_id, bytes, pinned, out, stream);
}

novaStatus_t cudaRelease(std::int32_t device_id, cudaBuffer_t *buf,
                         cudaStream_t stream) {
  return CudaDevicePool::release(device_id, buf, stream);
}

novaStatus_t cudaResize(std::int32_t device_id, cudaBuffer_t *buf,
                        std::size_t new_bytes, cudaStream_t stream) {
  return CudaDevicePool::resize(device_id, buf, new_bytes, stream);
}

novaStatus_t cudaEmptyCache(std::int32_t device_id,
                            std::size_t min_bytes_to_keep) {
  return CudaDevicePool::emptyCache(device_id, min_bytes_to_keep);
}

novaStatus_t cudaPoolStats(std::int32_t device_id, DeviceMemoryPoolStats *out) {
  return CudaDevicePool::stats(device_id, out);
}

novaStatus_t cudaPoolSetReleaseThreshold(std::int32_t device_id,
                                         std::uint64_t bytes) {
  return CudaDevicePool::setReleaseThreshold(device_id, bytes);
}

novaStatus_t cudaPoolDestroy(std::int32_t device_id) {
  return CudaDevicePool::destroy(device_id);
}
