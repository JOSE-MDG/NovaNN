/**
 * @file RngDispatchImpl.cpp
 * @brief Runtime dispatch for RNG fill kernels.
 *
 * @details
 * Implements @ref launchRngFillU64() and @ref launchRngFillU32() from
 * @ref rng.h over @ref ncore::dispatch::launch(), with one static
 * table per width. Same shape as @c ContiguousDispatchImpl.cpp.
 *
 * @see rng.h                 Public interface for this module.
 * @see KernelDispatcher.hpp  Shared backend-resolution flow.
 */

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/native/kernels/random/rng.h>

#include "utils/KernelDispatcher.hpp"

#ifdef NOVA_HAS_CUDA
#include "cuda/kernels/random/RngFillKernel.h"
#include "cuda/memory/CudaEngineStream.hpp"
#endif
#ifdef NOVA_HAS_HIP
#include "hip/kernels/random/RngFillKernel.h"
#include "hip/memory/HipEngineStream.hpp"
#endif

/**
 * @brief Trailing caller-stream parameter, aliased to the backend handle.
 *
 * @details
 * The dispatch tables carry one function pointer type for both
 * backends, so the stream parameter uses the handle of the compiled
 * backend.  The handle comes from the runtime when it is reachable and
 * from the opaque fallback otherwise, and both denote the same pointer
 * type.
 */
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
using rng_stream_t = cudaStream_t;
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
using rng_stream_t = hipStream_t;
#else
using rng_stream_t = void *;
#endif

namespace {

/**
 * @brief Function pointer types for backend-specific RNG fills.
 */
using kernel_u64_t = novaStatus_t (*)(const uint32_t *, const uint32_t *,
                                      uint64_t *, size_t, rng_stream_t);
using kernel_u32_t = novaStatus_t (*)(const uint32_t *, const uint32_t *,
                                      uint32_t *, size_t, rng_stream_t);

/**
 * @brief Static dispatch tables mapping device kinds to RNG kernels.
 *
 * @details
 * Populated at compile time. Entries for unavailable backends are
 * set to @c nullptr. Consumed by @ref ncore::dispatch::launch().
 */
const std::unordered_map<DeviceKind, kernel_u64_t> KERNEL_DISPATCHER_U64 = {
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
    {CUDA_DEVICE, launchCudaRngFillU64},
    {HIP_DEVICE, nullptr},
    {NULL_DEVICE, nullptr}
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
    {CUDA_DEVICE, nullptr},
    {HIP_DEVICE, launchHipRngFillU64},
    {NULL_DEVICE, nullptr},
#else
    {CUDA_DEVICE, nullptr}, {HIP_DEVICE, nullptr}, {NULL_DEVICE, nullptr}
#endif
};

const std::unordered_map<DeviceKind, kernel_u32_t> KERNEL_DISPATCHER_U32 = {
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
    {CUDA_DEVICE, launchCudaRngFillU32},
    {HIP_DEVICE, nullptr},
    {NULL_DEVICE, nullptr}
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
    {CUDA_DEVICE, nullptr},
    {HIP_DEVICE, launchHipRngFillU32},
    {NULL_DEVICE, nullptr},
#else
    {CUDA_DEVICE, nullptr}, {HIP_DEVICE, nullptr}, {NULL_DEVICE, nullptr}
#endif
};

} // namespace

/** Launch an RNG fill on the detected device. Resolution and errors
 * live in @ref ncore::dispatch::launch(). */
extern "C" novaStatus_t launchRngFillU64(const uint32_t key[2],
                                         const uint32_t base_ctr[4],
                                         uint64_t *dst, size_t n) {
  rng_stream_t stream = nullptr;
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
  const novaStatus_t status = CudaEngineStream::current(&stream);
  if (status.err != novaSuccess) {
    return status;
  }
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
  const novaStatus_t status = HipEngineStream::current(&stream);
  if (status.err != novaSuccess) {
    return status;
  }
#endif
  return ncore::dispatch::launch(KERNEL_DISPATCHER_U64, key, base_ctr, dst, n,
                                 stream);
}

extern "C" novaStatus_t launchRngFillU32(const uint32_t key[2],
                                         const uint32_t base_ctr[4],
                                         uint32_t *dst, size_t n) {
  rng_stream_t stream = nullptr;
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
  const novaStatus_t status = CudaEngineStream::current(&stream);
  if (status.err != novaSuccess) {
    return status;
  }
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
  const novaStatus_t status = HipEngineStream::current(&stream);
  if (status.err != novaSuccess) {
    return status;
  }
#endif
  return ncore::dispatch::launch(KERNEL_DISPATCHER_U32, key, base_ctr, dst, n,
                                 stream);
}
