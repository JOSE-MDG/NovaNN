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
#include <ncore/native/kernels/rng.h>

#include "utils/KernelDispatcher.hpp"

/** Trailing caller-stream parameter. Tag-pointer aliases matching each
 * backend runtime's own typedef: this TU compiles without either
 * runtime's include paths. */
#if defined(NOVA_HAS_CUDA) && !defined(NOVA_HAS_HIP)
struct CUstream_st;
using rng_stream_t = CUstream_st *;
#elif defined(NOVA_HAS_HIP) && !defined(NOVA_HAS_CUDA)
struct ihipStream_t;
using rng_stream_t = ihipStream_t *;
#else
using rng_stream_t = void *;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NOVA_HAS_CUDA
/** CUDA backend entry points, defined in @c RngFillKernel.cu. */
extern novaStatus_t launchCudaRngFillU64(const uint32_t key[2],
                                         const uint32_t base_ctr[4],
                                         uint64_t *dst, size_t n,
                                         rng_stream_t caller_stream);
extern novaStatus_t launchCudaRngFillU32(const uint32_t key[2],
                                         const uint32_t base_ctr[4],
                                         uint32_t *dst, size_t n,
                                         rng_stream_t caller_stream);
#endif

#ifdef NOVA_HAS_HIP
/** HIP backend entry points, defined in @c RngFillKernel.hip. */
extern novaStatus_t launchHipRngFillU64(const uint32_t key[2],
                                        const uint32_t base_ctr[4],
                                        uint64_t *dst, size_t n,
                                        rng_stream_t caller_stream);
extern novaStatus_t launchHipRngFillU32(const uint32_t key[2],
                                        const uint32_t base_ctr[4],
                                        uint32_t *dst, size_t n,
                                        rng_stream_t caller_stream);
#endif

#ifdef __cplusplus
}
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
  return ncore::dispatch::launch(KERNEL_DISPATCHER_U64, key, base_ctr, dst, n,
                                 nullptr);
}

extern "C" novaStatus_t launchRngFillU32(const uint32_t key[2],
                                         const uint32_t base_ctr[4],
                                         uint32_t *dst, size_t n) {
  return ncore::dispatch::launch(KERNEL_DISPATCHER_U32, key, base_ctr, dst, n,
                                 nullptr);
}
