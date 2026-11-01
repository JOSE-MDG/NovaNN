/**
 * @file RngFillKernel.cu
 * @brief CUDA kernels filling device memory with Philox draws.
 *
 * @see launch_config.hh  Shared grid/block resolution.
 * @see RngFillKernel.h   Launcher declarations.
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/heuristics/kernels/launch_config.hh>
#include <ncore/headeronly/macros.h>
#include <ncore/rand/philox.h>

#include "DetectCudaDeviceInfo.hpp"
#include "RngFillKernel.h"

namespace hk = ncore::heuristics::kernels;

namespace {

constexpr uint32_t kPhiloxLanes = 4U;
constexpr uint64_t kU32PerBlock = 4U;
constexpr uint64_t kU64PerBlock = 2U;
constexpr uint32_t kBlockBytes = 16U;
constexpr uint32_t kLaneBits = 32U;

/**
 * @brief Query the kernel attributes for a given device and cache them.
 *
 * @details
 * Compiled facts of one fill kernel. One build can serve devices of
 * several generations whose register counts and block limits
 * differ, so the query is keyed by the current device instead of
 * running once per instantiation. The cache is thread-local with a
 * bounded pool of device slots; devices beyond the pool query per
 * call without aliasing the slots.
 */
template <typename Kernel> hk::KernelAttrs kernelAttrs(Kernel *func) {
  constexpr int kCachedDevices = 8;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
    device = 0;
  }
  if (device >= kCachedDevices) {
    hk::KernelAttrs a{};
    cudaFuncAttributes q{};
    if (cudaFuncGetAttributes(&q, reinterpret_cast<const void *>(func)) ==
        cudaSuccess) {
      a.regsPerThread = static_cast<uint32_t>(q.numRegs);
      a.staticSmemBytes = q.sharedSizeBytes;
      a.maxThreadsPerBlockKernel = static_cast<uint32_t>(q.maxThreadsPerBlock);
    }
    return a;
  }
  const int slot = device;
  thread_local std::array<hk::KernelAttrs, kCachedDevices> facts = {};
  thread_local uint32_t filledMask = 0U;
  const uint32_t bit = 1U << slot;
  if ((filledMask & bit) == 0U) {
    hk::KernelAttrs a{};
    cudaFuncAttributes q{};
    if (cudaFuncGetAttributes(&q, reinterpret_cast<const void *>(func)) ==
        cudaSuccess) {
      a.regsPerThread = static_cast<uint32_t>(q.numRegs);
      a.staticSmemBytes = q.sharedSizeBytes;
      a.maxThreadsPerBlockKernel = static_cast<uint32_t>(q.maxThreadsPerBlock);
    }
    facts[static_cast<size_t>(slot)] = a;
    filledMask |= bit;
  }
  return facts[static_cast<size_t>(slot)];
}

/// Map the pending launch error into status (also resets the sticky error).
inline novaStatus_t checkLaunch() {
  if (cudaGetLastError() != cudaSuccess) {
    return {.err = novaKernelLaunchError,
            .message = nova_get_error_msg(novaKernelLaunchError, nullptr)};
  }
  return novaStatus_t{.err = novaSuccess,
                      .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/// Shared launch geometry for a fill of @p blocks Philox blocks.
novaStatus_t fillGeometry(uint64_t blocks, uint32_t outAlignBytes,
                          hk::KernelAttrs attrs, dim3 *threads, dim3 *grid) {
  novaStatus_t st{};
  auto props = getCudaDeviceProperties(&st);
  if (st.err != novaSuccess) {
    return st;
  }
  hk::ElementWiseParams hp{};
  hp.device = hk::capsFromDetected(props);
  hp.numElements = blocks;
  hp.inputItemSize = 0U;
  hp.outputItemSize = kBlockBytes;
  hp.inputAlignBytes = 0U;
  hp.outputAlignBytes = outAlignBytes;
  hp.packedElemsPerUnit = 1U;
  hp.inputContiguous = false;
  hp.outputContiguous = true;
  hp.numInputs = 0U;
  hp.arithmeticIntensity = 0.0F;
  hp.reuseAfter = false;
  hp.kernel = attrs;
  thread_local hk::FullConfigCache cache;
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(hp, &cache);
  threads->x = cfg.threads.x;
  threads->y = cfg.threads.y;
  threads->z = cfg.threads.z;
  grid->x = cfg.blocks.x;
  grid->y = cfg.blocks.y;
  grid->z = cfg.blocks.z;
  return st;
}

/**
 * @brief Emit four Philox lanes per block over a grid-stride range.
 *
 * @details
 * Each thread owns whole blocks: the counter is derived from the
 * block index plus the launch base, so threads never share state.
 * A 16B-aligned destination takes one 128-bit store per full block;
 * masked tails keep the scalar loop.
 *
 * @param[in]  key0   First stream key word.
 * @param[in]  key1   Second stream key word.
 * @param[in]  c0     Counter word 0 of the launch base.
 * @param[in]  c1     Counter word 1 of the launch base.
 * @param[in]  c2     Counter word 2 of the launch base.
 * @param[in]  c3     Counter word 3 of the launch base.
 * @param[out] out    Destination values in device memory.
 * @param[in]  blocks Block count covering @p n values.
 * @param[in]  n      Values to emit.
 */
__global__ void rngFillU32Blocks(uint32_t key0, uint32_t key1, uint32_t c0,
                                 uint32_t c1, uint32_t c2, uint32_t c3,
                                 uint32_t *restrict out, uint64_t blocks,
                                 size_t n) {
  const uint32_t key[2] = {key0, key1};
  const uint32_t base[4] = {c0, c1, c2, c3};
  const bool aligned = (reinterpret_cast<uintptr_t>(out) % kBlockBytes) == 0U;
  const size_t idx =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x));
  const size_t step =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  for (size_t b = idx; b < static_cast<size_t>(blocks); b += step) {
    uint32_t ctr[4] = {};
    philox_ctr_add(base, static_cast<uint64_t>(b), ctr);
    uint32_t lanes[4] = {};
    philox4x32_10(ctr, key, lanes);
    const size_t first = b * kU32PerBlock;
    if (aligned && first + kPhiloxLanes <= n) {
      const uint4 packed = {lanes[0], lanes[1], lanes[2], lanes[3]};
      *reinterpret_cast<uint4 *>(out + first) = packed;
    } else {
      for (size_t lane = 0; lane < kPhiloxLanes; ++lane) {
        if (first + lane < n) {
          out[first + lane] = lanes[lane];
        }
      }
    }
  }
}

/**
 * @brief Emit two packed 64-bit values per block over a grid-stride range.
 *
 * @details
 * Same ownership and masking as the 32-bit entry point. Each block
 * contributes lanes 0-1 packed little end first, then lanes 2-3.
 * A 16B-aligned destination takes one 128-bit store per full pair;
 * masked tails keep the scalar loop.
 *
 * @param[in]  key0   First stream key word.
 * @param[in]  key1   Second stream key word.
 * @param[in]  c0     Counter word 0 of the launch base.
 * @param[in]  c1     Counter word 1 of the launch base.
 * @param[in]  c2     Counter word 2 of the launch base.
 * @param[in]  c3     Counter word 3 of the launch base.
 * @param[out] out    Destination values in device memory.
 * @param[in]  blocks Block count covering @p n values.
 * @param[in]  n      Values to emit.
 */
__global__ void rngFillU64Blocks(uint32_t key0, uint32_t key1, uint32_t c0,
                                 uint32_t c1, uint32_t c2, uint32_t c3,
                                 uint64_t *restrict out, uint64_t blocks,
                                 size_t n) {
  const uint32_t key[2] = {key0, key1};
  const uint32_t base[4] = {c0, c1, c2, c3};
  const bool aligned = (reinterpret_cast<uintptr_t>(out) % kBlockBytes) == 0U;
  const size_t idx =
      static_cast<size_t>(threadIdx.x) +
      (static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x));
  const size_t step =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  for (size_t b = idx; b < static_cast<size_t>(blocks); b += step) {
    uint32_t ctr[4] = {};
    philox_ctr_add(base, static_cast<uint64_t>(b), ctr);
    uint32_t lanes[4] = {};
    philox4x32_10(ctr, key, lanes);
    const size_t first = b * kU64PerBlock;
    if (aligned && first + kU64PerBlock <= n) {
      const ulonglong2 packed = {
          .x = static_cast<uint64_t>(lanes[0]) |
               (static_cast<uint64_t>(lanes[1]) << kLaneBits),
          .y = static_cast<uint64_t>(lanes[2]) |
               (static_cast<uint64_t>(lanes[3]) << kLaneBits),
      };
      *reinterpret_cast<ulonglong2 *>(out + first) = packed;
    } else {
      out[first] = static_cast<uint64_t>(lanes[0]) |
                   (static_cast<uint64_t>(lanes[1]) << kLaneBits);
      if (first + 1U < n) {
        out[first + 1U] = static_cast<uint64_t>(lanes[2]) |
                          (static_cast<uint64_t>(lanes[3]) << kLaneBits);
      }
    }
  }
}

} // namespace

novaStatus_t launchCudaRngFillU32(const uint32_t key[2],
                                  const uint32_t base_ctr[4], uint32_t *dst,
                                  size_t n, cudaStream_t caller_stream) {
  if (key == nullptr || base_ctr == nullptr || (dst == nullptr && n > 0)) {
    novaStatus_t s{};
    s.err = novaInvalidPointer;
    s.message = "Launcher got null key, counter, or output. Pass valid "
                "buffers.\n";
    return s;
  }
  if (n == 0U) {
    novaStatus_t s{};
    s.err = novaSuccess;
    s.message = nova_get_error_msg(s.err, nullptr);
    return s;
  }
  const uint64_t blocks =
      (static_cast<uint64_t>(n) + (kU32PerBlock - 1U)) / kU32PerBlock;
  if (blocks == 0U) {
    novaStatus_t s{};
    s.err = novaInvalidValue;
    s.message = "Request too large to cover. Split it into smaller fills.\n";
    return s;
  }
  dim3 threads{};
  dim3 grid{};
  novaStatus_t st =
      fillGeometry(blocks, hk::pointerAlign(dst), kernelAttrs(rngFillU32Blocks),
                   &threads, &grid);
  if (st.err != novaSuccess) {
    return st;
  }
  rngFillU32Blocks<<<grid, threads, 0, caller_stream>>>(
      key[0], key[1], base_ctr[0], base_ctr[1], base_ctr[2], base_ctr[3], dst,
      blocks, n);
  return checkLaunch();
}

novaStatus_t launchCudaRngFillU64(const uint32_t key[2],
                                  const uint32_t base_ctr[4], uint64_t *dst,
                                  size_t n, cudaStream_t caller_stream) {
  if (key == nullptr || base_ctr == nullptr || (dst == nullptr && n > 0)) {
    novaStatus_t s{};
    s.err = novaInvalidPointer;
    s.message = "Launcher got null key, counter, or output. Pass valid "
                "buffers.\n";
    return s;
  }
  if (n == 0U) {
    novaStatus_t s{};
    s.err = novaSuccess;
    s.message = nova_get_error_msg(s.err, nullptr);
    return s;
  }
  const uint64_t blocks =
      (static_cast<uint64_t>(n) + (kU64PerBlock - 1U)) / kU64PerBlock;
  if (blocks == 0U) {
    novaStatus_t s{};
    s.err = novaInvalidValue;
    s.message = "Request too large to cover. Split it into smaller fills.\n";
    return s;
  }
  dim3 threads{};
  dim3 grid{};
  novaStatus_t st =
      fillGeometry(blocks, hk::pointerAlign(dst), kernelAttrs(rngFillU64Blocks),
                   &threads, &grid);
  if (st.err != novaSuccess) {
    return st;
  }
  rngFillU64Blocks<<<grid, threads, 0, caller_stream>>>(
      key[0], key[1], base_ctr[0], base_ctr[1], base_ctr[2], base_ctr[3], dst,
      blocks, n);
  return checkLaunch();
}
