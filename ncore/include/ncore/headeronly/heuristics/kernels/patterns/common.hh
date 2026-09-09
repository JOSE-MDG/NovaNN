/**
 * @file common.hh
 * @brief Shared vocabulary for kernel launch heuristics.
 *
 * @details
 * Root types for the whole subsystem: launch descriptor, pattern tag,
 * cached device/kernel/launch facts, the abstract parameter base, and
 * the shared constants. All members are plain
 * arithmetic so results are reproducible across backends.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ncore::heuristics::kernels {

/**
 * @struct Dim3
 * @brief Three-dimensional extent mapping 1:1 to `dim3`.
 */
struct Dim3 {
  uint32_t x = 1; ///< Size along x.
  uint32_t y = 1; ///< Size along y.
  uint32_t z = 1; ///< Size along z.
};

/**
 * @struct LaunchConfig
 * @brief Grid/block pair for one kernel launch.
 *
 * @details
 * Every kernel launched through this descriptor uses a grid-stride
 * loop, so any grid size stays correct and only performance varies.
 * All-ones is the degraded floor, never an error.
 */
struct LaunchConfig {
  Dim3 threads; ///< Threads per block (`blockDim`).
  Dim3 blocks;  ///< Number of blocks (`gridDim`).
};

/**
 * @enum ExecutionPattern
 * @brief Kernel memory/compute pattern selecting the heuristic.
 */
enum class ExecutionPattern : uint8_t {
  ElementWise,          ///< Flat streaming maps.
  Reduction,            ///< Axis reduction over a batch.
  LayoutTransformation, ///< Transpose, permute, contiguous-making.
  ParallelScan,         ///< Prefix scans over an axis times batch.
  SortAndHistogram,     ///< Radix sort and histogram counting.
  SpatialNeighborhood,  ///< Stencil, convolution, pooling.
  GEMM,                 ///< Dense matrix multiplication.
  GatherScatter,        ///< Indirect indexed access.
  FusedOperator         ///< Multi-pattern composition in one kernel.
};

/// Fixed block size for the cheap path: 8 CUDA warps, 4 HIP wavefronts.
inline constexpr uint32_t FAST_THREADS = 256;
/// Fixed elements per thread for the cheap path.
inline constexpr uint32_t FAST_ELEMS_PER_THREAD = 4;
/// Grid cap in blocks per SM for the cheap path.
inline constexpr uint32_t FAST_BLOCKS_PER_SM = 8;
/// Factor between estimated kernel time and launch overhead below
/// which analysis is skipped.
inline constexpr uint32_t ROUTE_LAUNCH_FACTOR = 10;
/// Launch overhead when unmeasured, in nanoseconds.
inline constexpr uint64_t DEFAULT_LAUNCH_OVERHEAD_NS = 5000;
/// Bandwidth efficiency with proven aligned-contiguous access.
inline constexpr float ROUTE_ETA_ALIGNED = 0.7f;
/// Bandwidth efficiency otherwise.
inline constexpr float ROUTE_ETA_SCALAR = 0.4f;
/// Blind wavefront fallback for unknown runtime caps.
inline constexpr uint32_t FALLBACK_WARP_SIZE = 32;

/**
 * @struct DeviceCaps
 * @brief Cached hardware facts, built once per process.
 *
 * @details
 * Populated by `makeDeviceCaps()` from the cached device properties;
 * never queried per launch. Numeric-only so it compiles everywhere.
 */
struct DeviceCaps {
  uint32_t warpSize = 0;                 ///< Threads per warp/wavefront.
  uint32_t maxThreadsPerBlockDevice = 0; ///< Device block-size limit.
  uint32_t maxThreadsPerSM = 0;          ///< Resident threads per SM.
  uint32_t maxBlocksPerSM = 0;           ///< Resident blocks per SM.
  uint32_t smCount = 0;                  ///< SM (CUDA) / CU (HIP) count.
  size_t smemPerBlock = 0;               ///< Shared memory per block, bytes.
  size_t smemPerSM = 0;                  ///< Shared memory per SM, bytes.
  bool l1SmemSplitConfigurable = false;  ///< Shared/L1 budget is splittable.
  uint32_t regsPerSM = 0;                ///< Registers per SM.
  uint32_t regsAllocGranularity =
      0; ///< Per-thread quantum (8 on SM75+/CDNA, 0 disables).
  uint32_t maxGridX = 0, maxGridY = 0, maxGridZ = 0; ///< Grid limits per axis.
  uint32_t archVersion = 0;                   ///< Packed arch generation.
  uint64_t memBandwidthBytesPerSec = 0;       ///< Theoretical bandwidth.
  uint64_t peakFlops = 0;                     ///< Theoretical throughput.
  size_t l2Size = 0;                          ///< L2 capacity, bytes.
  size_t l2PersistentReservable = 0;          ///< Reservable persistent share.
  size_t l2MaxWindow = 0;                     ///< Max persistent policy window.
  bool asyncCopyEngines = false;              ///< HW global/shared async copy.
  bool clusterSharedMem = false;              ///< Multi-block shared access.
  bool crossBlockAtomicsWithOrdering = false; ///< Ordered cross-block atomics.
};

/**
 * @struct KernelAttrs
 * @brief Compiled-kernel facts from the toolchain.
 */
struct KernelAttrs {
  uint32_t regsPerThread = 0;            ///< Registers per thread.
  size_t staticSmemBytes = 0;            ///< Static shared memory, bytes.
  uint32_t maxThreadsPerBlockKernel = 0; ///< Kernel block-size limit.
};

/**
 * @struct LaunchContext
 * @brief Per-launch facts that are not hardware properties.
 */
struct LaunchContext {
  bool inGraphCapture = false;     ///< Inside a captured launch sequence.
  uint64_t expectedDurationNs = 0; ///< Expected kernel time, 0 if unknown.
  uint64_t launchOverheadNs = 0;   ///< Measured overhead, 0 for default.
  float expectedDivergence = 0.0F; ///< Expected intra-warp divergence.
};

/**
 * @struct LaunchParamsBase
 * @brief Abstract root of every pattern parameter struct.
 *
 * @details
 * Pass derived objects by const reference only. The derived type
 * already encodes the pattern through `pattern()`, so no separate
 * pattern argument travels alongside.
 */
struct LaunchParamsBase {
  DeviceCaps device;     ///< Cached hardware facts.
  KernelAttrs kernel;    ///< Compiled-kernel facts.
  LaunchContext context; ///< Per-launch facts.

  virtual ~LaunchParamsBase() = default;

  // Explicit specials: the virtual destructor would otherwise deprecate
  // the implicit copies, and declaring them would suppress the default
  // constructor derived structs rely on.
  LaunchParamsBase() = default;
  LaunchParamsBase(const LaunchParamsBase &) = default;
  LaunchParamsBase &operator=(const LaunchParamsBase &) = default;

  /**
   * @brief Pattern tag matching the dynamic derived type.
   * @return The pattern this parameter set describes.
   */
  [[nodiscard]] virtual ExecutionPattern pattern() const noexcept = 0;
};

namespace detail {

/// Hash-combine fold for table keys.
inline uint64_t foldU64(uint64_t h, uint64_t v) noexcept {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
  return h;
}

/// 32-bit dtype tag folding both halves; truncation keeps the low
/// bits of the combined fold, never one side alone.
inline uint32_t foldDtypeTag(uint64_t inTag, uint64_t outTag) noexcept {
  const uint64_t h = foldU64(inTag * 0x9E3779B1ULL, outTag);
  return static_cast<uint32_t>(h ^ (h >> 32U));
}

/// Wavefront width from runtime caps, blind fallback otherwise.
inline uint32_t effectiveWarp(const DeviceCaps &d) noexcept {
  return d.warpSize != 0 ? d.warpSize : FALLBACK_WARP_SIZE;
}

/// Ceiling division; zero divisor yields the dividend, huge
/// dividends saturate instead of wrapping the addend.
constexpr uint64_t ceilDiv(uint64_t a, uint64_t b) noexcept {
  return b == 0 ? a : (a / b) + (a % b != 0 ? 1U : 0U);
}

/// Saturating multiply; saturates instead of wrapping.
inline uint64_t satMul(uint64_t a, uint64_t b) noexcept {
  if (a == 0 || b == 0) {
    return 0;
  }
  if (a > ~0ULL / b) {
    return ~0ULL;
  }
  return a * b;
}

/// Saturating addition; saturates instead of wrapping.
inline uint64_t satAdd(uint64_t a, uint64_t b) noexcept {
  if (a > ~0ULL - b) {
    return ~0ULL;
  }
  return a + b;
}

/// Next power of two at or above `v`; 0 and 1 yield 1, inputs past
/// 2^63 saturate instead of wrapping to zero.
constexpr uint64_t nextPow2(uint64_t v) noexcept {
  if (v <= 1) {
    return 1;
  }
  if (v > (1ULL << 63U)) {
    return ~0ULL;
  }
  v -= 1;
  v |= v >> 1U;
  v |= v >> 2U;
  v |= v >> 4U;
  v |= v >> 8U;
  v |= v >> 16U;
  v |= v >> 32U;
  return v + 1;
}

/**
 * @brief Thread count honoring the register quantum.
 *
 * @details
 * Starts from the fixed ceiling and steps down whole warps while the
 * warp-quantized register demand exceeds the SM file. Pure cached
 * arithmetic; unknown files (`regsPerSM == 0`) skip the quantum walk.
 * Sub-warp ceilings return the ceiling itself for the clamp to enforce.
 */
inline uint32_t fastThreads(const DeviceCaps &d,
                            const KernelAttrs &k) noexcept {
  uint32_t cap = FAST_THREADS;
  if (d.maxThreadsPerBlockDevice != 0 && d.maxThreadsPerBlockDevice < cap) {
    cap = d.maxThreadsPerBlockDevice;
  }
  if (k.maxThreadsPerBlockKernel != 0 && k.maxThreadsPerBlockKernel < cap) {
    cap = k.maxThreadsPerBlockKernel;
  }
  const uint32_t warp = effectiveWarp(d);
  uint32_t warps = cap / warp;
  if (warps == 0) {
    return cap != 0 ? cap : warp;
  }
  if (d.regsPerSM != 0 && k.regsPerThread != 0) {
    const uint32_t gran =
        d.regsAllocGranularity != 0 ? d.regsAllocGranularity : 1U;
    const uint64_t perWarp =
        ceilDiv(static_cast<uint64_t>(k.regsPerThread), gran) * gran * warp;
    while (warps > 1 && static_cast<uint64_t>(warps) * perWarp > d.regsPerSM) {
      --warps;
    }
  }
  return warps * warp;
}

/**
 * @brief Shared clamp pass applied to every resolver result.
 *
 * @details
 * One-dimensional thread counts round down to whole warps; multi-dim
 * blocks shrink outer axes first and collapse to 1-D past the cap;
 * every axis clamps to its device limit; zero or one total threads
 * degrade to the all-ones floor.
 */
inline LaunchConfig clampConfig(LaunchConfig c, const DeviceCaps &d,
                                const KernelAttrs &k) noexcept {
  uint64_t total = satMul(
      satMul(static_cast<uint64_t>(c.threads.x), c.threads.y), c.threads.z);
  if (total <= 1) {
    return LaunchConfig{};
  }
  const uint32_t warp = effectiveWarp(d);
  if (c.threads.y == 1 && c.threads.z == 1 && total % warp != 0) {
    total = (total / warp) * warp;
    if (total == 0) {
      total = warp;
    }
    c.threads.x = static_cast<uint32_t>(total);
  }
  uint32_t cap = d.maxThreadsPerBlockDevice;
  if (k.maxThreadsPerBlockKernel != 0 &&
      (cap == 0 || k.maxThreadsPerBlockKernel < cap)) {
    cap = k.maxThreadsPerBlockKernel;
  }
  if (cap != 0 && total > cap) {
    if (c.threads.y == 1 && c.threads.z == 1) {
      total = (cap / warp) * warp;
      if (total == 0) {
        total = warp <= cap ? warp : cap;
      }
      c.threads.x = static_cast<uint32_t>(total);
    } else {
      // Multi-dim blocks keep tile width; outer axes shrink first and
      // a lone over-wide column collapses to warp-floored 1-D.
      uint64_t x = c.threads.x, y = c.threads.y, z = c.threads.z;
      if (z > 1) {
        z = cap / satMul(x, y);
        if (z == 0) {
          z = 1;
        }
      }
      if (satMul(satMul(x, y), z) > cap && y > 1) {
        y = cap / satMul(x, z);
        if (y == 0) {
          y = 1;
        }
      }
      if (satMul(satMul(x, y), z) > cap) {
        const uint64_t t = (cap / warp) * warp;
        x = t != 0 ? t : (warp <= cap ? warp : cap);
        y = 1;
        z = 1;
      }
      c.threads.x = static_cast<uint32_t>(x);
      c.threads.y = static_cast<uint32_t>(y);
      c.threads.z = static_cast<uint32_t>(z);
      total = satMul(satMul(x, y), z);
    }
  }
  if (d.maxGridX != 0 && c.blocks.x > d.maxGridX) {
    c.blocks.x = d.maxGridX;
  }
  if (d.maxGridY != 0 && c.blocks.y > d.maxGridY) {
    c.blocks.y = d.maxGridY;
  }
  if (d.maxGridZ != 0 && c.blocks.z > d.maxGridZ) {
    c.blocks.z = d.maxGridZ;
  }
  if (c.blocks.x == 0 || c.blocks.y == 0 || c.blocks.z == 0 ||
      c.threads.x == 0) {
    return LaunchConfig{};
  }
  return c;
}

} // namespace detail

/**
 * @brief Proven pointer alignment capped at 16 bytes.
 *
 * @details
 * Largest power of two dividing the address; null yields 1. Lets
 * callers prove vectorization width with two integer ops.
 */
inline uint32_t pointerAlign(const void *p) noexcept {
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  if (addr == 0) {
    return 1;
  }
  uintptr_t low = addr & (~addr + 1U);
  low = std::min<uintptr_t>(low, 16);
  return static_cast<uint32_t>(low);
}

/**
 * @brief Build caps from either backend's detected properties.
 *
 * @details
 * Copies the four numeric fields both property structs share; every
 * other cap stays zero (unknown) so the router and clamps skip what
 * they cannot prove. Host-only glue for kernel launch sites.
 */
template <typename DetectedProps>
inline DeviceCaps capsFromDetected(const DetectedProps &p) noexcept {
  DeviceCaps d{};
  d.warpSize = static_cast<uint32_t>(p.warpSize);
  d.maxThreadsPerBlockDevice = static_cast<uint32_t>(p.maxThreadsPerBlock);
  d.maxThreadsPerSM = static_cast<uint32_t>(p.maxThreadsPerMultiProcessor);
  d.smCount = static_cast<uint32_t>(p.multiProcessorCount);
  return d;
}

} // namespace ncore::heuristics::kernels
