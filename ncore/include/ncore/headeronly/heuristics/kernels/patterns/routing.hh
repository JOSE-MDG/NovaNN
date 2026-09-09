/**
 * @file routing.hh
 * @brief Entry-point routing between two launch heuristics.

 * @details
 * `resolveTier` runs over cached values only, then the selected
 * resolver runs for the pattern.
 */

#pragma once

#include <array>
#include <cstdint>

#include "common.hh"
#include "element_wise.hh"
#include "fused_operator.hh"
#include "gather_scatter.hh"
#include "gemm.hh"
#include "layout_transform.hh"
#include "parallel_scan.hh"
#include "reduction.hh"
#include "sort_histogram.hh"
#include "spatial_neighborhood.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum HeuristicTier
 * @brief Two internal paths: fixed cheap geometry vs full analysis.
 */
enum class HeuristicTier : uint8_t { Fast, Full };

/**
 * @struct FullCacheKey
 * @brief Identity for one evaluated launch config.
 */
struct FullCacheKey {
  ExecutionPattern pattern = ExecutionPattern::ElementWise;
  uint64_t shapeBucket = 0; ///< Hash-combine fold of quantized shape.
  uint32_t dtypeTag = 0;    ///< Caller-defined dtype pair tag.
  uint32_t archTag = 0;     ///< Caller-defined architecture tag.
};

/**
 * @struct FullCacheEntry
 * @brief One occupied slot of the config table.
 */
struct FullCacheEntry {
  bool occupied = false;
  FullCacheKey key;
  LaunchConfig config;
};

/// Table capacity; linear probe, drops on full.
inline constexpr uint32_t FULL_CACHE_CAPACITY = 64;

/**
 * @struct FullConfigCache
 * @brief Fixed table of recent launch configs.
 *
 * @details
 * Misses recompute inline and optionally store. A wrong entry
 * only costs performance: every stored config already passed the
 * shared clamps, and grid-stride kernels stay correct under any grid.
 */
struct FullConfigCache {
  std::array<FullCacheEntry, FULL_CACHE_CAPACITY> slots;

  /**
   * @brief Probe the table.
   * @return True with the stored config in `out` on hit.
   */
  bool lookup(const FullCacheKey &key, LaunchConfig *out) const noexcept {
    if (out == nullptr) {
      return false;
    }
    uint64_t h = 1469598103934665603ULL;
    h ^= static_cast<uint64_t>(key.pattern);
    h *= 1099511628211ULL;
    h ^= key.shapeBucket;
    h *= 1099511628211ULL;
    h ^= (static_cast<uint64_t>(key.dtypeTag) << 32U) | key.archTag;
    h *= 1099511628211ULL;
    const uint32_t start = static_cast<uint32_t>(h % FULL_CACHE_CAPACITY);
    for (uint32_t i = 0; i < FULL_CACHE_CAPACITY; ++i) {
      const uint32_t idx = (start + i) % FULL_CACHE_CAPACITY;
      const FullCacheEntry &e = slots[idx];
      if (!e.occupied) {
        return false;
      }
      if (e.key.pattern == key.pattern &&
          e.key.shapeBucket == key.shapeBucket &&
          e.key.dtypeTag == key.dtypeTag && e.key.archTag == key.archTag) {
        *out = e.config;
        return true;
      }
    }
    return false;
  }

  /// Store or replace; silently drops when full.
  void store(const FullCacheKey &key, LaunchConfig cfg) noexcept {
    uint64_t h = 1469598103934665603ULL;
    h ^= static_cast<uint64_t>(key.pattern);
    h *= 1099511628211ULL;
    h ^= key.shapeBucket;
    h *= 1099511628211ULL;
    h ^= (static_cast<uint64_t>(key.dtypeTag) << 32U) | key.archTag;
    h *= 1099511628211ULL;
    const uint32_t start = static_cast<uint32_t>(h % FULL_CACHE_CAPACITY);
    for (uint32_t i = 0; i < FULL_CACHE_CAPACITY; ++i) {
      const uint32_t idx = (start + i) % FULL_CACHE_CAPACITY;
      FullCacheEntry &e = slots[idx];
      if (!e.occupied) {
        e.occupied = true;
        e.key = key;
        e.config = cfg;
        return;
      }
      if (e.key.pattern == key.pattern &&
          e.key.shapeBucket == key.shapeBucket &&
          e.key.dtypeTag == key.dtypeTag && e.key.archTag == key.archTag) {
        e.config = cfg;
        return;
      }
    }
  }
};

namespace detail {

/// Effective overhead: captured sequences replay for free.
inline uint64_t routeOverhead(const LaunchParamsBase &p) noexcept {
  if (p.context.inGraphCapture) {
    return 0;
  }
  if (p.context.launchOverheadNs != 0) {
    return p.context.launchOverheadNs;
  }
  return DEFAULT_LAUNCH_OVERHEAD_NS;
}

/// One full resident wave of the machine, saturating.
inline uint64_t routeWaveSize(const DeviceCaps &d) noexcept {
  return satMul(static_cast<uint64_t>(d.smCount), d.maxThreadsPerSM);
}

} // namespace detail

/**
 * @brief O(1) router over cached values only.
 */
inline HeuristicTier resolveTier(const LaunchParamsBase &params) noexcept {
  const ExecutionPattern tag = params.pattern();
  bool covered = false;
  uint64_t work = 0;
  uint64_t bytes = 0;
  uint64_t flops = 0;
  uint64_t tiles = 0;
  bool coalesced = false;
  switch (tag) {
  case ExecutionPattern::ElementWise: {
    const auto &p = static_cast<const ElementWiseParams &>(params);
    covered = coversElementWiseFast(p);
    work = fastParallelWork(p);
    bytes = fastTrafficBytes(p);
    coalesced = fastProvenCoalesced(p);
    break;
  }
  case ExecutionPattern::Reduction: {
    const auto &p = static_cast<const ReductionParams &>(params);
    covered = coversReductionFast(p);
    work = fastParallelWork(p);
    bytes = fastTrafficBytes(p);
    coalesced = fastProvenCoalesced(p);
    break;
  }
  case ExecutionPattern::LayoutTransformation: {
    const auto &p = static_cast<const LayoutTransformParams &>(params);
    covered = coversLayoutTransformFast(p);
    work = fastParallelWork(p);
    bytes = fastTrafficBytes(p);
    break;
  }
  case ExecutionPattern::ParallelScan: {
    const auto &p = static_cast<const ParallelScanParams &>(params);
    covered = coversParallelScanFast(p);
    work = fastParallelWork(p);
    bytes = fastTrafficBytes(p);
    break;
  }
  case ExecutionPattern::SortAndHistogram: {
    const auto &p = static_cast<const SortHistogramParams &>(params);
    covered = coversSortHistogramFast(p);
    work = fastParallelWork(p);
    bytes = fastTrafficBytes(p);
    break;
  }
  case ExecutionPattern::SpatialNeighborhood: {
    const auto &p = static_cast<const SpatialNeighborhoodParams &>(params);
    covered = coversSpatialNeighborhoodFast(p);
    work = fastParallelWork(p);
    bytes = fastTrafficBytes(p);
    tiles = fastOutputTiles(p);
    break;
  }
  case ExecutionPattern::GEMM: {
    const auto &p = static_cast<const GemmParams &>(params);
    covered = coversGemmFast(p);
    work = fastParallelWork(p);
    flops = fastComputeFlops(p);
    tiles = fastOutputTiles(p);
    coalesced = fastProvenCoalesced(p);
    break;
  }
  case ExecutionPattern::GatherScatter: {
    const auto &p = static_cast<const GatherScatterParams &>(params);
    covered = coversGatherScatterFast(p);
    work = fastParallelWork(p);
    bytes = fastTrafficBytes(p);
    break;
  }
  case ExecutionPattern::FusedOperator: {
    const auto &p = static_cast<const FusedOperatorParams &>(params);
    covered = coversFusedOperatorFast(p);
    break;
  }
  }
  if (!covered) {
    return HeuristicTier::Full;
  }
  const DeviceCaps &d = params.device;
  if (d.smCount != 0) {
    if (work != 0 && work < detail::routeWaveSize(d)) {
      return HeuristicTier::Fast;
    }
    if (tiles != 0 && tiles < d.smCount) {
      return HeuristicTier::Fast;
    }
  }
  if (params.context.inGraphCapture) {
    return HeuristicTier::Full;
  }
  const uint64_t ovh = detail::routeOverhead(params);
  if (ovh != 0) {
    if (flops != 0 && d.peakFlops != 0) {
      const double tEst =
          static_cast<double>(flops) / (0.7 * static_cast<double>(d.peakFlops));
      if (tEst < static_cast<double>(ROUTE_LAUNCH_FACTOR * ovh) / 1e9) {
        return HeuristicTier::Fast;
      }
      return HeuristicTier::Full;
    }
    if (bytes != 0 && d.memBandwidthBytesPerSec != 0) {
      const double eta = coalesced ? static_cast<double>(ROUTE_ETA_ALIGNED)
                                   : static_cast<double>(ROUTE_ETA_SCALAR);
      const double tEst =
          static_cast<double>(bytes) /
          (eta * static_cast<double>(d.memBandwidthBytesPerSec));
      if (tEst < static_cast<double>(ROUTE_LAUNCH_FACTOR * ovh) / 1e9) {
        return HeuristicTier::Fast;
      }
      return HeuristicTier::Full;
    }
  }
  return HeuristicTier::Fast;
}

} // namespace ncore::heuristics::kernels
