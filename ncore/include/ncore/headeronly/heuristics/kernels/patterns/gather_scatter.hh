/**
 * @file gather_scatter.hh
 * @brief Launch heuristic for indirect indexed access.
 */

#pragma once

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum IndexSpread
 * @brief Expected index locality, Unknown stays neutral.
 */
enum class IndexSpread : uint8_t { Clustered, Scattered, Unknown };

/**
 * @enum WarpDedup
 * @brief Intra-warp duplicate coordination, Auto decides.
 */
enum class WarpDedup : uint8_t { Auto, On, Off };

/**
 * @enum IndexPresort
 * @brief Lightweight index pre-sort, Auto decides.
 */
enum class IndexPresort : uint8_t { Auto, On, Off };

/// Largest segment Fast issues directly, in bytes.
inline constexpr uint64_t GATHER_FAST_MAX_SEGMENT_BYTES = 16;

/**
 * @struct IndexDistribution
 * @brief Index locality descriptor.
 */
struct IndexDistribution {
  IndexSpread spread = IndexSpread::Unknown; ///< Grouping expectation.
  float locality01 = 0.5F; ///< 1 clustered on few lines, 0 scattered.
};

/**
 * @struct GatherScatterParams
 * @brief Facts for one indirect launch.
 */
struct GatherScatterParams : LaunchParamsBase {
  uint64_t numIndices = 0;         ///< Elements to read or write.
  uint32_t dataItemSize = 0;       ///< Data bytes per element.
  uint32_t indexItemSize = 0;      ///< Index bytes (4 or 8).
  bool indicesUnique = false;      ///< Duplicates impossible.
  uint64_t segmentSize = 0;        ///< Contiguous elements per index.
  IndexDistribution distribution;  ///< Locality descriptor.
  bool expectIntraWarpDup = false; ///< Same index twice in one warp.
  float readWriteRatio = 0.0F;     ///< Reads per write.
  WarpDedup dedup = WarpDedup::Auto;
  IndexPresort presort = IndexPresort::Auto;

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::GatherScatter;
  }
};

/// Total indexed elements.
inline uint64_t fastParallelWork(const GatherScatterParams &p) noexcept {
  return detail::satMul(p.numIndices, p.segmentSize);
}

/// Bytes moved counting index stream plus data both ways.
inline uint64_t fastTrafficBytes(const GatherScatterParams &p) noexcept {
  const uint64_t elems = detail::satMul(p.numIndices, p.segmentSize);
  return detail::satAdd(
      detail::satMul(p.numIndices, p.indexItemSize),
      detail::satMul(elems, detail::satMul(p.dataItemSize, 2U)));
}

/**
 * @brief True for thread-per-element segments.
 */
inline bool coversGatherScatterFast(const GatherScatterParams &p) noexcept {
  if (p.numIndices == 0) {
    return true;
  }
  return detail::satMul(p.segmentSize, p.dataItemSize) <=
         GATHER_FAST_MAX_SEGMENT_BYTES;
}

/**
 * @brief 1D grid-stride over index elements.
 *
 * @details
 * Direct issue is correct regardless of uniqueness: the op's native
 * atomics serialize true duplicates. Dedup and presort only ever save
 * traffic, so skipping them cannot break results.
 */
inline LaunchConfig
resolveGatherScatterFast(const GatherScatterParams &p) noexcept {
  if (p.numIndices == 0) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t work =
      detail::satMul(p.numIndices, p.segmentSize != 0 ? p.segmentSize : 1U);
  const uint64_t perThread =
      static_cast<uint64_t>(threads) * FAST_ELEMS_PER_THREAD;
  uint64_t blocks = detail::ceilDiv(work, perThread);
  const uint64_t cap =
      static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  LaunchConfig cfg{};
  cfg.threads.x = threads;
  cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

/**
 * @brief Assignment granularity for large segments.
 *
 * @details
 * Medium segments promote to warp-per-segment so lanes stay
 * coalesced; segments past efficient warp reuse promote to
 * block-per-segment. Dedup, presort, and mix tuning belong to kernel
 * selection downstream: geometry here only sizes the launch.
 */
inline LaunchConfig
resolveGatherScatter(const GatherScatterParams &p) noexcept {
  if (p.numIndices == 0 || p.dataItemSize == 0) {
    return LaunchConfig{};
  }
  const uint32_t warp = detail::effectiveWarp(p.device);
  const uint64_t segBytes = detail::satMul(p.segmentSize, p.dataItemSize);
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t totalElems =
      detail::satMul(p.numIndices, p.segmentSize != 0 ? p.segmentSize : 1U);
  const uint64_t cap =
      static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
  uint64_t blocks;
  uint32_t blockThreads;
  if (segBytes > detail::satMul(detail::satMul(static_cast<uint64_t>(warp),
                                               p.dataItemSize),
                                4U) &&
      p.device.smemPerBlock != 0) {
    blockThreads = threads;
    blocks = p.numIndices;
  } else if (segBytes > GATHER_FAST_MAX_SEGMENT_BYTES) {
    blockThreads = warp;
    const uint64_t groups =
        detail::ceilDiv(totalElems, static_cast<uint64_t>(warp) * 4U);
    blocks = groups;
  } else {
    blockThreads = threads;
    blocks = detail::ceilDiv(totalElems, static_cast<uint64_t>(threads) *
                                             FAST_ELEMS_PER_THREAD);
  }
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  LaunchConfig cfg{};
  cfg.threads.x = blockThreads;
  cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

} // namespace ncore::heuristics::kernels
