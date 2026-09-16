/**
 * @file sort_histogram.hh
 * @brief Launch heuristic for radix sorts and histograms.
 */

#pragma once

#include <algorithm>

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum SortHistMode
 * @brief Sort vs histogram interpretation of the launch.
 */
enum class SortHistMode : uint8_t { Sort, Histogram };

/**
 * @enum Privatization
 * @brief Counter privatization level, Auto decides.
 */
enum class Privatization : uint8_t {
  Auto,
  PerThread,
  PerWarp,
  PerBlock,
  GlobalDirect
};

/// Largest sort Fast handles in one launch.
inline constexpr uint64_t SORT_FAST_MAX_ELEMENTS = 65536;

/**
 * @struct DistributionHint
 * @brief Quantified input skew; unknown is `{0, 0.0}`.
 */
struct DistributionHint {
  uint64_t numDistinct = 0;     ///< Distinct values, 0 when unknown.
  float concentration01 = 0.0F; ///< 0 uniform, 1 fully concentrated.
};

/**
 * @struct SortHistogramParams
 * @brief Facts for one sort or histogram launch.
 */
struct SortHistogramParams : LaunchParamsBase {
  uint64_t numElements = 0;               ///< Elements to sort or count.
  uint32_t numBins = 0;                   ///< Bins, 0 for pure sort.
  SortHistMode mode = SortHistMode::Sort; ///< Sort vs histogram.
  int64_t valueMin = 0;                   ///< Input range floor.
  int64_t valueMax = 0;                   ///< Input range ceiling.
  uint8_t bitsPerPass = 0;                ///< Radix width, 0 for Auto (8).
  uint32_t keyItemSize = 0;               ///< Key bytes.
  uint32_t valueItemSize = 0;             ///< Payload bytes, 0 when N/A.
  DistributionHint distribution;          ///< Skew descriptor.
  Privatization privatization = Privatization::Auto;

  ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::SortAndHistogram;
  }
};

namespace detail {

/// Fixed Fast radix width.
constexpr uint8_t sortFastBitsPerPass() noexcept { return 8; }

/// Pass count from key width at the fixed radix width.
inline uint64_t sortPassCount(const SortHistogramParams &p) noexcept {
  const uint8_t bpp =
      p.bitsPerPass != 0 ? p.bitsPerPass : sortFastBitsPerPass();
  if (bpp == 0 || bpp > 8) {
    return 4ULL * (p.keyItemSize != 0 ? p.keyItemSize : 4U);
  }
  const uint64_t keyBits =
      static_cast<uint64_t>(p.keyItemSize != 0 ? p.keyItemSize : 4U) * 8U;
  return ceilDiv(keyBits, bpp);
}

} // namespace detail

/// Total elements counted or moved per pass.
inline uint64_t fastParallelWork(const SortHistogramParams &p) noexcept {
  return detail::satMul(p.numElements, detail::sortPassCount(p));
}

/// Bytes moved counting keys plus payloads per pass.
inline uint64_t fastTrafficBytes(const SortHistogramParams &p) noexcept {
  return detail::satMul(
      detail::satMul(p.numElements,
                     static_cast<uint64_t>(p.keyItemSize) + p.valueItemSize),
      detail::sortPassCount(p));
}

/**
 * @brief True when partials fit one block (bounded sort otherwise).
 */
inline bool coversSortHistogramFast(const SortHistogramParams &p) noexcept {
  if (p.numElements == 0) {
    return true;
  }
  if (p.mode == SortHistMode::Histogram) {
    const uint64_t need = static_cast<uint64_t>(p.numBins) * 4U;
    return p.numBins > 0 &&
           (p.device.smemPerBlock == 0 || need <= p.device.smemPerBlock);
  }
  return p.numElements <= SORT_FAST_MAX_ELEMENTS;
}

/**
 * @brief Shared-privatized partials, one merge.
 */
inline LaunchConfig
resolveSortHistogramFast(const SortHistogramParams &p) noexcept {
  if (p.numElements == 0) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t perThread =
      static_cast<uint64_t>(threads) * FAST_ELEMS_PER_THREAD;
  uint64_t blocks = detail::ceilDiv(p.numElements, perThread);
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
 * @brief Privatization level, radix passes, grid.
 *
 * @details
 * Oversized bin counts fall back to a narrower thread budget with a
 * saturating grid; digit tables that overflow shared memory narrow
 * the radix width. Finer privatization levels and skew-driven choices
 * belong to kernel selection downstream: geometry here only sizes
 * the launch.
 */
inline LaunchConfig
resolveSortHistogram(const SortHistogramParams &p) noexcept {
  if (p.numElements == 0) {
    return LaunchConfig{};
  }
  uint8_t bpp = static_cast<uint8_t>(p.bitsPerPass != 0 ? p.bitsPerPass : 8);
  if (p.mode == SortHistMode::Sort && p.device.smemPerBlock != 0) {
    bpp = std::min<uint8_t>(bpp, 8);
    while (bpp > 1) {
      const uint64_t digits = 1ULL << bpp;
      if (digits * 4U <= p.device.smemPerBlock) {
        break;
      }
      bpp = static_cast<uint8_t>(bpp / 2U);
    }
  }
  uint32_t threads = detail::fastThreads(p.device, p.kernel);
  if (p.mode == SortHistMode::Histogram && p.numBins > 0 &&
      p.device.smemPerBlock != 0) {
    const uint64_t need = static_cast<uint64_t>(p.numBins) * 4U;
    if (need > p.device.smemPerBlock) {
      threads = threads > 128 ? 128 : threads;
    }
  }
  const uint64_t perThread =
      static_cast<uint64_t>(threads) * FAST_ELEMS_PER_THREAD;
  uint64_t blocks = detail::ceilDiv(p.numElements, perThread);
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

} // namespace ncore::heuristics::kernels
