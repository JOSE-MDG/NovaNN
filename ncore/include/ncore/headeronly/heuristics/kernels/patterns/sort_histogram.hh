/**
 * @file sort_histogram.hh
 * @brief Launch heuristic for radix sorts and histograms.
 */

#pragma once

#include <algorithm>
#include <bit>

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

/// Pass count from the proven value range and the chosen radix width.
inline uint64_t sortPassCount(const SortHistogramParams &p) noexcept {
  if (p.mode == SortHistMode::Histogram) {
    return 1U;
  }
  uint8_t bpp = p.bitsPerPass != 0 ? p.bitsPerPass : sortFastBitsPerPass();
  if (bpp == 0 || bpp > 8) {
    bpp = sortFastBitsPerPass();
  }
  const uint64_t keyBits =
      p.valueMax > p.valueMin
          ? static_cast<uint64_t>(
                std::bit_width(static_cast<uint64_t>(p.valueMax) -
                               static_cast<uint64_t>(p.valueMin)))
          : static_cast<uint64_t>(p.keyItemSize != 0 ? p.keyItemSize : 4U) * 8U;
  if (p.distribution.numDistinct != 0 &&
      p.distribution.numDistinct <= (1ULL << bpp)) {
    return 1U;
  }
  return ceilDiv(keyBits, bpp);
}

} // namespace detail

/// Total elements counted or moved per pass.
inline uint64_t fastParallelWork(const SortHistogramParams &p) noexcept {
  return detail::satMul(p.numElements, detail::sortPassCount(p));
}

/// Bytes moved counting keys plus payloads per pass.
inline uint64_t fastTrafficBytes(const SortHistogramParams &p) noexcept {
  const uint64_t payload = p.mode == SortHistMode::Sort ? p.valueItemSize : 0U;
  return detail::satMul(
      detail::satMul(p.numElements,
                     static_cast<uint64_t>(p.keyItemSize) + payload),
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
  const uint64_t cap = fastGridCap(p.device, p.kernel, threads);
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
 * saturating grid. Privatization multiplies the bin storage, so it is part
 * of the same budget. A concentrated distribution promotes Auto to warp
 * privatization. Narrower radix passes reread the input more often and
 * therefore coarsen each thread proportionally.
 */
inline LaunchConfig
resolveSortHistogram(const SortHistogramParams &p) noexcept {
  if (p.numElements == 0) {
    return LaunchConfig{};
  }
  uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint32_t warp = detail::effectiveWarp(p.device);
  Privatization priv = p.privatization;
  if (priv == Privatization::Auto && p.distribution.concentration01 > 0.8F) {
    priv = Privatization::PerWarp;
  }
  if (p.mode == SortHistMode::Histogram && p.numBins > 0 &&
      p.device.smemPerBlock != 0) {
    uint64_t copies = 1U;
    if (priv == Privatization::PerWarp) {
      copies = threads / warp != 0 ? threads / warp : 1U;
    } else if (priv == Privatization::PerThread) {
      copies = threads;
    }
    const uint64_t need =
        detail::satMul(static_cast<uint64_t>(p.numBins) * 4U, copies);
    if (need > p.device.smemPerBlock) {
      threads = threads > 128 ? 128 : threads;
    }
  }
  uint64_t width = 8U;
  if (p.mode == SortHistMode::Sort && p.bitsPerPass >= 1U &&
      p.bitsPerPass < 8U) {
    width = p.bitsPerPass;
  }
  const uint64_t perThread = detail::satMul(
      static_cast<uint64_t>(threads) * FAST_ELEMS_PER_THREAD, 8U / width);
  uint64_t blocks = detail::ceilDiv(p.numElements, perThread);
  const uint64_t cap = fastGridCap(p.device, p.kernel, threads);
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
