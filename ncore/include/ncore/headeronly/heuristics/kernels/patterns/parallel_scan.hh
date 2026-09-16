/**
 * @file parallel_scan.hh
 * @brief Launch heuristic for cooperative prefix scans.
 */

#pragma once

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum ScanOp
 * @brief Combine operator class by reordering freedom.
 */
enum class ScanOp : uint8_t {
  Add,
  Mul,
  Min,
  Max,
  CustomAssociative,
  CustomRestricted
};

/**
 * @enum BufferStrategy
 * @brief Single vs double shared buffering, Auto decides.
 */
enum class BufferStrategy : uint8_t { Auto, Single, Double };

/**
 * @enum ScanCrossBlock
 * @brief Chained single-pass vs classic multi-pass, Auto decides.
 */
enum class ScanCrossBlock : uint8_t { Auto, ChainedSinglePass, MultiPass };

/// Largest axis one block scans in Fast.
inline constexpr uint64_t SCAN_FAST_MAX_EXTENT = 2048;

/**
 * @struct ParallelScanParams
 * @brief Facts for one scan launch.
 */
struct ParallelScanParams : LaunchParamsBase {
  uint64_t scanExtent = 0;    ///< Axis length to scan.
  uint64_t numScans = 0;      ///< Independent scans (batch).
  uint32_t itemSize = 0;      ///< Bytes per element.
  bool inclusive = false;     ///< Inclusive vs exclusive output.
  ScanOp op = ScanOp::Add;    ///< Combine operator class.
  bool deterministic = false; ///< Arrival-order combining forbidden.
  BufferStrategy buffering = BufferStrategy::Auto;
  ScanCrossBlock crossBlock = ScanCrossBlock::Auto;

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::ParallelScan;
  }
};

namespace detail {

/// Reorderable simple ops admit every scan optimization.
constexpr bool scanOpSimple(ScanOp op) noexcept {
  return op == ScanOp::Add || op == ScanOp::Mul || op == ScanOp::Min ||
         op == ScanOp::Max;
}

/// Odd count per thread while it fits, breaking bank stride alignment.
inline uint64_t scanElemsPerThread(const ParallelScanParams &p,
                                   uint64_t tile) noexcept {
  for (uint64_t ept = 7; ept >= 3; ept -= 2) {
    if (tile % ept == 0 || ept == 3) {
      const uint64_t bytes = satMul(tile, static_cast<uint64_t>(p.itemSize));
      if (p.device.smemPerBlock == 0 || bytes <= p.device.smemPerBlock) {
        return ept;
      }
    }
  }
  return 3;
}

} // namespace detail

/// Total scanned elements across the batch.
inline uint64_t fastParallelWork(const ParallelScanParams &p) noexcept {
  return detail::satMul(p.scanExtent, p.numScans);
}

/// Bytes moved counting read, scan, and write-back.
inline uint64_t fastTrafficBytes(const ParallelScanParams &p) noexcept {
  return detail::satMul(detail::satMul(p.scanExtent, p.numScans),
                        detail::satMul(p.itemSize, 2U));
}

/**
 * @brief True for single-block axes with a reorderable op.
 */
inline bool coversParallelScanFast(const ParallelScanParams &p) noexcept {
  if (p.scanExtent == 0 || p.numScans == 0) {
    return true;
  }
  return p.scanExtent <= SCAN_FAST_MAX_EXTENT &&
         (detail::scanOpSimple(p.op) || p.op == ScanOp::CustomAssociative);
}

/**
 * @brief One block per scan, single buffer.
 */
inline LaunchConfig
resolveParallelScanFast(const ParallelScanParams &p) noexcept {
  if (p.scanExtent == 0 || p.numScans == 0) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t cap =
      static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
  uint64_t blocks = p.numScans;
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
 * @brief Tiling, buffering, cross-block mode.
 *
 * @details
 * Whole axes stay single-block. Longer axes split into pieces; the
 * chained single-pass mode needs ordered cross-block atomics, a
 * reorderable op, and no determinism demand, else classic multi-pass
 * (block sums, scanned, add-back) selected by the caller as follow-up
 * launches.
 */
inline LaunchConfig resolveParallelScan(const ParallelScanParams &p) noexcept {
  if (p.scanExtent == 0 || p.numScans == 0 || p.itemSize == 0) {
    return LaunchConfig{};
  }
  const uint32_t warp = detail::effectiveWarp(p.device);
  uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t tile = static_cast<uint64_t>(threads) *
                        detail::scanElemsPerThread(p, threads * 7);
  uint64_t pieces = detail::ceilDiv(p.scanExtent, tile);
  if (pieces == 0) {
    pieces = 1;
  }
  if (pieces == 1) {
    uint64_t need = detail::ceilDiv(
        p.scanExtent, detail::scanElemsPerThread(p, p.scanExtent));
    uint32_t want = static_cast<uint32_t>(need > ~0U ? ~0U : need);
    want = (want / warp) * warp;
    if (want != 0 && want < threads) {
      threads = want;
    }
  }
  if (threads == 0) {
    threads = warp;
  }
  uint64_t blocks = detail::satMul(p.numScans, pieces);
  const uint64_t cap =
      static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  LaunchConfig cfg{};
  cfg.threads.x = threads;
  cfg.blocks.x = static_cast<uint32_t>(p.numScans > ~0U ? ~0U : p.numScans);
  cfg.blocks.y = static_cast<uint32_t>(pieces > ~0U ? ~0U : pieces);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  if (cfg.blocks.y == 0) {
    cfg.blocks.y = 1;
  }
  if (detail::satMul(cfg.blocks.x, cfg.blocks.y) > blocks && blocks != 0) {
    cfg.blocks.y = static_cast<uint32_t>(blocks / cfg.blocks.x);
    if (cfg.blocks.y == 0) {
      cfg.blocks.y = 1;
    }
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

} // namespace ncore::heuristics::kernels
