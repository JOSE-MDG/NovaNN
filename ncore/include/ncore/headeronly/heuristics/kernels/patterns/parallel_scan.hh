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

/**
 * @brief Odd elements per thread that fit the final tile in smem.
 *
 * @details
 * Candidates 7, 5, 3 are coprime with the 32 banks without demanding
 * divisibility. When none fits at @p threads, the block is halved (one
 * warp floor) and the scan repeats; a tile that still cannot stage
 * falls back to one element per thread.
 */
inline uint64_t scanElemsPerThread(const ParallelScanParams &p,
                                   uint32_t threads) noexcept {
  const uint32_t warp = detail::effectiveWarp(p.device);
  uint32_t t = threads != 0 ? threads : warp;
  const uint64_t fixed =
      detail::satAdd(static_cast<uint64_t>(p.kernel.staticSmemBytes),
                     static_cast<uint64_t>(p.kernel.dynSmemPerBlock));
  for (;;) {
    for (uint64_t ept = 7; ept >= 3; ept -= 2) {
      if (p.device.smemPerBlock == 0) {
        return ept;
      }
      const uint64_t bytes = detail::satAdd(
          detail::satMul(t, detail::satMul(ept, p.itemSize)), fixed);
      if (bytes <= p.device.smemPerBlock) {
        return ept;
      }
    }
    if (t <= warp) {
      break;
    }
    t = (t / 2U) < warp ? warp : t / 2U;
  }
  return 1;
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
  const uint64_t cap = fastGridCap(p.device, p.kernel, threads);
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
  uint32_t threads = pickBlockThreads(p.device, p.kernel);
  const uint64_t ept = detail::scanElemsPerThread(p, threads);
  const uint64_t tile = detail::satMul(threads, ept);
  uint64_t pieces = detail::ceilDiv(p.scanExtent, tile);
  if (pieces == 0) {
    pieces = 1;
  }
  if (pieces == 1) {
    const uint64_t needThreads = detail::ceilDiv(p.scanExtent, ept);
    const uint32_t need =
        needThreads > ~0U ? ~0U : static_cast<uint32_t>(needThreads);
    uint32_t t = pickBlockThreads(p.device, p.kernel, need);
    if (p.device.smCount != 0 &&
        p.numScans >= 2ULL * static_cast<uint64_t>(p.device.smCount)) {
      t = std::max(t, std::min(minThreadsForBlockLimit(p.device),
                               pickBlockThreads(p.device, p.kernel)));
    }
    threads = t;
  }
  if (threads == 0) {
    threads = warp;
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
  return detail::clampConfig(cfg, p.device, p.kernel);
}

} // namespace ncore::heuristics::kernels
