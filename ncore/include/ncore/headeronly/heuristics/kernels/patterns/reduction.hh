/**
 * @file reduction.hh
 * @brief Launch heuristic for axis reductions over a batch.
 */

#pragma once

#include <algorithm>

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum ReductionOp
 * @brief Operator classes distinguished by launch-relevant behavior.
 *
 * @details
 * Plain values combine with one associative binary op. Index-carrying
 * and multi-value ops widen partial storage. Welford ops keep a
 * count/mean/M2 triple. Selection-based ops cannot combine through a
 * binary operator at all. Custom is the conservative escape hatch for
 * anything unlisted: stateful-sized partials, no shuffle-only path.
 */
enum class ReductionOp : uint8_t {
  Add,
  Mul,
  Min,
  Max,
  And,
  Or,
  ArgMax,
  ArgMin,
  MinMax,
  MeanVar,
  Norm,
  LogSumExp,
  Median,
  Unique,
  Custom
};

/**
 * @enum IntraBlockStrategy
 * @brief Warp-shuffle-only vs shared-memory combining, Auto decides.
 */
enum class IntraBlockStrategy : uint8_t { Auto, ShuffleOnly, SharedMem };

/**
 * @enum CrossBlockStrategy
 * @brief Single-pass atomics vs classic two-pass, Auto decides.
 */
enum class CrossBlockStrategy : uint8_t { Auto, SinglePassAtomics, TwoPass };

/**
 * @struct ReductionParams
 * @brief Facts for one reduction launch.
 */
struct ReductionParams : LaunchParamsBase {
  uint64_t reduceExtent = 0;         ///< Axis length to reduce.
  uint64_t numReductions = 0;        ///< Independent reductions (batch).
  bool reduceContiguous = false;     ///< Reduce axis stride-free.
  ReductionOp op = ReductionOp::Add; ///< Reduction operator.
  uint32_t inputItemSize = 0;        ///< Input bytes per element.
  uint32_t accumItemSize = 0;        ///< Accumulator bytes per live value.
  bool deterministic = false;        ///< Bit-wise repeatability required.
  uint32_t blocksPerOutput = 0;      ///< Writers per destination for atomics.
  IntraBlockStrategy intraBlock = IntraBlockStrategy::Auto;
  CrossBlockStrategy crossBlock = CrossBlockStrategy::Auto;

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::Reduction;
  }
};

namespace detail {

/// Partial values per element: pairs, Welford triples, Custom worst case.
constexpr uint64_t reductionPartials(ReductionOp op) noexcept {
  switch (op) {
  case ReductionOp::ArgMax:
  case ReductionOp::ArgMin:
  case ReductionOp::MinMax:
  case ReductionOp::Custom:
    return 2;
  case ReductionOp::MeanVar:
    return 3;
  default:
    return 1;
  }
}

/// False for selection-based and unknown ops, which no binary
/// operator can combine. Forces shared staging and two-pass combining.
constexpr bool reductionSimpleCombine(ReductionOp op) noexcept {
  switch (op) {
  case ReductionOp::Median:
  case ReductionOp::Unique:
  case ReductionOp::Custom:
    return false;
  default:
    return true;
  }
}

/// Single-block capacity in input elements under the smem budget.
inline uint64_t reductionSingleBlockCap(const ReductionParams &p) noexcept {
  const uint32_t warp = detail::effectiveWarp(p.device);
  uint32_t maxT = p.device.maxThreadsPerBlockDevice;
  if (p.kernel.maxThreadsPerBlockKernel != 0 &&
      (maxT == 0 || p.kernel.maxThreadsPerBlockKernel < maxT)) {
    maxT = p.kernel.maxThreadsPerBlockKernel;
  }
  if (maxT == 0) {
    maxT = 1024;
  }
  const uint64_t perThread = 4;
  uint64_t cap = static_cast<uint64_t>(maxT) * perThread;
  if (p.device.smemPerBlock != 0 && p.accumItemSize != 0) {
    const uint64_t partials = reductionPartials(p.op);
    const uint64_t smemCap = satMul(
        satMul(p.device.smemPerBlock / satMul(p.accumItemSize, partials), warp),
        perThread);
    cap = std::min(smemCap, cap);
  }
  return cap;
}

} // namespace detail

/// Total input elements across the batch.
inline uint64_t fastParallelWork(const ReductionParams &p) noexcept {
  return detail::satMul(p.reduceExtent, p.numReductions);
}

/// Bytes moved counting inputs once plus partials per reduction.
inline uint64_t fastTrafficBytes(const ReductionParams &p) noexcept {
  return detail::satAdd(
      detail::satMul(detail::satMul(p.reduceExtent, p.numReductions),
                     p.inputItemSize),
      detail::satMul(
          p.numReductions,
          detail::satMul(p.accumItemSize, detail::reductionPartials(p.op))));
}

/// Whether the reduce axis is contiguous.
inline bool fastProvenCoalesced(const ReductionParams &p) noexcept {
  return p.reduceContiguous;
}

/**
 * @brief True when the whole axis fits one block.
 */
inline bool coversReductionFast(const ReductionParams &p) noexcept {
  if (p.reduceExtent == 0 || p.numReductions == 0) {
    return true;
  }
  return p.reduceExtent <= detail::reductionSingleBlockCap(p);
}

/**
 * @brief One block per reduction, masked tail.
 */
inline LaunchConfig resolveReductionFast(const ReductionParams &p) noexcept {
  if (p.reduceExtent == 0 || p.numReductions == 0) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t cap =
      static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
  uint64_t blocks = p.numReductions;
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  LaunchConfig cfg{};
  cfg.threads.x = threads;
  cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  return detail::clampConfig(cfg, p.device, p.kernel);
}

/**
 * @brief Coarsening, split pieces, grid.
 *
 * @details
 * `blockIdx.x` spans batches, `blockIdx.y` spans split pieces of one
 * axis when it exceeds a block. The first pass always needs
 * reductions-by-splits blocks; when the axis needs combining across
 * blocks the caller launches the follow-up with one warp-sized block
 * per reduction. Deterministic runs and high atomic contention take
 * the two-pass follow-up; ordered single-pass atomics otherwise. That
 * choice lives in the kernel, not in this grid.
 */
inline LaunchConfig resolveReduction(const ReductionParams &p) noexcept {
  if (p.reduceExtent == 0 || p.numReductions == 0) {
    return LaunchConfig{};
  }
  const uint32_t warp = detail::effectiveWarp(p.device);
  uint32_t width = p.reduceContiguous ? 4 : 1;
  if (p.inputItemSize > 4) {
    width = 1;
  }
  uint64_t elemsPerThread = width * 4;
  if (p.accumItemSize > p.inputItemSize && elemsPerThread > width) {
    elemsPerThread = width * 2;
  }
  const uint64_t singleCap = detail::reductionSingleBlockCap(p);
  uint64_t splits = detail::ceilDiv(p.reduceExtent, singleCap);
  if (splits == 0) {
    splits = 1;
  }
  uint32_t threads = detail::fastThreads(p.device, p.kernel);
  if (splits == 1 && p.reduceExtent <= warp * elemsPerThread &&
      p.intraBlock != IntraBlockStrategy::SharedMem &&
      detail::reductionSimpleCombine(p.op)) {
    uint64_t need = detail::ceilDiv(p.reduceExtent, elemsPerThread * width);
    uint32_t want = static_cast<uint32_t>(need > ~0U ? ~0U : need) * width;
    want = (want / warp) * warp;
    if (want != 0 && want < threads) {
      threads = want;
    }
  }
  if (threads == 0) {
    threads = warp;
  }
  uint64_t blocks = detail::satMul(p.numReductions, splits);
  const uint64_t cap =
      static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  if (blocks == 0) {
    blocks = 1;
  }
  LaunchConfig cfg{};
  cfg.threads.x = threads;
  cfg.blocks.x =
      static_cast<uint32_t>(p.numReductions > ~0U ? ~0U : p.numReductions);
  cfg.blocks.y = static_cast<uint32_t>(splits > ~0U ? ~0U : splits);
  if (detail::satMul(cfg.blocks.x, cfg.blocks.y) > blocks) {
    cfg.blocks.y =
        static_cast<uint32_t>(blocks / (cfg.blocks.x != 0 ? cfg.blocks.x : 1U));
    if (cfg.blocks.y == 0) {
      cfg.blocks.y = 1;
    }
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

} // namespace ncore::heuristics::kernels
