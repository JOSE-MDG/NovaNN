/**
 * @file fused_operator.hh
 * @brief Launch heuristic for multi-pattern fused kernels.
 */

#pragma once

#include <array>

#include "common.hh"
#include "element_wise.hh"
#include "gather_scatter.hh"
#include "gemm.hh"
#include "layout_transform.hh"
#include "parallel_scan.hh"
#include "reduction.hh"
#include "sort_histogram.hh"
#include "spatial_neighborhood.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum RecomputePolicy
 * @brief Recompute cheap intermediates vs keeping them, Auto decides.
 */
enum class RecomputePolicy : uint8_t { Auto, Recompute, Keep };

/**
 * @enum IntermediatePlacement
 * @brief Where stage intermediates live, Auto decides.
 *
 * @details
 * Global placement breaks the fusion at the first boundary and sizes the
 * launch from the prefix alone. TensorMemory is the SM90+ accumulator tier
 * (Blackwell TMEM); the remaining chain still falls registers, then shared
 * with dead-stage reuse, then breaks.
 */
enum class IntermediatePlacement : uint8_t {
  Auto,
  Registers,
  SharedMem,
  Global,
  TensorMemory
};

/**
 * @enum SyncScope
 * @brief Block-wide vs warp-only stage barriers, Auto decides.
 *
 * @details
 * ClusterWide and GridWide bound the grid to one resident wave because
 * every participating block must be resident together. ClusterWide spans
 * one thread block cluster, GridWide needs a cooperative launch.
 */
enum class SyncScope : uint8_t {
  Auto,
  BlockWide,
  WarpOnly,
  ClusterWide,
  GridWide
};

/// Stage slots per fused launch.
inline constexpr uint8_t FUSED_MAX_STAGES = 8;
/// Fast stage cap.
inline constexpr uint8_t FUSED_FAST_MAX_STAGES = 4;
/// Nesting cap for recursive stage walks; deeper levels degrade to
/// pattern-only keys and unknown work instead of recursing forever.
inline constexpr uint32_t FUSED_MAX_NESTING = 8;

/**
 * @struct FusedStage
 * @brief One stage: its pattern plus its own parameter struct.
 */
struct FusedStage {
  ExecutionPattern pattern = ExecutionPattern::ElementWise;
  const LaunchParamsBase *params = nullptr; ///< Stage facts, borrowed.
};

/**
 * @struct FusedOperatorParams
 * @brief Facts for one fused launch.
 */
struct FusedOperatorParams : LaunchParamsBase {
  std::array<FusedStage, FUSED_MAX_STAGES> stages; ///< Stages in order.
  uint8_t numStages = 0;                           ///< Active stage count.
  bool smemReusable = false;  ///< Dead stages release shared.
  uint32_t maxLiveValues = 0; ///< Peak simultaneously live values.
  RecomputePolicy recompute = RecomputePolicy::Auto;
  IntermediatePlacement placement = IntermediatePlacement::Auto;
  SyncScope sync = SyncScope::Auto;
  bool incrementalBlocks = false; ///< Streaming window formulation.

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::FusedOperator;
  }
};

/**
 * @struct FusedPlan
 * @brief Geometry and the stage decisions the fused kernel must honour.
 */
struct FusedPlan {
  LaunchConfig config;              ///< Grid/block pair.
  uint64_t work = 0;                ///< Element work used to size the grid.
  uint64_t bytes = 0;               ///< Traffic of the fused prefix.
  bool splitAtBoundary = false;     ///< The fusion stops after its first stage.
  SyncScope sync = SyncScope::Auto; ///< Barrier scope required by the kernel.
};

/// True for short keep-everything fusions of valid stages.
inline bool coversFusedOperatorFast(const FusedOperatorParams &p) noexcept;

namespace detail {

// Summed stage work; defined after fusedStageWork.
inline uint64_t fusedWork(const FusedOperatorParams &p,
                          uint32_t depth = 0) noexcept;

/// Element-sized work per stage; nested fusions recurse with depth,
/// unknown stages contribute zero and keep the device-wide grid.
inline uint64_t fusedStageWork(const FusedStage &s, uint32_t depth) noexcept {
  if (s.params == nullptr || s.params->pattern() != s.pattern) {
    return 0;
  }
  switch (s.pattern) {
  case ExecutionPattern::ElementWise:
    return fastParallelWork(static_cast<const ElementWiseParams &>(*s.params));
  case ExecutionPattern::Reduction:
    return fastParallelWork(static_cast<const ReductionParams &>(*s.params));
  case ExecutionPattern::LayoutTransformation:
    return layoutElements(
        static_cast<const LayoutTransformParams &>(*s.params));
  case ExecutionPattern::ParallelScan:
    return fastParallelWork(static_cast<const ParallelScanParams &>(*s.params));
  case ExecutionPattern::SortAndHistogram:
    return static_cast<const SortHistogramParams &>(*s.params).numElements;
  case ExecutionPattern::SpatialNeighborhood:
    return spatialOutputs(
        static_cast<const SpatialNeighborhoodParams &>(*s.params));
  case ExecutionPattern::GEMM: {
    const GemmParams &g = static_cast<const GemmParams &>(*s.params);
    return satMul(g.M, g.N);
  }
  case ExecutionPattern::GatherScatter:
    return fastParallelWork(
        static_cast<const GatherScatterParams &>(*s.params));
  case ExecutionPattern::FusedOperator:
    return fusedWork(static_cast<const FusedOperatorParams &>(*s.params),
                     depth + 1);
  }
  return 0;
}

/// Summed stage work, saturating; zero when nothing is measurable or
/// past the nesting cap.
inline uint64_t fusedWork(const FusedOperatorParams &p,
                          uint32_t depth) noexcept {
  if (depth >= FUSED_MAX_NESTING) {
    return 0;
  }
  uint64_t total = 0;
  const uint32_t n =
      p.numStages < FUSED_MAX_STAGES ? p.numStages : FUSED_MAX_STAGES;
  for (uint32_t i = 0; i < n; ++i) {
    total = satAdd(total, fusedStageWork(p.stages[i], depth));
  }
  return total;
}

inline uint64_t fusedTrafficBytes(const FusedOperatorParams &p,
                                  uint32_t depth) noexcept;

/// Traffic of one stage, including the operands a GEMM stage reloads.
inline uint64_t fusedStageBytes(const FusedStage &s, uint32_t depth) noexcept {
  if (s.params == nullptr || s.params->pattern() != s.pattern) {
    return 0;
  }
  switch (s.pattern) {
  case ExecutionPattern::ElementWise:
    return fastTrafficBytes(static_cast<const ElementWiseParams &>(*s.params));
  case ExecutionPattern::Reduction:
    return fastTrafficBytes(static_cast<const ReductionParams &>(*s.params));
  case ExecutionPattern::LayoutTransformation: {
    const auto &q = static_cast<const LayoutTransformParams &>(*s.params);
    return satMul(layoutElements(q), satMul(q.itemSize, 2U));
  }
  case ExecutionPattern::ParallelScan:
    return fastTrafficBytes(static_cast<const ParallelScanParams &>(*s.params));
  case ExecutionPattern::SortAndHistogram:
    return fastTrafficBytes(
        static_cast<const SortHistogramParams &>(*s.params));
  case ExecutionPattern::SpatialNeighborhood:
    return fastTrafficBytes(
        static_cast<const SpatialNeighborhoodParams &>(*s.params));
  case ExecutionPattern::GEMM: {
    const auto &g = static_cast<const GemmParams &>(*s.params);
    const uint64_t operands = satAdd(satMul(g.M, g.K), satMul(g.K, g.N));
    return satMul(operands, g.inputDtype == Float64 ? 8U : 4U);
  }
  case ExecutionPattern::GatherScatter:
    return fastTrafficBytes(
        static_cast<const GatherScatterParams &>(*s.params));
  case ExecutionPattern::FusedOperator:
    return fusedTrafficBytes(
        static_cast<const FusedOperatorParams &>(*s.params), depth + 1);
  }
  return 0;
}

/// Summed stage traffic, saturating and bounded by the nesting cap.
inline uint64_t fusedTrafficBytes(const FusedOperatorParams &p,
                                  uint32_t depth = 0) noexcept {
  if (depth >= FUSED_MAX_NESTING) {
    return 0;
  }
  uint64_t total = 0;
  const uint32_t n =
      p.numStages < FUSED_MAX_STAGES ? p.numStages : FUSED_MAX_STAGES;
  for (uint32_t i = 0; i < n; ++i) {
    total = satAdd(total, fusedStageBytes(p.stages[i], depth));
  }
  return total;
}

/// Grid from summed stage work, device-wide when unmeasurable.
inline uint64_t fusedGridBlocks(const FusedOperatorParams &p,
                                uint32_t threads) noexcept {
  const uint64_t cap = fastGridCap(p.device, p.kernel, threads);
  uint64_t blocks = cap != 0 ? cap : 1;
  const uint64_t work = fusedWork(p);
  if (work != 0) {
    const uint64_t perThread =
        static_cast<uint64_t>(threads) * FAST_ELEMS_PER_THREAD;
    blocks = ceilDiv(work, perThread);
  }
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  return blocks != 0 ? blocks : 1;
}

/// Stage slots are filled, tagged, and paired with facts.
inline bool fusedStagesValid(const FusedOperatorParams &p) noexcept {
  if (p.numStages == 0 || p.numStages > FUSED_MAX_STAGES) {
    return false;
  }
  for (uint32_t i = 0; i < p.numStages; ++i) {
    if (p.stages[i].params == nullptr ||
        p.stages[i].params->pattern() != p.stages[i].pattern) {
      return false;
    }
  }
  return true;
}

/// Per-stage cheap-path check; tags already validated above. A nested
/// fusion answers false: depth is unbounded, so it always takes the
/// exhaustive path instead of recursing.
inline bool fusedStageFastCoverable(const FusedStage &s) noexcept {
  switch (s.pattern) {
  case ExecutionPattern::ElementWise:
    return coversElementWiseFast(
        static_cast<const ElementWiseParams &>(*s.params));
  case ExecutionPattern::Reduction:
    return coversReductionFast(static_cast<const ReductionParams &>(*s.params));
  case ExecutionPattern::LayoutTransformation:
    return coversLayoutTransformFast(
        static_cast<const LayoutTransformParams &>(*s.params));
  case ExecutionPattern::ParallelScan:
    return coversParallelScanFast(
        static_cast<const ParallelScanParams &>(*s.params));
  case ExecutionPattern::SortAndHistogram:
    return coversSortHistogramFast(
        static_cast<const SortHistogramParams &>(*s.params));
  case ExecutionPattern::SpatialNeighborhood:
    return coversSpatialNeighborhoodFast(
        static_cast<const SpatialNeighborhoodParams &>(*s.params));
  case ExecutionPattern::GEMM:
    return coversGemmFast(static_cast<const GemmParams &>(*s.params));
  case ExecutionPattern::GatherScatter:
    return coversGatherScatterFast(
        static_cast<const GatherScatterParams &>(*s.params));
  case ExecutionPattern::FusedOperator:
    return false;
  }
  return false;
}

} // namespace detail

/// Total work, with recomputed stages counted again.
inline uint64_t fastParallelWork(const FusedOperatorParams &p) noexcept {
  const uint64_t work = detail::fusedWork(p);
  return p.recompute == RecomputePolicy::Recompute ? detail::satMul(work, 2U)
                                                   : work;
}

/// Summed stage traffic.
inline uint64_t fastTrafficBytes(const FusedOperatorParams &p) noexcept {
  return detail::fusedTrafficBytes(p);
}

inline bool coversFusedOperatorFast(const FusedOperatorParams &p) noexcept {
  if (!detail::fusedStagesValid(p)) {
    return p.numStages == 0;
  }
  if (p.numStages > FUSED_FAST_MAX_STAGES) {
    return false;
  }
  for (uint32_t i = 0; i < p.numStages; ++i) {
    if (!detail::fusedStageFastCoverable(p.stages[i])) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Keep-all intermediates, fixed threads.
 */
inline LaunchConfig
resolveFusedOperatorFast(const FusedOperatorParams &p) noexcept {
  if (!detail::fusedStagesValid(p)) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  LaunchConfig cfg{};
  cfg.threads.x = threads;
  const uint64_t blocks = detail::fusedGridBlocks(p, threads);
  cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

/**
 * @brief Occupancy-sized fusion launch.
 *
 * @details
 * Placement falls registers, then shared with dead-stage reuse, then
 * breaks the fusion at the first illegal boundary: the caller launches
 * the prefix returned here and the remainder separately, never spilling
 * through global memory inside one fused launch. A grid-wide or cluster
 * barrier, or a streaming window, bounds the grid to one resident wave.
 * Live values cap the block because more simultaneous values need more
 * registers per thread; the plan carries the work and traffic used to
 * size it together with the required barrier scope.
 */
inline FusedPlan
resolveFusedOperatorFullPlan(const FusedOperatorParams &p) noexcept {
  FusedPlan plan{};
  if (!detail::fusedStagesValid(p)) {
    return plan;
  }
  KernelAttrs kk = p.kernel;
  if (p.smemReusable && kk.dynSmemPerBlock != 0U) {
    kk.dynSmemPerBlock /= 2U;
  }
  uint32_t threads = pickBlockThreads(p.device, kk);
  if (threads == 0) {
    threads = detail::effectiveWarp(p.device);
  }
  if (p.maxLiveValues > 32U) {
    const uint64_t rounded = detail::nextPow2(p.maxLiveValues);
    const uint32_t liveCap =
        rounded > ~0U ? ~0U : static_cast<uint32_t>(rounded);
    threads = std::min(threads, liveCap != 0 ? liveCap : threads);
  }
  FusedOperatorParams sized = p;
  sized.kernel = kk;
  plan.splitAtBoundary = p.placement == IntermediatePlacement::Global;
  if (plan.splitAtBoundary) {
    sized.numStages = 1U;
  }
  uint64_t blocks = detail::fusedGridBlocks(sized, threads);
  const bool bounded = p.sync == SyncScope::ClusterWide ||
                       p.sync == SyncScope::GridWide || p.incrementalBlocks;
  if (bounded) {
    const uint64_t cap = residentGridCap(p.device, kk, threads, 1U, 1U);
    if (cap != 0 && blocks > cap) {
      blocks = cap;
    }
  }
  plan.config.threads.x = threads;
  plan.config.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  if (plan.config.blocks.x == 0) {
    plan.config.blocks.x = 1;
  }
  plan.config = detail::clampConfig(plan.config, p.device, kk);
  plan.work = fastParallelWork(sized);
  plan.bytes = fastTrafficBytes(sized);
  plan.sync = p.sync;
  return plan;
}

/**
 * @brief Fusion geometry reduced to its grid/block pair.
 */
inline LaunchConfig
resolveFusedOperator(const FusedOperatorParams &p) noexcept {
  return resolveFusedOperatorFullPlan(p).config;
}

} // namespace ncore::heuristics::kernels
