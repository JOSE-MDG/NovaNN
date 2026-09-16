/**
 * @file launch_config.hh
 * @brief Single entry point for kernel launch heuristics.
 *
 * @details
 * Include this header and call `resolveLaunchConfig` with any pattern
 * parameter struct.
 */

#pragma once

#include "patterns/routing.hh"

namespace ncore::heuristics::kernels {

namespace detail {

// Per-stage shape/dtype fold; defined after every fullKeyFor overload.
inline uint64_t fusedStageKey(const FusedStage &s, uint32_t arch,
                              uint32_t depth) noexcept;

/// Shape buckets quantize to powers of two so nearby shapes share.
inline FullCacheKey fullKeyFor(const ElementWiseParams &p,
                               uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::ElementWise;
  key.shapeBucket =
      foldU64(static_cast<uint64_t>(0xE1E4), nextPow2(p.numElements));
  key.dtypeTag = foldDtypeTag(p.inputItemSize, p.outputItemSize);
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const ReductionParams &p,
                               uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::Reduction;
  key.shapeBucket =
      foldU64(foldU64(nextPow2(p.reduceExtent), nextPow2(p.numReductions)),
              static_cast<uint64_t>(p.op));
  key.dtypeTag = foldDtypeTag(p.inputItemSize, p.accumItemSize);
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const LayoutTransformParams &p,
                               uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::LayoutTransformation;
  uint64_t h = 0x1A90U;
  for (uint32_t i = 0; i < p.layout.ndim && i < LAYOUT_MAX_RANK; ++i) {
    h = foldU64(h, nextPow2(p.layout.shape[i] > 0
                                ? static_cast<uint64_t>(p.layout.shape[i])
                                : 0U));
  }
  key.shapeBucket = h;
  key.dtypeTag = p.itemSize;
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const ParallelScanParams &p,
                               uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::ParallelScan;
  key.shapeBucket =
      foldU64(foldU64(nextPow2(p.scanExtent), nextPow2(p.numScans)),
              static_cast<uint64_t>(p.op));
  key.dtypeTag = p.itemSize;
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const SortHistogramParams &p,
                               uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::SortAndHistogram;
  key.shapeBucket =
      foldU64(foldU64(nextPow2(p.numElements), nextPow2(p.numBins)),
              static_cast<uint64_t>(p.mode));
  key.dtypeTag = foldDtypeTag(p.keyItemSize, p.valueItemSize);
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const SpatialNeighborhoodParams &p,
                               uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::SpatialNeighborhood;
  uint64_t h = foldU64(nextPow2(p.inH), nextPow2(p.inW));
  h = foldU64(h, nextPow2(p.inD));
  h = foldU64(h, nextPow2(p.outChannels));
  key.shapeBucket = h;
  key.dtypeTag = p.itemSize;
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const GemmParams &p, uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::GEMM;
  uint64_t h = foldU64(nextPow2(p.M), nextPow2(p.N));
  h = foldU64(h, nextPow2(p.K));
  h = foldU64(h, static_cast<uint64_t>(p.splitK));
  h = foldU64(h, static_cast<uint64_t>(p.schedule));
  key.shapeBucket = h;
  key.dtypeTag = foldDtypeTag(static_cast<uint64_t>(p.inputDtype),
                              static_cast<uint64_t>(p.accumDtype));
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const GatherScatterParams &p,
                               uint32_t arch) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::GatherScatter;
  key.shapeBucket = foldU64(nextPow2(p.numIndices), nextPow2(p.segmentSize));
  key.dtypeTag = foldDtypeTag(p.dataItemSize, p.indexItemSize);
  key.archTag = arch;
  return key;
}

inline FullCacheKey fullKeyFor(const FusedOperatorParams &p, uint32_t arch,
                               uint32_t depth = 0) noexcept {
  FullCacheKey key{};
  key.pattern = ExecutionPattern::FusedOperator;
  uint64_t h = p.numStages;
  const uint32_t n =
      p.numStages < FUSED_MAX_STAGES ? p.numStages : FUSED_MAX_STAGES;
  const bool deep = depth >= FUSED_MAX_NESTING;
  for (uint32_t i = 0; i < n; ++i) {
    h = foldU64(h, static_cast<uint64_t>(p.stages[i].pattern));
    const FusedStage &s = p.stages[i];
    if (!deep && s.params != nullptr && s.params->pattern() == s.pattern) {
      h = foldU64(h, fusedStageKey(s, arch, depth));
    }
  }
  key.shapeBucket = h;
  key.archTag = arch;
  return key;
}

/// Per-stage shape/dtype fold backing the fused key.
inline uint64_t fusedStageKey(const FusedStage &s, uint32_t arch,
                              uint32_t depth) noexcept {
  switch (s.pattern) {
  case ExecutionPattern::ElementWise: {
    const FullCacheKey k =
        fullKeyFor(static_cast<const ElementWiseParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::Reduction: {
    const FullCacheKey k =
        fullKeyFor(static_cast<const ReductionParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::LayoutTransformation: {
    const FullCacheKey k =
        fullKeyFor(static_cast<const LayoutTransformParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::ParallelScan: {
    const FullCacheKey k =
        fullKeyFor(static_cast<const ParallelScanParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::SortAndHistogram: {
    const FullCacheKey k =
        fullKeyFor(static_cast<const SortHistogramParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::SpatialNeighborhood: {
    const FullCacheKey k = fullKeyFor(
        static_cast<const SpatialNeighborhoodParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::GEMM: {
    const FullCacheKey k =
        fullKeyFor(static_cast<const GemmParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::GatherScatter: {
    const FullCacheKey k =
        fullKeyFor(static_cast<const GatherScatterParams &>(*s.params), arch);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  case ExecutionPattern::FusedOperator: {
    const FullCacheKey k = fullKeyFor(
        static_cast<const FusedOperatorParams &>(*s.params), arch, depth + 1);
    return foldU64(k.shapeBucket, k.dtypeTag);
  }
  }
  return foldU64(0x9E3779B9ULL, static_cast<uint64_t>(s.pattern));
}

/// Lookup-or-compute wrapper shared by every Full branch.
template <typename Params, typename Resolver>
inline LaunchConfig resolveCached(const Params &d, const DeviceCaps &dev,
                                  const KernelAttrs &k, FullConfigCache *cache,
                                  Resolver resolve) noexcept {
  const FullCacheKey key = fullKeyFor(d, dev.archVersion);
  LaunchConfig cfg{};
  if (cache != nullptr && cache->lookup(key, &cfg)) {
    return clampConfig(cfg, dev, k);
  }
  cfg = resolve(d);
  if (cache != nullptr) {
    cache->store(key, cfg);
  }
  return clampConfig(cfg, dev, k);
}

} // namespace detail

/**
 * @brief Resolve the grid/block pair for one launch.
 *
 * @details
 * Probes the optional config cache when provided and stores misses.
 * Returns the all-ones floor on degenerate input, never an illegal
 * launch.
 *
 * @param[in] params Pattern facts with cached device data.
 * @param[in] cache  Optional config cache, null to skip.
 * @return Grid/block pair for `dim3` construction.
 */
inline LaunchConfig
resolveLaunchConfig(const LaunchParamsBase &params,
                    FullConfigCache *cache = nullptr) noexcept {
  const ExecutionPattern tag = params.pattern();
  LaunchConfig cfg{};
  if (resolveTier(params) == HeuristicTier::Fast) {
    switch (tag) {
    case ExecutionPattern::ElementWise:
      cfg = resolveElementWiseFast(
          static_cast<const ElementWiseParams &>(params));
      break;
    case ExecutionPattern::Reduction:
      cfg = resolveReductionFast(static_cast<const ReductionParams &>(params));
      break;
    case ExecutionPattern::LayoutTransformation:
      cfg = resolveLayoutTransformFast(
          static_cast<const LayoutTransformParams &>(params));
      break;
    case ExecutionPattern::ParallelScan:
      cfg = resolveParallelScanFast(
          static_cast<const ParallelScanParams &>(params));
      break;
    case ExecutionPattern::SortAndHistogram:
      cfg = resolveSortHistogramFast(
          static_cast<const SortHistogramParams &>(params));
      break;
    case ExecutionPattern::SpatialNeighborhood:
      cfg = resolveSpatialNeighborhoodFast(
          static_cast<const SpatialNeighborhoodParams &>(params));
      break;
    case ExecutionPattern::GEMM:
      cfg = resolveGemmFast(static_cast<const GemmParams &>(params));
      break;
    case ExecutionPattern::GatherScatter:
      cfg = resolveGatherScatterFast(
          static_cast<const GatherScatterParams &>(params));
      break;
    case ExecutionPattern::FusedOperator:
      cfg = resolveFusedOperatorFast(
          static_cast<const FusedOperatorParams &>(params));
      break;
    }
    return cfg;
  }
  switch (tag) {
  case ExecutionPattern::ElementWise:
    cfg = detail::resolveCached(static_cast<const ElementWiseParams &>(params),
                                params.device, params.kernel, cache,
                                resolveElementWise);
    break;
  case ExecutionPattern::Reduction:
    cfg = detail::resolveCached(static_cast<const ReductionParams &>(params),
                                params.device, params.kernel, cache,
                                resolveReduction);
    break;
  case ExecutionPattern::LayoutTransformation:
    cfg = detail::resolveCached(
        static_cast<const LayoutTransformParams &>(params), params.device,
        params.kernel, cache, resolveLayoutTransform);
    break;
  case ExecutionPattern::ParallelScan:
    cfg = detail::resolveCached(static_cast<const ParallelScanParams &>(params),
                                params.device, params.kernel, cache,
                                resolveParallelScan);
    break;
  case ExecutionPattern::SortAndHistogram:
    cfg = detail::resolveCached(
        static_cast<const SortHistogramParams &>(params), params.device,
        params.kernel, cache, resolveSortHistogram);
    break;
  case ExecutionPattern::SpatialNeighborhood:
    cfg = detail::resolveCached(
        static_cast<const SpatialNeighborhoodParams &>(params), params.device,
        params.kernel, cache, resolveSpatialNeighborhood);
    break;
  case ExecutionPattern::GEMM:
    cfg =
        detail::resolveCached(static_cast<const GemmParams &>(params),
                              params.device, params.kernel, cache, resolveGemm);
    break;
  case ExecutionPattern::GatherScatter:
    cfg = detail::resolveCached(
        static_cast<const GatherScatterParams &>(params), params.device,
        params.kernel, cache, resolveGatherScatter);
    break;
  case ExecutionPattern::FusedOperator:
    cfg = detail::resolveCached(
        static_cast<const FusedOperatorParams &>(params), params.device,
        params.kernel, cache, resolveFusedOperator);
    break;
  }
  return cfg;
}

} // namespace ncore::heuristics::kernels
