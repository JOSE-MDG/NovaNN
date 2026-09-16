/**
 * @file gemm.hh
 * @brief Launch heuristic for dense matrix multiplication.
 */

#pragma once

#include <algorithm>

#include <ncore/core/dtype.h>

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum ActivationKind
 * @brief Epilogue activation fused after the product.
 */
enum class ActivationKind : uint8_t { None, Relu, Gelu, Silu, Softmax, Custom };

/**
 * @enum SplitK
 * @brief K-splitting policy, Auto decides from wave analysis.
 */
enum class SplitK : uint8_t { Auto, NoSplit, SplitK };

/**
 * @enum CtaSchedule
 * @brief Block-per-tile vs persistent work queue, Auto decides.
 *
 * @details
 * StreamK forces the SM-proportional fixed grid with fractional K
 * assignment done by the kernel; Auto never selects it unasked.
 */
enum class CtaSchedule : uint8_t { Auto, BlockPerTile, Persistent, StreamK };

/**
 * @enum TraversalOrder
 * @brief Output tile walk order for L2 reuse, Auto decides.
 */
enum class TraversalOrder : uint8_t { Auto, RowMajor, Swizzled, Grouped };

/**
 * @enum WarpRole
 * @brief Lockstep vs load/compute warp specialization, Auto decides.
 */
enum class WarpRole : uint8_t {
  Auto,
  Lockstep,
  Specialized,
  Cooperative,
  PingPong
};

/// Fast coverage bound on every dimension.
inline constexpr uint64_t GEMM_FAST_MAX_DIM = 128;

/**
 * @struct Epilogue
 * @brief Post-GEMM work fused before the global write.
 */
struct Epilogue {
  bool bias = false;                                ///< Add bias vector.
  ActivationKind activation = ActivationKind::None; ///< Activation.
  bool normalize = false;                           ///< Normalization pass.
};

/**
 * @struct GemmTuningEntry
 * @brief One measured configuration (table row format).
 */
struct GemmTuningEntry {
  uint64_t mBucket = 0, nBucket = 0, kBucket = 0; ///< Shape bucket, pow2.
  uint32_t dtypeTag = 0; ///< Input/accum pair tag, foldDtypeTag encoding.
  uint32_t archTag = 0;  ///< Architecture tag, DeviceCaps encoding.
  uint32_t tileM = 0, tileN = 0, tileK = 0; ///< Winning tile.
  uint8_t stages = 0;                       ///< Pipeline stages used.
  uint32_t numSplits = 0;                   ///< K splits used.
};

/**
 * @struct GemmTuningTable
 * @brief Optional empirical table; null runs analytic-only.
 */
struct GemmTuningTable {
  const GemmTuningEntry *entries = nullptr; ///< Rows, ascending bucket order.
  uint32_t numEntries = 0;                  ///< Row count.
};

/**
 * @struct GemmParams
 * @brief Facts for one GEMM launch.
 */
struct GemmParams : LaunchParamsBase {
  uint64_t M = 0, N = 0, K = 0;              ///< Matrix dimensions.
  DType_ inputDtype = Float32;               ///< Input element type.
  DType_ accumDtype = Float32;               ///< Accumulator type.
  bool aRowMajor = false, bRowMajor = false; ///< Input layouts.
  Epilogue epilogue;                         ///< Fused post work.
  uint8_t numStages = 0;                     ///< Pipeline stages, 0 for Auto.
  SplitK splitK = SplitK::Auto;
  CtaSchedule schedule = CtaSchedule::Auto;
  TraversalOrder traversal = TraversalOrder::Auto;
  WarpRole warpRole = WarpRole::Auto;
  const GemmTuningTable *tuningTable = nullptr;

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::GEMM;
  }
};

namespace detail {

/// Saturating FLOP count 2*M*N*K.
inline uint64_t gemmFlops(const GemmParams &p) noexcept {
  uint64_t mn = satMul(p.M, p.N);
  if (mn == ~0ULL) {
    return ~0ULL;
  }
  uint64_t mnk = satMul(mn, p.K);
  if (mnk > ~0ULL / 2U) {
    return ~0ULL;
  }
  return mnk * 2U;
}

/// Trivial epilogues fuse for free; anything else needs Full. Custom
/// is always non-trivial: unlisted work must never claim the cheap path.
constexpr bool gemmEpilogueTrivial(const Epilogue &e) noexcept {
  return !e.normalize && e.activation != ActivationKind::Softmax &&
         e.activation != ActivationKind::Gelu &&
         e.activation != ActivationKind::Silu &&
         e.activation != ActivationKind::Custom;
}

/// Table lookup by bucket triple plus dtype and arch tags; null when
/// no interpretable hit.
inline const GemmTuningEntry *gemmTableHit(const GemmParams &p) noexcept {
  if (p.tuningTable == nullptr || p.tuningTable->entries == nullptr ||
      p.tuningTable->numEntries == 0) {
    return nullptr;
  }
  const uint64_t mb = nextPow2(p.M);
  const uint64_t nb = nextPow2(p.N);
  const uint64_t kb = nextPow2(p.K);
  const uint32_t wantDtype = foldDtypeTag(static_cast<uint64_t>(p.inputDtype),
                                          static_cast<uint64_t>(p.accumDtype));
  const uint32_t wantArch = p.device.archVersion;
  for (uint32_t i = 0; i < p.tuningTable->numEntries; ++i) {
    const GemmTuningEntry &e = p.tuningTable->entries[i];
    if (e.mBucket == mb && e.nBucket == nb && e.kBucket == kb &&
        e.dtypeTag == wantDtype && e.archTag == wantArch) {
      return &e;
    }
  }
  return nullptr;
}

} // namespace detail

/// Total output elements.
inline uint64_t fastParallelWork(const GemmParams &p) noexcept {
  return detail::satMul(p.M, p.N);
}

/// 16x16 output tiles.
inline uint64_t fastOutputTiles(const GemmParams &p) noexcept {
  return detail::satMul(detail::ceilDiv(p.M, 16U), detail::ceilDiv(p.N, 16U));
}

/// Fused multiply-adds.
inline uint64_t fastComputeFlops(const GemmParams &p) noexcept {
  return detail::gemmFlops(p);
}

/// GEMM tiles stream coalesced by construction.
inline bool fastProvenCoalesced(const GemmParams &) noexcept { return true; }

/**
 * @brief True for tiny shapes with a trivial epilogue.
 */
inline bool coversGemmFast(const GemmParams &p) noexcept {
  if (p.M == 0 || p.N == 0 || p.K == 0) {
    return true;
  }
  return p.M <= GEMM_FAST_MAX_DIM && p.N <= GEMM_FAST_MAX_DIM &&
         p.K <= GEMM_FAST_MAX_DIM && detail::gemmEpilogueTrivial(p.epilogue);
}

/**
 * @brief Fixed 16-cubed tile, block per tile.
 */
inline LaunchConfig resolveGemmFast(const GemmParams &p) noexcept {
  if (p.M == 0 || p.N == 0 || p.K == 0) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t tiles =
      detail::satMul(detail::ceilDiv(p.M, 16U), detail::ceilDiv(p.N, 16U));
  const uint64_t cap =
      static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
  uint64_t blocks = tiles != 0 ? tiles : 1;
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  LaunchConfig cfg{};
  cfg.threads.x = threads >= 256 ? 16 : threads / 16;
  if (cfg.threads.x == 0) {
    cfg.threads.x = 1;
  }
  cfg.threads.y = threads / cfg.threads.x;
  if (cfg.threads.y == 0) {
    cfg.threads.y = 1;
  }
  cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

/**
 * @brief Analytic tiles, split-K refill, schedule.
 *
 * @details
 * Fixed 128x128 output tiles with K strips of 32. Past the idle-wave
 * threshold with large K it splits K to refill waves; tile counts
 * dwarfing the SMs switch to a persistent work queue. A forced StreamK
 * schedule takes the same SM-proportional fixed grid with fractional
 * K assignment left to the kernel. A tuning-table hit overrides tiles
 * and splits and takes the persistent schedule.
 * Pipeline stages, traversal order, and warp roles belong to kernel
 * selection downstream: geometry here only sizes the launch.
 */
inline LaunchConfig resolveGemm(const GemmParams &p) noexcept {
  if (p.M == 0 || p.N == 0 || p.K == 0) {
    return LaunchConfig{};
  }
  const GemmTuningEntry *hit = detail::gemmTableHit(p);
  uint32_t tileM = 128, tileN = 128;
  uint32_t splits = 1;
  bool persistent = false;
  if (hit != nullptr && hit->tileM != 0 && hit->tileN != 0) {
    tileM = hit->tileM;
    tileN = hit->tileN;
    splits = hit->numSplits != 0 ? hit->numSplits : splits;
    persistent = true;
  } else {
    uint64_t tiles = detail::satMul(detail::ceilDiv(p.M, tileM),
                                    detail::ceilDiv(p.N, tileN));
    const uint64_t sm = p.device.smCount;
    if (sm != 0 && p.K >= 1024 && tiles > 0) {
      const uint64_t waves = detail::ceilDiv(tiles, sm);
      const uint64_t last = tiles - ((waves - 1) * sm);
      if (p.splitK != SplitK::NoSplit && last * 4U < sm * 3U) {
        splits =
            static_cast<uint32_t>(detail::ceilDiv(sm, tiles > 0 ? tiles : 1U));
        splits = std::max<uint32_t>(splits, 2);
        splits = std::min<uint32_t>(splits, 8);
      }
    }
    if (p.splitK == SplitK::NoSplit) {
      splits = 1;
    }
    if (p.splitK == SplitK::SplitK && splits < 2) {
      splits = 2;
    }
    if ((p.schedule == CtaSchedule::Persistent ||
         (p.schedule == CtaSchedule::Auto && sm != 0 && tiles > sm * 8U)) &&
        p.schedule != CtaSchedule::BlockPerTile) {
      persistent = true;
    }
  }
  const uint32_t warp = detail::effectiveWarp(p.device);
  uint32_t threads = detail::fastThreads(p.device, p.kernel);
  threads = std::min<uint32_t>(threads, 256);
  threads = (threads / warp) * warp;
  if (threads == 0) {
    threads = warp;
  }
  uint64_t tiles =
      detail::satMul(detail::ceilDiv(p.M, tileM), detail::ceilDiv(p.N, tileN));
  // Forced StreamK sizes the same SM-proportional fixed grid as
  // persistent; fractional K assignment lives in the kernel.
  const bool fixedGrid = persistent || p.schedule == CtaSchedule::StreamK;
  uint64_t blocks = fixedGrid ? static_cast<uint64_t>(p.device.smCount)
                              : detail::satMul(tiles, splits);
  if (blocks == 0) {
    blocks = 1;
  }
  const uint64_t cap =
      fixedGrid ? 0U
                : static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
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
