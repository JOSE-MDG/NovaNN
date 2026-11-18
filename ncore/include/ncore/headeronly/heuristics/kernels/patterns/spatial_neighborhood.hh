/**
 * @file spatial_neighborhood.hh
 * @brief Launch heuristic for stencil, convolution, and pooling.
 */

#pragma once

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum BorderStrategy
 * @brief Clamp once at load vs predicated in the loop, Auto decides.
 */
enum class BorderStrategy : uint8_t { Auto, AtLoad, InLoop };

/// Fixed Fast output tile edge.
inline constexpr uint32_t SPATIAL_FAST_TILE = 16;

/**
 * @struct SpatialNeighborhoodParams
 * @brief Facts for one spatial launch.
 */
struct SpatialNeighborhoodParams : LaunchParamsBase {
  uint32_t filterH = 0, filterW = 0, filterD = 0; ///< Filter extents, D=1.
  uint64_t inH = 0, inW = 0, inD = 0;             ///< Input extents, D=1.
  uint64_t outH = 0, outW = 0, outD = 0;    ///< Output extents, 0 derives.
  uint32_t inChannels = 0, outChannels = 0; ///< Channel counts.
  uint32_t itemSize = 0;                    ///< Bytes per element.
  uint32_t strideH = 0, strideW = 0, strideD = 0;       ///< Strides.
  uint32_t dilationH = 0, dilationW = 0, dilationD = 0; ///< Dilations.
  bool filterConstEligible = false;      ///< Filter fits read-only.
  bool allowTensorReformulation = false; ///< Matmul rewrite permitted.
  uint32_t channelsPerThread = 0;        ///< Channels/thread, 0 Auto.
  BorderStrategy border = BorderStrategy::Auto;

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::SpatialNeighborhood;
  }
};

namespace detail {

/**
 * @brief Output extent along one axis: explicit value or SAME-convention.
 *
 * @details
 * An explicit output wins because the caller knows the real geometry.
 * Otherwise the SAME convention floors the padded input by the stride,
 * with a zero stride meaning one.
 */
inline uint64_t spatialOutExtent(uint64_t in, uint32_t stride,
                                 uint64_t explicitOut) noexcept {
  if (explicitOut != 0) {
    return explicitOut;
  }
  return ceilDiv(in, stride != 0 ? stride : 1U);
}

/// Output elements over H/W/D.
inline uint64_t spatialOutputs(const SpatialNeighborhoodParams &p) noexcept {
  const uint64_t oh = spatialOutExtent(p.inH, p.strideH, p.outH);
  const uint64_t ow = spatialOutExtent(p.inW, p.strideW, p.outW);
  const uint64_t od = spatialOutExtent(p.inD, p.strideD, p.outD);
  return satMul(satMul(satMul(oh, ow), od), p.outChannels);
}

/// Effective filter span along one axis; zero filter counts as one.
constexpr uint64_t filterSpan(uint32_t f, uint32_t d) noexcept {
  if (f == 0) {
    return 1;
  }
  return (static_cast<uint64_t>(f - 1) * (d != 0 ? d : 1U)) + 1U;
}

} // namespace detail

/// Total output elements.
inline uint64_t fastParallelWork(const SpatialNeighborhoodParams &p) noexcept {
  return detail::spatialOutputs(p);
}

/// Bytes moved counting tile reads plus output writes.
inline uint64_t fastTrafficBytes(const SpatialNeighborhoodParams &p) noexcept {
  const uint64_t span =
      detail::satMul(detail::filterSpan(p.filterH, p.dilationH),
                     detail::filterSpan(p.filterW, p.dilationW));
  return detail::satMul(detail::spatialOutputs(p),
                        detail::satMul(p.itemSize, detail::satAdd(span, 1U)));
}

/// Output tiles over the volume.
inline uint64_t fastOutputTiles(const SpatialNeighborhoodParams &p) noexcept {
  const uint64_t tile =
      static_cast<uint64_t>(SPATIAL_FAST_TILE) * SPATIAL_FAST_TILE;
  const uint64_t out =
      detail::spatialOutputs(p) / (p.outChannels != 0 ? p.outChannels : 1U);
  return detail::ceilDiv(out, tile);
}

/**
 * @brief True for unit stride/dilation without reformulation.
 */
inline bool
coversSpatialNeighborhoodFast(const SpatialNeighborhoodParams &p) noexcept {
  if (p.inH == 0 || p.inW == 0 || p.inD == 0 || p.outChannels == 0) {
    return true;
  }
  return p.strideH == 1 && p.strideW == 1 && p.strideD == 1 &&
         p.dilationH == 1 && p.dilationW == 1 && p.dilationD == 1 &&
         !p.allowTensorReformulation;
}

/**
 * @brief Fixed 16x16 tiles, border at load.
 */
inline LaunchConfig
resolveSpatialNeighborhoodFast(const SpatialNeighborhoodParams &p) noexcept {
  if (p.inH == 0 || p.inW == 0 || p.inD == 0 || p.outChannels == 0) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  const uint64_t tiles = detail::satMul(fastOutputTiles(p), p.outChannels);
  const uint64_t cap = fastGridCap(p.device, p.kernel, threads);
  uint64_t blocks = tiles != 0 ? tiles : 1;
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  LaunchConfig cfg{};
  cfg.threads.x = SPATIAL_FAST_TILE;
  cfg.threads.y = threads >= 256 ? SPATIAL_FAST_TILE : threads / 16;
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
 * @brief Scored tiles, channel packing, border mode.
 *
 * @details
 * Candidate output tiles score useful payload over loaded bytes, times
 * the modeled occupancy and the fill of the last resident wave. A tile
 * whose halo-inflated staging exceeds the block budget is skipped; if
 * every tile is skipped, tile 8 is returned and the kernel must split
 * channels to stage, since no tile fits as a whole. Channels pack per
 * thread while registers allow. Borders clamp once at load unless most
 * blocks touch them. A dense small-filter many-channel convolution on
 * tensor hardware defers to GEMM.
 */
inline LaunchConfig
resolveSpatialNeighborhood(const SpatialNeighborhoodParams &p) noexcept {
  if (p.inH == 0 || p.inW == 0 || p.inD == 0 || p.outChannels == 0 ||
      p.itemSize == 0) {
    return LaunchConfig{};
  }
  const uint64_t spanH = detail::filterSpan(p.filterH, p.dilationH);
  const uint64_t spanW = detail::filterSpan(p.filterW, p.dilationW);
  const uint32_t warp = detail::effectiveWarp(p.device);
  uint32_t cpt = p.channelsPerThread;
  if (cpt == 0) {
    cpt = (p.inChannels >= 4 && p.kernel.regsPerThread <= 32) ? 4 : 1;
  }
  const uint64_t outH = detail::spatialOutExtent(p.inH, p.strideH, p.outH);
  const uint64_t outW = detail::spatialOutExtent(p.inW, p.strideW, p.outW);
  const uint64_t outD = detail::spatialOutExtent(p.inD, p.strideD, p.outD);
  const uint64_t channelGroups = detail::ceilDiv(p.outChannels, cpt);
  // Warp-rounded threads for a tile at the chosen channels per thread.
  const auto tileThreads = [&](uint32_t tile) noexcept -> uint32_t {
    const uint64_t raw = static_cast<uint64_t>(tile) * tile / cpt;
    const uint64_t rounded = detail::ceilDiv(raw, warp) * warp;
    if (rounded == 0) {
      return warp;
    }
    return static_cast<uint32_t>(rounded > ~0U ? ~0U : rounded);
  };
  uint32_t bestTile = 8;
  uint32_t bestThreads = 0;
  double bestScore = 0.0;
  for (uint32_t tile = 8; tile <= 32; tile *= 2) {
    const uint64_t staged = detail::satMul(
        detail::satMul(detail::satMul(tile + spanH, tile + spanW), p.itemSize),
        p.inChannels);
    if (p.device.smemPerBlock != 0 && staged > p.device.smemPerBlock) {
      continue;
    }
    const uint32_t tt = tileThreads(tile);
    KernelAttrs kk = p.kernel;
    kk.dynSmemPerBlock = detail::satAdd(kk.dynSmemPerBlock, staged);
    const Occupancy occ = computeOccupancy(p.device, kk, tt);
    const uint64_t tilesCand = detail::satMul(
        detail::satMul(detail::ceilDiv(outH, tile),
                       detail::ceilDiv(outW, tile)),
        detail::satMul(detail::ceilDiv(outD, tile), channelGroups));
    double fill = 1.0;
    if (p.device.smCount != 0) {
      const uint64_t perSM = occ.blocksPerSM != 0 ? occ.blocksPerSM : 1U;
      const uint64_t resident = detail::satMul(perSM, p.device.smCount);
      if (resident != 0) {
        const double f =
            static_cast<double>(tilesCand) / static_cast<double>(resident);
        fill = f < 1.0 ? f : 1.0;
      }
    }
    const double useful =
        staged != 0 ? static_cast<double>(static_cast<uint64_t>(tile) * tile *
                                          p.itemSize) /
                          static_cast<double>(staged)
                    : 0.0;
    const double occFactor =
        occ.known() ? static_cast<double>(occ.fraction) : 1.0;
    const double score = useful * occFactor * fill;
    if (score >= bestScore) {
      bestScore = score;
      bestTile = tile;
      bestThreads = tt;
    }
  }
  uint32_t threads = bestThreads != 0 ? bestThreads : tileThreads(bestTile);
  uint32_t maxT = p.device.maxThreadsPerBlockDevice;
  if (p.kernel.maxThreadsPerBlockKernel != 0 &&
      (maxT == 0 || p.kernel.maxThreadsPerBlockKernel < maxT)) {
    maxT = p.kernel.maxThreadsPerBlockKernel;
  }
  if (maxT != 0 && threads > maxT) {
    threads = (maxT / warp) * warp;
  }
  if (threads == 0) {
    threads = warp;
  }
  const uint64_t tilesH = detail::ceilDiv(outH, bestTile);
  const uint64_t tilesW = detail::ceilDiv(outW, bestTile);
  const uint64_t tilesD = detail::ceilDiv(outD, bestTile);
  uint64_t blocks =
      detail::satMul(detail::satMul(tilesH, tilesW), channelGroups);
  if (blocks == 0) {
    blocks = 1;
  }
  const uint64_t cap = fastGridCap(p.device, p.kernel, threads);
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  LaunchConfig cfg{};
  cfg.threads.x = bestTile;
  cfg.threads.y = threads / bestTile;
  if (cfg.threads.y == 0) {
    cfg.threads.y = 1;
  }
  cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  cfg.blocks.z =
      static_cast<uint32_t>(tilesD > 1 ? (tilesD > ~0U ? ~0U : tilesD) : 1U);
  return detail::clampConfig(cfg, p.device, p.kernel);
}

} // namespace ncore::heuristics::kernels
