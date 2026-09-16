/**
 * @file layout_transform.hh
 * @brief Launch heuristic for tiled layout kernels.
 */

#pragma once

#include <array>

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum BankStrategy
 * @brief Padding vs address swizzle for bank conflicts, Auto decides.
 */
enum class BankStrategy : uint8_t { Auto, Padding, Swizzle };

/// Maximum rank staged through shared memory.
inline constexpr uint8_t LAYOUT_MAX_RANK = 8;

/**
 * @struct LayoutShape
 * @brief Full shape plus strides on both sides, in elements.
 */
struct LayoutShape {
  uint8_t ndim = 0;                                  ///< Active rank, up to 8.
  std::array<int64_t, LAYOUT_MAX_RANK> shape{};      ///< Extents per dimension.
  std::array<int64_t, LAYOUT_MAX_RANK> inStrides{};  ///< Input strides.
  std::array<int64_t, LAYOUT_MAX_RANK> outStrides{}; ///< Output strides.
};

/**
 * @struct LayoutTransformParams
 * @brief Facts for one layout launch.
 */
struct LayoutTransformParams : LaunchParamsBase {
  LayoutShape layout;    ///< Shape and strides.
  uint32_t itemSize = 0; ///< Bytes per element.
  std::array<uint8_t, LAYOUT_MAX_RANK>
      perm{}; ///< Output dim i takes input dim perm[i]. ///< Output dim i takes input dim perm[i].
  BankStrategy bankStrategy = BankStrategy::Auto;

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::LayoutTransformation;
  }
};

namespace detail {

/// Total elements; non-positive extents degrade to zero.
inline uint64_t layoutElements(const LayoutTransformParams &p) noexcept {
  uint64_t n = 1;
  for (uint8_t i = 0; i < p.layout.ndim && i < LAYOUT_MAX_RANK; ++i) {
    if (p.layout.shape[i] <= 0) {
      return 0;
    }
    n = satMul(n, static_cast<uint64_t>(p.layout.shape[i]));
  }
  return p.layout.ndim == 0 ? 0 : n;
}

/// Permutation maps every source dim exactly once.
inline bool layoutPermValid(const LayoutTransformParams &p) noexcept {
  if (p.layout.ndim == 0 || p.layout.ndim > LAYOUT_MAX_RANK) {
    return false;
  }
  uint32_t seen = 0;
  for (uint32_t i = 0; i < p.layout.ndim; ++i) {
    if (p.perm[i] >= p.layout.ndim) {
      return false;
    }
    const uint32_t bit = 1U << p.perm[i];
    if ((seen & bit) != 0) {
      return false;
    }
    seen |= bit;
  }
  return true;
}

/// Identity permutation needs no staging; ranks past the staging
/// limit answer false.
inline bool layoutIsIdentity(const LayoutTransformParams &p) noexcept {
  if (p.layout.ndim > LAYOUT_MAX_RANK) {
    return false;
  }
  for (uint32_t i = 0; i < p.layout.ndim; ++i) {
    if (p.perm[i] != i) {
      return false;
    }
  }
  return true;
}

} // namespace detail

/// Total elements moved.
inline uint64_t fastParallelWork(const LayoutTransformParams &p) noexcept {
  return detail::layoutElements(p);
}

/// Bytes moved counting one read plus one write per element.
inline uint64_t fastTrafficBytes(const LayoutTransformParams &p) noexcept {
  return detail::satMul(detail::layoutElements(p),
                        detail::satMul(p.itemSize, 2U));
}

/**
 * @brief True for rank 1 or 2 with a valid permutation.
 */
inline bool coversLayoutTransformFast(const LayoutTransformParams &p) noexcept {
  return p.layout.ndim >= 1 && p.layout.ndim <= 2 && detail::layoutPermValid(p);
}

/**
 * @brief Streaming copy or fixed padded 32x32 tile.
 */
inline LaunchConfig
resolveLayoutTransformFast(const LayoutTransformParams &p) noexcept {
  const uint64_t n = detail::layoutElements(p);
  if (n == 0 || !detail::layoutPermValid(p)) {
    return LaunchConfig{};
  }
  const uint32_t threads = detail::fastThreads(p.device, p.kernel);
  LaunchConfig cfg{};
  if (detail::layoutIsIdentity(p)) {
    const uint64_t perThread =
        static_cast<uint64_t>(threads) * FAST_ELEMS_PER_THREAD;
    uint64_t blocks = detail::ceilDiv(n, perThread);
    const uint64_t cap =
        static_cast<uint64_t>(p.device.smCount) * FAST_BLOCKS_PER_SM;
    if (cap != 0 && blocks > cap) {
      blocks = cap;
    }
    cfg.threads.x = threads;
    cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
    if (cfg.blocks.x == 0) {
      cfg.blocks.x = 1;
    }
    return detail::clampConfig(cfg, p.device, p.kernel);
  }
  // Fixed 32-column tile: one padding column staged for bank safety.
  // Rows per block follow the fitted thread count, never a constant.
  const int64_t rows = p.layout.shape[0];
  const int64_t cols = p.layout.ndim > 1 ? p.layout.shape[1] : 1;
  cfg.threads.x = 32;
  cfg.threads.y = threads >= 256 ? 8 : threads / 32;
  if (cfg.threads.y == 0) {
    cfg.threads.y = 1;
  }
  const uint64_t rowsPerBlock = cfg.threads.y;
  const uint64_t ncols = cols > 0 ? static_cast<uint64_t>(cols) : 1U;
  const uint64_t nrows = rows > 0 ? static_cast<uint64_t>(rows) : 1U;
  const uint64_t bx = detail::ceilDiv(ncols, 32U);
  const uint64_t by = detail::ceilDiv(nrows, rowsPerBlock);
  cfg.blocks.x = static_cast<uint32_t>(bx > ~0U ? ~0U : bx);
  cfg.blocks.y = static_cast<uint32_t>(by > ~0U ? ~0U : by);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  if (cfg.blocks.y == 0) {
    cfg.blocks.y = 1;
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

/**
 * @brief Budgeted tile shortlist, bank-safe staging.
 *
 * @details
 * Picks the largest tile whose padded staging fits the block budget
 * and scales rows per thread from register headroom. Padding is the
 * only bank strategy implemented; traversal and hardware staging
 * belong to kernel selection downstream.
 */
inline LaunchConfig
resolveLayoutTransform(const LayoutTransformParams &p) noexcept {
  const uint64_t n = detail::layoutElements(p);
  if (n == 0 || !detail::layoutPermValid(p) || p.itemSize == 0) {
    return LaunchConfig{};
  }
  if (detail::layoutIsIdentity(p)) {
    return resolveLayoutTransformFast(p);
  }
  uint32_t tileX = 32;
  while (tileX > 8) {
    const uint64_t rowBytes = static_cast<uint64_t>(tileX) * p.itemSize;
    if (rowBytes == 32 || rowBytes == 64 || rowBytes == 128) {
      break;
    }
    tileX /= 2;
  }
  uint32_t tileY = 32;
  const uint64_t pad = p.itemSize;
  while (tileY > 8) {
    const uint64_t staged =
        (static_cast<uint64_t>(tileX + 1) * tileY * p.itemSize) + pad;
    if (p.device.smemPerBlock == 0 || staged <= p.device.smemPerBlock) {
      break;
    }
    tileY /= 2;
  }
  uint32_t rowsPerThread = 1;
  if (p.kernel.regsPerThread <= 32) {
    rowsPerThread = 4;
  } else if (p.kernel.regsPerThread <= 48) {
    rowsPerThread = 2;
  }
  uint64_t threads = static_cast<uint64_t>(tileX) * tileY / rowsPerThread;
  const uint32_t warp = detail::effectiveWarp(p.device);
  threads = (threads / warp) * warp;
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
  const int64_t dimA = p.layout.shape[0];
  const int64_t dimB = p.layout.ndim > 1 ? p.layout.shape[1] : 1;
  const uint64_t outA = dimA > 0 ? static_cast<uint64_t>(dimA) : 1U;
  const uint64_t outB = dimB > 0 ? static_cast<uint64_t>(dimB) : 1U;
  LaunchConfig cfg{};
  cfg.threads.x = tileX;
  cfg.threads.y = static_cast<uint32_t>(threads / tileX);
  if (cfg.threads.y == 0) {
    cfg.threads.y = 1;
  }
  const uint64_t bx = detail::ceilDiv(outB, tileX);
  const uint64_t by = detail::ceilDiv(outA, tileY);
  const uint64_t outer = p.layout.ndim > 2 && p.layout.shape[2] > 0
                             ? static_cast<uint64_t>(p.layout.shape[2])
                             : 1U;
  cfg.blocks.x = static_cast<uint32_t>(bx > ~0U ? ~0U : bx);
  cfg.blocks.y = static_cast<uint32_t>(by > ~0U ? ~0U : by);
  cfg.blocks.z = static_cast<uint32_t>(outer > ~0U ? ~0U : outer);
  if (cfg.blocks.x == 0) {
    cfg.blocks.x = 1;
  }
  if (cfg.blocks.y == 0) {
    cfg.blocks.y = 1;
  }
  return detail::clampConfig(cfg, p.device, p.kernel);
}

} // namespace ncore::heuristics::kernels
