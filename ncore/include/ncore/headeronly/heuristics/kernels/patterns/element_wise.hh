/**
 * @file element_wise.hh
 * @brief Launch heuristic for flat streaming element-wise kernels.
 */

#pragma once

#include "common.hh"

namespace ncore::heuristics::kernels {

/**
 * @enum IlpPreference
 * @brief Instruction-parallelism vs occupancy bias, Auto decides.
 */
enum class IlpPreference : uint8_t { Auto, PreferIlp, PreferOccupancy };

/**
 * @enum TailStrategy
 * @brief Remainder handling, Auto decides.
 */
enum class TailStrategy : uint8_t { Auto, SeparateBranch, Masked };

/**
 * @struct ElementWiseParams
 * @brief Facts for one element-wise launch.
 */
struct ElementWiseParams : LaunchParamsBase {
  uint64_t numElements = 0;         ///< Elements to process.
  uint32_t inputItemSize = 0;       ///< Input bytes per element.
  uint32_t outputItemSize = 0;      ///< Output bytes per element.
  uint32_t inputAlignBytes = 0;     ///< Proven input alignment.
  uint32_t outputAlignBytes = 0;    ///< Proven output alignment.
  uint32_t packedElemsPerUnit = 0;  ///< Packing unit, 1 when N/A.
  bool inputContiguous = false;     ///< Input stride-free.
  bool outputContiguous = false;    ///< Output stride-free.
  uint32_t numInputs = 0;           ///< Distinct input tensors.
  float arithmeticIntensity = 0.0F; ///< FLOPs per byte moved.
  bool reuseAfter = false;          ///< Output reread by the next kernel.
  IlpPreference ilpPreference = IlpPreference::Auto;
  TailStrategy tailStrategy = TailStrategy::Auto;

  [[nodiscard]] ExecutionPattern pattern() const noexcept override {
    return ExecutionPattern::ElementWise;
  }
};

/**
 * @struct ElementWisePlan
 * @brief Geometry plus the codegen facts the kernel needs.
 */
struct ElementWisePlan {
  LaunchConfig config;        ///< Grid/block pair.
  uint32_t vectorWidthElems;  ///< Elements per vector transaction.
  uint32_t elementsPerThread; ///< Inner-loop coarsening.
  TailStrategy tailUsed;      ///< Remainder handling selected.
};

namespace detail {

/// Largest vector width proven by sizes, alignment, contiguity, packing.
inline uint32_t elementWiseWidth(const ElementWiseParams &p) noexcept {
  if (!p.inputContiguous || !p.outputContiguous) {
    return 1;
  }
  const uint32_t item =
      p.inputItemSize > p.outputItemSize ? p.inputItemSize : p.outputItemSize;
  if (item == 0) {
    return 1;
  }
  uint32_t width = 16 / item;
  if (width == 0) {
    return 1;
  }
  while (width > 1) {
    const uint32_t bytes = width * item;
    const bool fitsTxn = bytes == 8 || bytes == 16;
    if (fitsTxn && p.inputAlignBytes % bytes == 0 &&
        p.outputAlignBytes % bytes == 0 &&
        (p.packedElemsPerUnit <= 1 || width % p.packedElemsPerUnit == 0)) {
      return width;
    }
    width /= 2;
  }
  return 1;
}

/// Bandwidth-bound test against the device roofline knee.
inline bool elementWiseBandwidthBound(const ElementWiseParams &p) noexcept {
  if (p.device.memBandwidthBytesPerSec == 0 || p.device.peakFlops == 0) {
    return true;
  }
  const double knee = static_cast<double>(p.device.peakFlops) /
                      static_cast<double>(p.device.memBandwidthBytesPerSec);
  return static_cast<double>(p.arithmeticIntensity) < knee;
}

} // namespace detail

/// Total elements to cover.
inline uint64_t fastParallelWork(const ElementWiseParams &p) noexcept {
  return p.numElements;
}

/// Bytes moved counting every input plus every output.
inline uint64_t fastTrafficBytes(const ElementWiseParams &p) noexcept {
  return detail::satMul(p.numElements, static_cast<uint64_t>(p.inputItemSize) +
                                           p.outputItemSize);
}

/// Whether vector width follows from alignment and contiguity.
inline bool fastProvenCoalesced(const ElementWiseParams &p) noexcept {
  return p.inputContiguous && p.outputContiguous && p.inputAlignBytes >= 16 &&
         p.outputAlignBytes >= 16;
}

/**
 * @brief Always true: every shape fits the cheap path.
 */
inline bool coversElementWiseFast(const ElementWiseParams &) noexcept {
  return true;
}

/**
 * @brief Fixed coarsening, capped grid, masked tail.
 */
inline LaunchConfig
resolveElementWiseFast(const ElementWiseParams &p) noexcept {
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
  if (blocks == 0) {
    blocks = 1;
  }
  LaunchConfig cfg{};
  cfg.threads.x = threads;
  cfg.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  return detail::clampConfig(cfg, p.device, p.kernel);
}

/**
 * @brief Vector width, roofline coarsening, wave-sized grid.
 */
inline ElementWisePlan
resolveElementWiseFullPlan(const ElementWiseParams &p) noexcept {
  ElementWisePlan plan{};
  if (p.numElements == 0) {
    return plan;
  }
  plan.vectorWidthElems = detail::elementWiseWidth(p);
  const bool bandwidthBound = detail::elementWiseBandwidthBound(p);
  bool wantIlp = bandwidthBound;
  if (p.ilpPreference == IlpPreference::PreferIlp) {
    wantIlp = true;
  } else if (p.ilpPreference == IlpPreference::PreferOccupancy) {
    wantIlp = false;
  }
  plan.elementsPerThread =
      wantIlp ? plan.vectorWidthElems * 4 : plan.vectorWidthElems;
  if (plan.elementsPerThread == 0) {
    plan.elementsPerThread = 1;
  }
  if (p.numInputs > 4 && plan.elementsPerThread > plan.vectorWidthElems) {
    plan.elementsPerThread = plan.vectorWidthElems;
  }
  const uint32_t warp = detail::effectiveWarp(p.device);
  uint32_t maxT = p.device.maxThreadsPerBlockDevice;
  if (p.kernel.maxThreadsPerBlockKernel != 0 &&
      (maxT == 0 || p.kernel.maxThreadsPerBlockKernel < maxT)) {
    maxT = p.kernel.maxThreadsPerBlockKernel;
  }
  const uint64_t need = detail::ceilDiv(p.numElements, plan.elementsPerThread);
  const uint64_t want = need;
  // Unknown caps still need a sane ceiling; known caps behave as before.
  uint32_t ceiling = maxT;
  if (ceiling == 0) {
    ceiling = detail::fastThreads(p.device, p.kernel);
  }
  uint32_t threads =
      want > ceiling ? ceiling : static_cast<uint32_t>(want > ~0U ? ~0U : want);
  threads = (threads / warp) * warp;
  if (threads == 0) {
    threads = warp <= ceiling ? warp : ceiling;
  }
  uint64_t blocks = detail::ceilDiv(
      p.numElements, static_cast<uint64_t>(threads) * plan.elementsPerThread);
  const uint64_t cap = static_cast<uint64_t>(p.device.smCount) * 4U;
  if (cap != 0 && blocks > cap) {
    blocks = cap;
  }
  if (blocks == 0) {
    blocks = 1;
  }
  const uint64_t stride =
      static_cast<uint64_t>(threads) * plan.elementsPerThread;
  if (p.tailStrategy == TailStrategy::SeparateBranch ||
      (p.tailStrategy == TailStrategy::Auto &&
       p.numElements % stride >=
           static_cast<uint64_t>(threads) * plan.vectorWidthElems)) {
    plan.tailUsed = TailStrategy::SeparateBranch;
  } else {
    plan.tailUsed = TailStrategy::Masked;
  }
  plan.config.threads.x = threads;
  plan.config.blocks.x = static_cast<uint32_t>(blocks > ~0U ? ~0U : blocks);
  plan.config = detail::clampConfig(plan.config, p.device, p.kernel);
  return plan;
}

/**
 * @brief Plan reduced to its grid/block pair.
 */
inline LaunchConfig resolveElementWise(const ElementWiseParams &p) noexcept {
  return resolveElementWiseFullPlan(p).config;
}

} // namespace ncore::heuristics::kernels
