/**
 * @file Caps.hpp
 * @brief Stub device caps and kernel attrs for the launch-heuristics suites.
 *
 * The fixture mirrors a small 8-SM device; a case needing other limits
 * overrides the returned struct.
 */

#pragma once

#include <cstdint>

#include <ncore/headeronly/heuristics/kernels/launch_config.hh>

namespace hk = ncore::heuristics::kernels;

namespace tests::heuristics {

/// @brief Caps fixture mirroring a small 8-SM device.
inline hk::DeviceCaps testCaps() {
  ncore::heuristics::kernels::DeviceCaps d{};
  d.warpSize = 32U;
  d.maxThreadsPerBlockDevice = 1024U;
  d.maxThreadsPerSM = 2048U;
  d.maxBlocksPerSM = 32U;
  d.smCount = 8U;
  d.smemPerBlock = 49152U;
  d.smemPerSM = 166912U;
  d.regsPerSM = 65536U;
  d.regsAllocGranularity = 8U;
  d.maxGridX = 2147483647U;
  d.maxGridY = 65535U;
  d.maxGridZ = 65535U;
  d.archVersion = 90U;
  d.memBandwidthBytesPerSec = 900000000000ULL;
  d.peakFlops = 100000000000000ULL;
  return d;
}

/// @brief Kernel attrs fixture; @p regs relaxes the register budget.
inline hk::KernelAttrs testKernel(uint32_t regs = 16U) {
  hk::KernelAttrs k{};
  k.regsPerThread = regs;
  k.maxThreadsPerBlockKernel = 1024U;
  return k;
}

} // namespace tests::heuristics
