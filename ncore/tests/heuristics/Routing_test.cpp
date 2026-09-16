/**
 * @file Routing_test.cpp
 * @brief Unit tests for tier routing, shared clamps, and the Full cache.
 *
 * All cases run host-side with stub device caps; no GPU is required.
 * Caps fixtures mirror a small 8-SM device unless a case needs
 * otherwise.
 */

#include <array>
#include <bit>
#include <cstdint>

#include <gtest/gtest.h>

#include <ncore/core/dtype.h>
#include <ncore/headeronly/heuristics/kernels/launch_config.hh>

namespace hk = ncore::heuristics::kernels;

namespace {

hk::DeviceCaps testCaps() {
  hk::DeviceCaps d{};
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

hk::KernelAttrs testKernel(uint32_t regs = 16U) {
  hk::KernelAttrs k{};
  k.regsPerThread = regs;
  k.maxThreadsPerBlockKernel = 1024U;
  return k;
}

hk::ElementWiseParams testElementWise(uint64_t n) {
  hk::ElementWiseParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = n;
  p.inputItemSize = 4U;
  p.outputItemSize = 4U;
  p.inputAlignBytes = 16U;
  p.outputAlignBytes = 16U;
  p.packedElemsPerUnit = 1U;
  p.inputContiguous = true;
  p.outputContiguous = true;
  p.numInputs = 1U;
  p.arithmeticIntensity = 0.08F;
  return p;
}

} // namespace

/**
 * @brief Tiny GEMM routes Fast: sub-wave work, launch-dominated time.
 * @test 8x4x2 must never pay occupancy or table costs.
 */
TEST(TierRouting, Matmul8x4x2RoutesFast) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 8U;
  p.N = 4U;
  p.K = 2U;
  p.inputDtype = DType_::Float32;
  p.accumDtype = DType_::Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;

  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.blocks.x, 1U);
  EXPECT_EQ(cfg.threads.x, 16U);
  EXPECT_EQ(cfg.threads.y, 16U);
}

/**
 * @brief Huge row reduction routes Full via the coverage gate.
 * @test 1M-extent axis cannot fit one block, so Fast is unavailable.
 */
TEST(TierRouting, ReduceSumHugeRoutesFull) {
  hk::ReductionParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.reduceExtent = 1024ULL * 1024ULL;
  p.numReductions = 10000ULL;
  p.reduceContiguous = true;
  p.op = hk::ReductionOp::Add;
  p.inputItemSize = 4U;
  p.accumItemSize = 4U;

  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief Just under one wave routes Fast on size alone.
 */
TEST(TierRouting, JustUnderOneWaveRoutesFast) {
  auto p = testElementWise((8ULL * 2048ULL) - 1ULL);
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
}

/**
 * @brief Past one wave with starved bandwidth routes Full.
 * @test R1 fails and the roofline estimate exceeds the launch budget.
 */
TEST(TierRouting, PastWaveStarvedBandwidthRoutesFull) {
  auto p = testElementWise(8ULL * 2048ULL);
  p.device.memBandwidthBytesPerSec = 1ULL;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief Isolated mid-size launch stays Fast; captured goes Full.
 * @test Capture amortizes one Full evaluation over free replays.
 */
TEST(TierRouting, CaptureFlipsMidSizeToFull) {
  auto p = testElementWise(100000ULL);
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  p.context.inGraphCapture = true;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief Coverage failure vetoes Fast even for tiny work.
 * @test 8K strided-free axis still exceeds one block.
 */
TEST(TierRouting, CoverageFailVetoesSmallWork) {
  hk::ReductionParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.reduceExtent = 8192ULL;
  p.numReductions = 1ULL;
  p.reduceContiguous = true;
  p.op = hk::ReductionOp::Add;
  p.inputItemSize = 4U;
  p.accumItemSize = 4U;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief 100-thread 1D count rounds down to whole warps.
 */
TEST(LaunchClamps, WarpRounding) {
  hk::LaunchConfig cfg{};
  cfg.threads.x = 100U;
  cfg.blocks.x = 4U;
  const hk::LaunchConfig out =
      hk::detail::clampConfig(cfg, testCaps(), testKernel());
  EXPECT_EQ(out.threads.x, 96U);
  EXPECT_EQ(out.blocks.x, 4U);
}

/**
 * @brief Zero threads degrade to the all-ones floor, never illegal.
 */
TEST(LaunchClamps, ZeroDegradesToFloor) {
  hk::LaunchConfig cfg{};
  cfg.threads.x = 0U;
  cfg.blocks.x = 0U;
  const hk::LaunchConfig out =
      hk::detail::clampConfig(cfg, testCaps(), testKernel());
  EXPECT_EQ(out.threads.x, 1U);
  EXPECT_EQ(out.threads.y, 1U);
  EXPECT_EQ(out.threads.z, 1U);
  EXPECT_EQ(out.blocks.x, 1U);
  EXPECT_EQ(out.blocks.y, 1U);
  EXPECT_EQ(out.blocks.z, 1U);
}

/**
 * @brief Grid caps at the per-axis device limit.
 */
TEST(LaunchClamps, GridAxisCap) {
  hk::DeviceCaps caps = testCaps();
  caps.maxGridX = 16U;
  hk::LaunchConfig cfg{};
  cfg.threads.x = 256U;
  cfg.blocks.x = 1000000U;
  const hk::LaunchConfig out = hk::detail::clampConfig(cfg, caps, testKernel());
  EXPECT_EQ(out.blocks.x, 16U);
}

/**
 * @brief Thread caps apply to multi-dim blocks, tile width kept.
 * @test (300,2) vs 512 shrinks rows; (64,8,4) vs 1024 halves depth;
 * a lone over-wide column collapses to warp-floored 1-D.
 */
TEST(LaunchClamps, MultiDimThreadCap) {
  hk::DeviceCaps caps = testCaps();
  hk::KernelAttrs kernel = testKernel();
  kernel.maxThreadsPerBlockKernel = 512U;

  hk::LaunchConfig two{};
  two.threads.x = 300U;
  two.threads.y = 2U;
  two.blocks.x = 4U;
  const hk::LaunchConfig out2 = hk::detail::clampConfig(two, caps, kernel);
  EXPECT_EQ(out2.threads.x, 300U);
  EXPECT_EQ(out2.threads.y, 1U);
  EXPECT_EQ(out2.threads.z, 1U);

  hk::LaunchConfig wide{};
  wide.threads.x = 1024U;
  wide.threads.y = 2U;
  wide.blocks.x = 4U;
  const hk::LaunchConfig outw =
      hk::detail::clampConfig(wide, testCaps(), testKernel());
  EXPECT_EQ(outw.threads.x, 1024U);
  EXPECT_EQ(outw.threads.y, 1U);

  hk::LaunchConfig three{};
  three.threads.x = 64U;
  three.threads.y = 8U;
  three.threads.z = 4U;
  three.blocks.x = 4U;
  const hk::LaunchConfig out3 =
      hk::detail::clampConfig(three, testCaps(), testKernel());
  EXPECT_EQ(out3.threads.x, 64U);
  EXPECT_EQ(out3.threads.y, 8U);
  EXPECT_EQ(out3.threads.z, 2U);

  hk::LaunchConfig col{};
  col.threads.x = 2048U;
  col.threads.y = 4U;
  col.blocks.x = 4U;
  const hk::LaunchConfig outc = hk::detail::clampConfig(col, caps, kernel);
  EXPECT_EQ(outc.threads.x, 512U);
  EXPECT_EQ(outc.threads.y, 1U);
  EXPECT_EQ(outc.threads.z, 1U);
}

/**
 * @brief 33 regs/thread steps Fast threads down two warp multiples.
 * @test Per-lane quantum 8: 33 rounds to 40 lanes worth, 8 warps need
 * 10240 regs > 8192; 6 warps fit.
 */
TEST(FastFitDown, Regs33StepsDownTwoWarps) {
  hk::DeviceCaps caps = testCaps();
  caps.regsPerSM = 8192U;
  EXPECT_EQ(hk::detail::fastThreads(caps, testKernel(33U)), 192U);
  EXPECT_EQ(hk::detail::fastThreads(caps, testKernel(32U)), 256U);
}

/**
 * @brief Empty launch returns the floor through the entry point.
 */
TEST(LaunchFloor, EmptyElementWise) {
  auto p = testElementWise(0ULL);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 1U);
  EXPECT_EQ(cfg.blocks.x, 1U);
}

/**
 * @brief Preloaded cache entry wins over the resolver.
 * @test Hit path returns stored geometry without recomputation.
 */
TEST(HeuristicCache, PreloadedHitReturnsStored) {
  hk::ReductionParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.reduceExtent = 1024ULL * 1024ULL;
  p.numReductions = 4ULL;
  p.reduceContiguous = true;
  p.op = hk::ReductionOp::Add;
  p.inputItemSize = 4U;
  p.accumItemSize = 4U;
  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);

  const hk::FullCacheKey key = hk::detail::fullKeyFor(p, p.device.archVersion);

  hk::FullConfigCache cache{};
  hk::LaunchConfig stored{};
  stored.threads.x = 64U;
  stored.blocks.x = 3U;
  cache.store(key, stored);

  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p, &cache);
  EXPECT_EQ(cfg.threads.x, 64U);
  EXPECT_EQ(cfg.blocks.x, 3U);
}

/**
 * @brief Keys separate launches differing only in input precision.
 * @test Same shape and output size, 2B vs 8B input: tags differ.
 */
TEST(HeuristicCache, KeysSeparateInputPrecision) {
  hk::ReductionParams a{};
  a.device = testCaps();
  a.kernel = testKernel();
  a.reduceExtent = 1024ULL * 1024ULL;
  a.numReductions = 4ULL;
  a.inputItemSize = 2U;
  a.accumItemSize = 4U;

  hk::ReductionParams b = a;
  b.inputItemSize = 8U;

  const hk::FullCacheKey ka = hk::detail::fullKeyFor(a, a.device.archVersion);
  const hk::FullCacheKey kb = hk::detail::fullKeyFor(b, b.device.archVersion);
  EXPECT_NE(ka.dtypeTag, kb.dtypeTag);
}

/**
 * @brief Miss computes inline and becomes a hit afterwards.
 */
TEST(HeuristicCache, MissStoresForNextCall) {
  auto p = testElementWise(1ULL << 30U);
  p.arithmeticIntensity = 100.0F;
  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);

  hk::FullConfigCache cache{};
  const hk::LaunchConfig first = hk::resolveLaunchConfig(p, &cache);
  const hk::LaunchConfig second = hk::resolveLaunchConfig(p, &cache);
  EXPECT_EQ(first.threads.x, second.threads.x);
  EXPECT_EQ(first.blocks.x, second.blocks.x);
  EXPECT_EQ(first.blocks.x, 32U);
}

/**
 * @brief Fused keys separate stage shapes under one pattern.
 * @test Same single-ElementWise pattern, 64 vs 1M elements: buckets differ.
 */
TEST(HeuristicCache, FusedKeysSeparateStageShapes) {
  auto smallStage = testElementWise(64ULL);
  auto bigStage = testElementWise(1ULL << 20U);

  hk::FusedOperatorParams a{};
  a.device = testCaps();
  a.kernel = testKernel();
  a.stages[0].pattern = hk::ExecutionPattern::ElementWise;
  a.stages[0].params = &smallStage;
  a.numStages = 1U;

  hk::FusedOperatorParams b = a;
  b.stages[0].params = &bigStage;

  const hk::FullCacheKey ka = hk::detail::fullKeyFor(a, a.device.archVersion);
  const hk::FullCacheKey kb = hk::detail::fullKeyFor(b, b.device.archVersion);
  EXPECT_NE(ka.shapeBucket, kb.shapeBucket);
}

/**
 * @brief Hand-rolled nextPow2 matches std::bit_ceil on valid input
 * and saturates past 2^63 where bit_ceil is undefined.
 */
TEST(DetailMath, NextPow2MatchesBitCeil) {

  const std::array<uint64_t, 8> cases = {
      0ULL, 1ULL, 2ULL, 3ULL, 5ULL, 1000ULL, 1ULL << 32U, 1ULL << 63U};
  for (uint64_t v : cases) {
    EXPECT_EQ(hk::detail::nextPow2(v), std::bit_ceil(v)) << "v=" << v;
  }
  EXPECT_EQ(hk::detail::nextPow2(~0ULL), ~0ULL);
  EXPECT_EQ(hk::detail::ceilDiv(~0ULL, 1024ULL), ((~0ULL) / 1024ULL) + 1ULL);
}

/**
 * @brief Unknown wavefront width falls back to 32: exact on NVIDIA,
 * native HIP default on RDNA, launchable everywhere.
 */
TEST(DetailMath, UnknownWarpFallsBackTo32) {
  EXPECT_EQ(hk::FALLBACK_WARP_SIZE, 32U);

  hk::DeviceCaps caps = testCaps();
  caps.warpSize = 0U;
  caps.regsPerSM = 0U;
  caps.maxThreadsPerBlockDevice = 0U;
  hk::KernelAttrs kernel{};
  EXPECT_EQ(hk::detail::fastThreads(caps, kernel), 256U);

  hk::LaunchConfig cfg{};
  cfg.threads.x = 100U;
  cfg.blocks.x = 2U;
  const hk::LaunchConfig out = hk::detail::clampConfig(cfg, caps, kernel);
  EXPECT_EQ(out.threads.x, 96U);
}

/**
 * @brief Ranks past the staging limit never read as identity.
 * @test ndim 200 with an identity prefix answers false; rank 2 holds true.
 */
TEST(DetailMath, OversizedRankNotIdentity) {
  hk::LayoutTransformParams p{};
  p.layout.ndim = 200U;
  for (uint32_t i = 0; i < 8U; ++i) {
    p.perm[i] = static_cast<uint8_t>(i);
  }
  EXPECT_FALSE(hk::detail::layoutIsIdentity(p));

  hk::LayoutTransformParams q{};
  q.layout.ndim = 2U;
  q.perm[0] = 0U;
  q.perm[1] = 1U;
  EXPECT_TRUE(hk::detail::layoutIsIdentity(q));
}

/**
 * @brief Sub-warp ceilings return the ceiling for the clamp to enforce.
 */
TEST(DetailMath, FastThreadsHonorsSubWarpCap) {
  hk::DeviceCaps caps = testCaps();
  caps.maxThreadsPerBlockDevice = 16U;
  EXPECT_EQ(hk::detail::fastThreads(caps, testKernel()), 16U);
}

/**
 * @brief Span products saturate instead of wrapping on huge extents.
 * @test filterSpan itself is exact for any u32 pair; the area product
 * in fastTrafficBytes saturates to ~0.
 */
TEST(DetailMath, FilterSpanSaturates) {
  EXPECT_EQ(hk::detail::filterSpan(0U, 7U), 1ULL);
  EXPECT_EQ(hk::detail::filterSpan(3U, 2U), 5ULL);
  EXPECT_EQ(hk::detail::filterSpan(0xFFFFFFFFU, 0xFFFFFFFFU),
            (0xFFFFFFFEULL * 0xFFFFFFFFULL) + 1ULL);

  hk::SpatialNeighborhoodParams p{};
  p.filterH = 0xFFFFFFFFU;
  p.filterW = 0xFFFFFFFFU;
  p.dilationH = 0xFFFFFFFFU;
  p.dilationW = 0xFFFFFFFFU;
  p.inH = 64ULL;
  p.inW = 64ULL;
  p.inD = 1ULL;
  p.outChannels = 1U;
  p.itemSize = 4U;
  EXPECT_EQ(hk::fastTrafficBytes(p), ~0ULL);
}
