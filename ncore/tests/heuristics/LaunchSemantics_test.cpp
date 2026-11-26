/**
 * @file LaunchSemantics_test.cpp
 * @brief Unit tests for the extended launch-heuristics plan semantics.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include <ncore/core/dtype.h>
#include <ncore/headeronly/heuristics/kernels/launch_config.hh>
#include <tests/heuristics/Caps.hpp>

using tests::heuristics::testCaps;
using tests::heuristics::testKernel;

namespace {

hk::ElementWiseParams stream(uint64_t n, uint32_t item = 4U,
                             uint32_t inputs = 1U) {
  hk::ElementWiseParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = n;
  p.inputItemSize = item;
  p.outputItemSize = item;
  p.inputAlignBytes = 16U;
  p.outputAlignBytes = 16U;
  p.packedElemsPerUnit = 1U;
  p.inputContiguous = true;
  p.outputContiguous = true;
  p.numInputs = inputs;
  return p;
}

hk::FusedOperatorParams fused(const hk::ElementWiseParams &a,
                              const hk::ElementWiseParams &b) {
  hk::FusedOperatorParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.stages[0].pattern = hk::ExecutionPattern::ElementWise;
  p.stages[0].params = &a;
  p.stages[1].pattern = hk::ExecutionPattern::ElementWise;
  p.stages[1].params = &b;
  p.numStages = 2U;
  return p;
}

} // namespace

/**
 * @brief Opt-in shared memory only widens the occupancy ceiling for a
 * kernel that requested it.
 * @test 32 KiB static staging exceeds the 1 KiB default but fits the
 * opt-in limit, so the default path cannot launch and the opt-in path
 * is shared-memory limited at five blocks per SM.
 */
TEST(KernelLimits, SmemOptInRaisesOccupancyCeiling) {
  hk::DeviceCaps caps = testCaps();
  caps.smemPerBlock = 1024U;
  caps.smemPerBlockOptin = 65536U;
  hk::KernelAttrs kernel = testKernel();
  kernel.staticSmemBytes = 32768U;

  const hk::Occupancy denied = hk::computeOccupancy(caps, kernel, 256U);
  EXPECT_EQ(denied.limiter, hk::OccupancyLimiter::CannotLaunch);
  EXPECT_FALSE(denied.known());

  kernel.smemOptIn = true;
  const hk::Occupancy allowed = hk::computeOccupancy(caps, kernel, 256U);
  EXPECT_TRUE(allowed.known());
  EXPECT_EQ(allowed.limiter, hk::OccupancyLimiter::SharedMem);
  EXPECT_EQ(allowed.blocksPerSM, 5U);
}

/**
 * @brief Per-thread dynamic shared memory steps whole warps down to the
 * largest block that can still stage its tile.
 * @test 1 KiB per thread fits four warps under a 128 KiB budget, not eight.
 */
TEST(KernelLimits, DynamicSmemStepsDownFastThreads) {
  hk::DeviceCaps caps = testCaps();
  caps.smemPerBlock = 131072U;
  hk::KernelAttrs kernel = testKernel();
  kernel.dynSmemPerThread = 1024U;

  EXPECT_EQ(hk::detail::fastThreads(caps, kernel), 128U);
  kernel.smemOptIn = true;
  caps.smemPerBlockOptin = 262144U;
  EXPECT_EQ(hk::detail::fastThreads(caps, kernel), 256U);
}

/**
 * @brief A block that cannot stage shared memory at any tried size falls
 * back to one warp rather than an illegal or empty geometry.
 */
TEST(KernelLimits, PickBlockThreadsFallsBackToWarp) {
  hk::DeviceCaps caps = testCaps();
  caps.smemPerBlock = 16U;
  hk::KernelAttrs kernel = testKernel();
  kernel.staticSmemBytes = 1024U;

  EXPECT_EQ(hk::pickBlockThreads(caps, kernel), caps.warpSize);
}

/**
 * @brief Compiled-kernel identity includes the opt-in flag.
 */
TEST(KernelLimits, KernelTagSeparatesSmemOptIn) {
  const hk::KernelAttrs plain = testKernel();
  hk::KernelAttrs opted = plain;
  opted.smemOptIn = true;
  EXPECT_NE(hk::detail::kernelTagFor(plain), hk::detail::kernelTagFor(opted));
}

/**
 * @brief A consumer that rereads the output gets a second resident wave,
 * so a workload between one and two waves is no longer trimmed.
 * @test 409600 elements need 100 blocks; one wave caps at 64, two at 128.
 */
TEST(ElementWiseSemantics, ReuseAfterDoublesResidentWave) {
  hk::ElementWiseParams p = stream(409600ULL);
  p.arithmeticIntensity = 0.08F;

  const hk::ElementWisePlan single = hk::resolveElementWiseFullPlan(p);
  ASSERT_EQ(single.config.blocks.x, 64U);

  p.reuseAfter = true;
  const hk::ElementWisePlan reused = hk::resolveElementWiseFullPlan(p);
  EXPECT_EQ(reused.config.blocks.x, 100U);
}

/**
 * @brief Traffic accounting charges every distinct input tensor.
 * @test 1024 elements at 4B in and out: 8 KiB with one input, 16 KiB
 * with three.
 */
TEST(ElementWiseSemantics, TrafficScalesWithInputCount) {
  hk::ElementWiseParams p = stream(1024ULL);
  EXPECT_EQ(hk::fastTrafficBytes(p), 8192ULL);
  p.numInputs = 3U;
  EXPECT_EQ(hk::fastTrafficBytes(p), 16384ULL);
}

/**
 * @brief Reuse intent changes cache identity even when geometry is equal.
 */
TEST(ElementWiseSemantics, KeySeparatesReuseAfter) {
  hk::ElementWiseParams p = stream(409600ULL);
  hk::ElementWiseParams reused = p;
  reused.reuseAfter = true;
  EXPECT_NE(
      hk::detail::fullKeyFor(p, p.device.archVersion).shapeBucket,
      hk::detail::fullKeyFor(reused, reused.device.archVersion).shapeBucket);
}

/**
 * @brief The scan plan records chainability, output convention, and the
 * buffering mode chosen by the caller.
 */
TEST(ScanSemantics, PlanReflectsScanMode) {
  hk::ParallelScanParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.scanExtent = 8192ULL;
  p.numScans = 4ULL;
  p.itemSize = 4U;
  p.inclusive = false;
  p.op = hk::ScanOp::Add;
  p.crossBlock = hk::ScanCrossBlock::Auto;
  p.buffering = hk::BufferStrategy::Double;

  const hk::ScanPlan plan = hk::resolveParallelScanFullPlan(p);
  EXPECT_TRUE(plan.chained);
  EXPECT_TRUE(plan.exclusive);
  EXPECT_TRUE(plan.doubleBuffered);

  p.deterministic = true;
  EXPECT_FALSE(hk::resolveParallelScanFullPlan(p).chained);
  p.deterministic = false;
  p.crossBlock = hk::ScanCrossBlock::MultiPass;
  EXPECT_FALSE(hk::resolveParallelScanFullPlan(p).chained);
  p.crossBlock = hk::ScanCrossBlock::Auto;
  p.inclusive = true;
  EXPECT_FALSE(hk::resolveParallelScanFullPlan(p).exclusive);
  p.buffering = hk::BufferStrategy::Single;
  EXPECT_FALSE(hk::resolveParallelScanFullPlan(p).doubleBuffered);
}

/**
 * @brief Double buffering stages two tiles, so the odd-elements-per-thread
 * search drops sooner under the same shared-memory budget.
 * @test 256 threads at 4B: seven elements fit one tile under 8 KiB, three
 * fit two tiles.
 */
TEST(ScanSemantics, DoubleBufferBudgetsTwoTiles) {
  hk::ParallelScanParams p{};
  p.device = testCaps();
  p.device.smemPerBlock = 8192U;
  p.kernel = testKernel();
  p.itemSize = 4U;
  p.inclusive = true;

  EXPECT_EQ(hk::detail::scanElemsPerThread(p, 256U), 7U);
  p.buffering = hk::BufferStrategy::Double;
  EXPECT_EQ(hk::detail::scanElemsPerThread(p, 256U), 3U);
}

/**
 * @brief Exclusive output reserves one extra element slot in the final
 * tile, which can cost a step in the odd-element candidate list.
 * @test A 7168-byte budget holds seven inclusive elements but only five
 * exclusive ones.
 */
TEST(ScanSemantics, ExclusiveReservesExtraElement) {
  hk::ParallelScanParams p{};
  p.device = testCaps();
  p.device.smemPerBlock = 7168U;
  p.kernel = testKernel();
  p.itemSize = 4U;
  p.inclusive = true;

  EXPECT_EQ(hk::detail::scanElemsPerThread(p, 256U), 7U);
  p.inclusive = false;
  EXPECT_EQ(hk::detail::scanElemsPerThread(p, 256U), 5U);
}

/**
 * @brief Scan mode decisions are part of the full cache key.
 */
TEST(ScanSemantics, KeySeparatesScanDecisions) {
  hk::ParallelScanParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.scanExtent = 8192ULL;
  p.numScans = 4ULL;
  p.itemSize = 4U;
  p.inclusive = true;
  p.op = hk::ScanOp::Add;

  const auto fold = [](const hk::ParallelScanParams &q) {
    return hk::detail::fullKeyFor(q, q.device.archVersion).shapeBucket;
  };
  const uint64_t base = fold(p);

  hk::ParallelScanParams q = p;
  q.inclusive = false;
  EXPECT_NE(base, fold(q));
  q = p;
  q.deterministic = true;
  EXPECT_NE(base, fold(q));
  q = p;
  q.buffering = hk::BufferStrategy::Double;
  EXPECT_NE(base, fold(q));
  q = p;
  q.crossBlock = hk::ScanCrossBlock::MultiPass;
  EXPECT_NE(base, fold(q));
}

/**
 * @brief The analytic GEMM plan carries the tiling and policy facts the
 * kernel needs, not just a grid pair.
 */
TEST(GemmSemantics, PlanCarriesTilingFacts) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 256ULL;
  p.N = 256ULL;
  p.K = 512ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  p.numStages = 2U;
  p.traversal = hk::TraversalOrder::Grouped;
  p.warpRole = hk::WarpRole::Cooperative;

  const hk::GemmPlan plan = hk::resolveGemmFullPlan(p);
  EXPECT_EQ(plan.tileM, 128U);
  EXPECT_EQ(plan.tileN, 128U);
  EXPECT_EQ(plan.tileK, 32U);
  EXPECT_EQ(plan.stages, 2U);
  EXPECT_EQ(plan.splits, 2U);
  EXPECT_EQ(plan.traversal, hk::TraversalOrder::Grouped);
  EXPECT_EQ(plan.warpRole, hk::WarpRole::Cooperative);
}

/**
 * @brief A column-major A operand orients the output tile the other way.
 */
TEST(GemmSemantics, ColumnMajorSwapsTiles) {
  const std::array<hk::GemmTuningEntry, 1> entries{{
      {
          .mBucket = 1024ULL,
          .nBucket = 1024ULL,
          .kBucket = 1024ULL,
          .dtypeTag = hk::detail::foldDtypeTag(static_cast<uint64_t>(Float32),
                                               static_cast<uint64_t>(Float32)),
          .archTag = 90U,
          .tileM = 64U,
          .tileN = 128U,
          .tileK = 32U,
          .stages = 3U,
          .numSplits = 4U,
      },
  }};
  const hk::GemmTuningTable table{.entries = entries.data(), .numEntries = 1U};

  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 1024ULL;
  p.N = 1024ULL;
  p.K = 1024ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  p.tuningTable = &table;

  const hk::GemmPlan row = hk::resolveGemmFullPlan(p);
  EXPECT_EQ(row.tileM, 64U);
  EXPECT_EQ(row.tileN, 128U);
  p.aRowMajor = false;
  const hk::GemmPlan col = hk::resolveGemmFullPlan(p);
  EXPECT_EQ(col.tileM, 128U);
  EXPECT_EQ(col.tileN, 64U);
}

/**
 * @brief Warp roles that stage load/compute pipelines cannot fill the
 * whole thread budget.
 */
TEST(GemmSemantics, SpecializedRolesCapThreads) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 256ULL;
  p.N = 256ULL;
  p.K = 256ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;

  EXPECT_EQ(hk::resolveGemmFullPlan(p).config.threads.x, 256U);
  p.warpRole = hk::WarpRole::Specialized;
  EXPECT_EQ(hk::resolveGemmFullPlan(p).config.threads.x, 128U);
  p.warpRole = hk::WarpRole::PingPong;
  EXPECT_EQ(hk::resolveGemmFullPlan(p).config.threads.x, 128U);
}

/**
 * @brief Requested stages only feed the analytic shared-memory budget; a
 * measured row keeps its own stage count and footprint.
 */
TEST(GemmSemantics, StagesOnlyBudgetAnalyticPath) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 256ULL;
  p.N = 256ULL;
  p.K = 256ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;

  p.numStages = 3U;
  EXPECT_EQ(hk::resolveGemmFullPlan(p).stages, 3U);
  p.numStages = 0U;
  EXPECT_EQ(hk::resolveGemmFullPlan(p).stages, 0U);

  const std::array<hk::GemmTuningEntry, 1> entries{{
      {
          .mBucket = 256ULL,
          .nBucket = 256ULL,
          .kBucket = 256ULL,
          .dtypeTag = hk::detail::foldDtypeTag(static_cast<uint64_t>(Float32),
                                               static_cast<uint64_t>(Float32)),
          .archTag = 90U,
          .tileM = 64U,
          .tileN = 64U,
          .tileK = 32U,
          .stages = 3U,
          .numSplits = 1U,
      },
  }};
  const hk::GemmTuningTable table{.entries = entries.data(), .numEntries = 1U};
  p.tuningTable = &table;
  p.numStages = 7U;
  const hk::GemmPlan measured = hk::resolveGemmFullPlan(p);
  EXPECT_EQ(measured.stages, 3U);
  p.numStages = 0U;
  const hk::GemmPlan unchanged = hk::resolveGemmFullPlan(p);
  EXPECT_EQ(unchanged.stages, 3U);
  EXPECT_EQ(unchanged.config.blocks.x, measured.config.blocks.x);
}

/**
 * @brief GEMM traffic and proven coalescing follow the dtype and layout
 * contract rather than the tile choice.
 */
TEST(GemmSemantics, TrafficAndCoalescingFollowContract) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 8ULL;
  p.N = 4ULL;
  p.K = 2ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;

  EXPECT_EQ(hk::fastTrafficBytes(p), 224ULL);
  p.inputDtype = Float64;
  EXPECT_EQ(hk::fastTrafficBytes(p), 448ULL);
  EXPECT_TRUE(hk::fastProvenCoalesced(p));
  p.aRowMajor = false;
  EXPECT_FALSE(hk::fastProvenCoalesced(p));
  p.aRowMajor = true;
  p.bRowMajor = false;
  EXPECT_FALSE(hk::fastProvenCoalesced(p));
}

/**
 * @brief Pipeline, traversal, layout, and warp-role decisions all
 * distinguish GEMM cache entries.
 */
TEST(GemmSemantics, KeySeparatesNewFields) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 256ULL;
  p.N = 256ULL;
  p.K = 256ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;

  const auto fold = [](const hk::GemmParams &q) {
    return hk::detail::fullKeyFor(q, q.device.archVersion).shapeBucket;
  };
  const uint64_t base = fold(p);

  hk::GemmParams q = p;
  q.numStages = 4U;
  EXPECT_NE(base, fold(q));
  q = p;
  q.traversal = hk::TraversalOrder::Swizzled;
  EXPECT_NE(base, fold(q));
  q = p;
  q.warpRole = hk::WarpRole::PingPong;
  EXPECT_NE(base, fold(q));
  q = p;
  q.aRowMajor = true;
  EXPECT_NE(base, fold(q));
  q = p;
  q.bRowMajor = true;
  EXPECT_NE(base, fold(q));
}

/**
 * @brief Fused work and traffic sum every valid stage.
 * @test 1024 and 2048 element streams with one and two inputs give 3072
 * elements and 32768 bytes.
 */
TEST(FusedSemantics, WorkAndTrafficSumStages) {
  const hk::ElementWiseParams a = stream(1024ULL, 4U, 1U);
  const hk::ElementWiseParams b = stream(2048ULL, 4U, 2U);
  const hk::FusedOperatorParams p = fused(a, b);

  EXPECT_EQ(hk::fastParallelWork(p), 3072ULL);
  EXPECT_EQ(hk::fastTrafficBytes(p), 32768ULL);
}

/**
 * @brief Recomputed stages count their work twice while traffic stays put.
 */
TEST(FusedSemantics, RecomputeDoublesWork) {
  const hk::ElementWiseParams a = stream(1024ULL, 4U, 1U);
  const hk::ElementWiseParams b = stream(2048ULL, 4U, 2U);
  hk::FusedOperatorParams p = fused(a, b);
  p.recompute = hk::RecomputePolicy::Recompute;

  EXPECT_EQ(hk::fastParallelWork(p), 6144ULL);
  EXPECT_EQ(hk::fastTrafficBytes(p), 32768ULL);
}

/**
 * @brief Global intermediate placement breaks fusion at the first
 * boundary, so the plan is sized from the prefix alone.
 */
TEST(FusedSemantics, GlobalPlacementSplitsAfterFirstStage) {
  const hk::ElementWiseParams head = stream(1024ULL);
  const hk::ElementWiseParams tail = stream(1ULL << 20U);
  hk::FusedOperatorParams p = fused(head, tail);

  const hk::FusedPlan whole = hk::resolveFusedOperatorFullPlan(p);
  EXPECT_FALSE(whole.splitAtBoundary);
  EXPECT_EQ(whole.work, (1024ULL + (1ULL << 20U)));

  p.placement = hk::IntermediatePlacement::Global;
  const hk::FusedPlan split = hk::resolveFusedOperatorFullPlan(p);
  EXPECT_TRUE(split.splitAtBoundary);
  EXPECT_EQ(split.work, 1024ULL);
  EXPECT_EQ(split.bytes, 8192ULL);
}

/**
 * @brief Wide barriers and streaming windows keep every participating
 * block resident, which caps the grid at one wave.
 */
TEST(FusedSemantics, WideSyncCapsGridToOneWave) {
  const hk::ElementWiseParams a = stream(1ULL << 20U);
  const hk::ElementWiseParams b = stream(1ULL << 20U);
  hk::FusedOperatorParams p = fused(a, b);

  hk::DeviceCaps caps = testCaps();
  caps.regsPerSM = 0U;
  caps.maxThreadsPerSM = 0U;
  caps.maxBlocksPerSM = 0U;
  caps.smemPerSM = 0U;
  caps.smemPerBlock = 0U;
  p.device = caps;

  const hk::FusedPlan free = hk::resolveFusedOperatorFullPlan(p);
  ASSERT_EQ(free.config.blocks.x, 64U);

  p.sync = hk::SyncScope::ClusterWide;
  EXPECT_EQ(hk::resolveFusedOperatorFullPlan(p).config.blocks.x, 8U);
  p.sync = hk::SyncScope::GridWide;
  EXPECT_EQ(hk::resolveFusedOperatorFullPlan(p).config.blocks.x, 8U);
  p.sync = hk::SyncScope::Auto;
  p.incrementalBlocks = true;
  EXPECT_EQ(hk::resolveFusedOperatorFullPlan(p).config.blocks.x, 8U);
}

/**
 * @brief Simultaneously live values shrink the block before occupancy
 * scoring, since each extra value consumes registers.
 */
TEST(FusedSemantics, LiveValuesCapThreads) {
  const hk::ElementWiseParams a = stream(4096ULL);
  const hk::ElementWiseParams b = stream(4096ULL);
  hk::FusedOperatorParams p = fused(a, b);

  ASSERT_EQ(hk::resolveFusedOperatorFullPlan(p).config.threads.x, 256U);
  p.maxLiveValues = 64U;
  EXPECT_EQ(hk::resolveFusedOperatorFullPlan(p).config.threads.x, 64U);
}

/**
 * @brief Every top-level fusion decision separates cached entries.
 */
TEST(FusedSemantics, KeySeparatesTopLevelFields) {
  const hk::ElementWiseParams stage = stream(1024ULL);
  hk::FusedOperatorParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.stages[0].pattern = hk::ExecutionPattern::ElementWise;
  p.stages[0].params = &stage;
  p.numStages = 1U;

  const auto fold = [](const hk::FusedOperatorParams &q) {
    return hk::detail::fullKeyFor(q, q.device.archVersion).shapeBucket;
  };
  const uint64_t base = fold(p);

  hk::FusedOperatorParams q = p;
  q.recompute = hk::RecomputePolicy::Recompute;
  EXPECT_NE(base, fold(q));
  q = p;
  q.placement = hk::IntermediatePlacement::Global;
  EXPECT_NE(base, fold(q));
  q = p;
  q.sync = hk::SyncScope::GridWide;
  EXPECT_NE(base, fold(q));
  q = p;
  q.smemReusable = true;
  EXPECT_NE(base, fold(q));
  q = p;
  q.maxLiveValues = 64U;
  EXPECT_NE(base, fold(q));
  q = p;
  q.incrementalBlocks = true;
  EXPECT_NE(base, fold(q));
}

/**
 * @brief A histogram counts in one pass regardless of the requested radix
 * width or value range.
 */
TEST(SortHistogramSemantics, HistogramUsesOnePass) {
  hk::SortHistogramParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 10000ULL;
  p.numBins = 256U;
  p.mode = hk::SortHistMode::Histogram;
  p.keyItemSize = 4U;
  p.valueItemSize = 4U;
  p.valueMin = -1000;
  p.valueMax = 1000;
  p.bitsPerPass = 4U;

  EXPECT_EQ(hk::detail::sortPassCount(p), 1U);
  EXPECT_EQ(hk::fastParallelWork(p), 10000ULL);
  EXPECT_EQ(hk::fastTrafficBytes(p), 40000ULL);
}

/**
 * @brief A proven narrow range or a small distinct-value set shortens the
 * radix chain; an explicit narrower pass count lengthens it.
 */
TEST(SortHistogramSemantics, RangeAndDistinctReducePasses) {
  hk::SortHistogramParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 1000ULL;
  p.mode = hk::SortHistMode::Sort;
  p.keyItemSize = 4U;
  p.valueItemSize = 4U;

  EXPECT_EQ(hk::detail::sortPassCount(p), 4U);
  EXPECT_EQ(hk::fastParallelWork(p), 4000ULL);
  EXPECT_EQ(hk::fastTrafficBytes(p), 32000ULL);

  p.valueMin = 0;
  p.valueMax = 255;
  EXPECT_EQ(hk::detail::sortPassCount(p), 1U);
  p.valueMax = 65535;
  EXPECT_EQ(hk::detail::sortPassCount(p), 2U);

  p.distribution.numDistinct = 200U;
  EXPECT_EQ(hk::detail::sortPassCount(p), 1U);
  p.distribution.numDistinct = 300U;
  EXPECT_EQ(hk::detail::sortPassCount(p), 2U);

  p.valueMax = 0;
  p.distribution.numDistinct = 0;
  p.bitsPerPass = 4U;
  EXPECT_EQ(hk::detail::sortPassCount(p), 8U);
}

/**
 * @brief Histogram counter copies consume the shared budget, so a
 * concentrated distribution promotes Auto privatization to warp level.
 */
TEST(SortHistogramSemantics, PrivatizationReshapesBudget) {
  hk::SortHistogramParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 100000ULL;
  p.numBins = 2048U;
  p.mode = hk::SortHistMode::Histogram;
  p.keyItemSize = 4U;

  EXPECT_EQ(hk::resolveSortHistogram(p).threads.x, 256U);
  p.distribution.concentration01 = 0.9F;
  EXPECT_EQ(hk::resolveSortHistogram(p).threads.x, 128U);
  p.privatization = hk::Privatization::PerBlock;
  EXPECT_EQ(hk::resolveSortHistogram(p).threads.x, 256U);
  p.privatization = hk::Privatization::PerThread;
  EXPECT_EQ(hk::resolveSortHistogram(p).threads.x, 128U);
}

/**
 * @brief Narrower radix passes reread the keys more often, coarsening the
 * elements assigned to each thread and halving the block count.
 */
TEST(SortHistogramSemantics, BitsPerPassCoarsensGrid) {
  hk::SortHistogramParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 8192ULL;
  p.mode = hk::SortHistMode::Sort;
  p.keyItemSize = 4U;
  p.valueItemSize = 4U;

  EXPECT_EQ(hk::resolveSortHistogram(p).blocks.x, 8U);
  p.bitsPerPass = 4U;
  EXPECT_EQ(hk::resolveSortHistogram(p).blocks.x, 4U);
}

/**
 * @brief Spatial traffic charges the full filter volume, depth included.
 */
TEST(SpatialSemantics, TrafficCountsFilterDepth) {
  auto make = [](uint32_t depth) {
    hk::SpatialNeighborhoodParams p{};
    p.device = testCaps();
    p.kernel = testKernel();
    p.filterH = 3U;
    p.filterW = 3U;
    p.filterD = depth;
    p.inH = 8ULL;
    p.inW = 8ULL;
    p.inD = 1ULL;
    p.outChannels = 1U;
    p.itemSize = 4U;
    p.strideH = 1U;
    p.strideW = 1U;
    p.strideD = 1U;
    p.dilationH = 1U;
    p.dilationW = 1U;
    p.dilationD = 1U;
    return p;
  };

  EXPECT_EQ(hk::fastTrafficBytes(make(1U)), 2560ULL);
  EXPECT_EQ(hk::fastTrafficBytes(make(3U)), 7168ULL);
}

/**
 * @brief Kernel staging is subtracted from the shared budget before a
 * tensor tile can be selected.
 */
TEST(SpatialSemantics, OwnSmemShrinksTensorBudget) {
  hk::SpatialNeighborhoodParams p{};
  p.device = testCaps();
  p.device.smemPerBlockOptin = 3000000ULL;
  p.kernel = testKernel();
  p.kernel.smemOptIn = true;
  p.filterH = 1U;
  p.filterW = 1U;
  p.filterD = 1U;
  p.inH = 64ULL;
  p.inW = 64ULL;
  p.inD = 8ULL;
  p.inChannels = 16U;
  p.outChannels = 16U;
  p.itemSize = 4U;
  p.strideH = 1U;
  p.strideW = 1U;
  p.strideD = 1U;
  p.dilationH = 1U;
  p.dilationW = 1U;
  p.dilationD = 1U;
  p.allowTensorReformulation = true;

  ASSERT_EQ(hk::resolveSpatialNeighborhood(p).threads.x, 32U);
  p.kernel.staticSmemBytes = 2000000ULL;
  EXPECT_NE(hk::resolveSpatialNeighborhood(p).threads.x, 32U);
}

/**
 * @brief Tensor reformulation stages the full depth span, so a deep
 * filter no longer fits the same wide tile.
 */
TEST(SpatialSemantics, SpanDRaisesTensorStagedBytes) {
  hk::SpatialNeighborhoodParams p{};
  p.device = testCaps();
  p.device.smemPerBlockOptin = 3000000ULL;
  p.kernel = testKernel();
  p.kernel.smemOptIn = true;
  p.filterH = 1U;
  p.filterW = 1U;
  p.filterD = 1U;
  p.inH = 64ULL;
  p.inW = 64ULL;
  p.inD = 8ULL;
  p.inChannels = 16U;
  p.outChannels = 16U;
  p.itemSize = 4U;
  p.strideH = 1U;
  p.strideW = 1U;
  p.strideD = 1U;
  p.dilationH = 1U;
  p.dilationW = 1U;
  p.dilationD = 1U;
  p.allowTensorReformulation = true;

  ASSERT_EQ(hk::resolveSpatialNeighborhood(p).threads.x, 32U);
  p.filterD = 33U;
  EXPECT_EQ(hk::resolveSpatialNeighborhood(p).threads.x, 8U);
}

/**
 * @brief A constant filter doubles the channels packed per thread, which
 * halves the channel groups in the grid.
 */
TEST(SpatialSemantics, FilterConstPacksChannels) {
  hk::SpatialNeighborhoodParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.filterH = 1U;
  p.filterW = 1U;
  p.filterD = 1U;
  p.inH = 16ULL;
  p.inW = 16ULL;
  p.inD = 1ULL;
  p.inChannels = 8U;
  p.outChannels = 16U;
  p.itemSize = 4U;
  p.strideH = 1U;
  p.strideW = 1U;
  p.strideD = 1U;
  p.dilationH = 1U;
  p.dilationW = 1U;
  p.dilationD = 1U;

  EXPECT_EQ(hk::resolveSpatialNeighborhood(p).blocks.x, 16U);
  p.filterConstEligible = true;
  EXPECT_EQ(hk::resolveSpatialNeighborhood(p).blocks.x, 8U);
}

/**
 * @brief Border strategy participates in cache identity, since the
 * resolver stages a halo only for the other two modes.
 */
TEST(SpatialSemantics, KeySeparatesBorderStrategy) {
  hk::SpatialNeighborhoodParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.filterH = 3U;
  p.filterW = 3U;
  p.filterD = 1U;
  p.inH = 16ULL;
  p.inW = 16ULL;
  p.inD = 1ULL;
  p.outChannels = 1U;
  p.itemSize = 4U;
  p.border = hk::BorderStrategy::AtLoad;

  hk::SpatialNeighborhoodParams q = p;
  q.border = hk::BorderStrategy::InLoop;
  EXPECT_NE(hk::detail::fullKeyFor(p, p.device.archVersion).shapeBucket,
            hk::detail::fullKeyFor(q, q.device.archVersion).shapeBucket);
}

/**
 * @brief Indirect traffic counts read-write asymmetry and the penalty for
 * duplicate indices inside a warp.
 */
TEST(GatherSemantics, TrafficFollowsHints) {
  hk::GatherScatterParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numIndices = 1024ULL;
  p.segmentSize = 4ULL;
  p.dataItemSize = 4U;
  p.indexItemSize = 4U;
  p.indicesUnique = true;
  p.readWriteRatio = 0.0F;

  EXPECT_EQ(hk::fastTrafficBytes(p), 20480ULL);
  p.readWriteRatio = 3.0F;
  EXPECT_EQ(hk::fastTrafficBytes(p), 69632ULL);
  p.readWriteRatio = 0.0F;
  p.indicesUnique = false;
  p.expectIntraWarpDup = true;
  EXPECT_EQ(hk::fastTrafficBytes(p), 36864ULL);
  p.indicesUnique = true;
  p.expectIntraWarpDup = false;
  p.readWriteRatio = 1000.0F;
  EXPECT_EQ(hk::fastTrafficBytes(p), 4182016ULL);
}

/**
 * @brief Clustered indices and pre-passes coarsen the assignment grain,
 * with dedup and presort sharing one extra step.
 */
TEST(GatherSemantics, LocalityAndPrepCoarsenDivisor) {
  hk::GatherScatterParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numIndices = 10000ULL;
  p.segmentSize = 4ULL;
  p.dataItemSize = 4U;
  p.indexItemSize = 4U;
  p.distribution.spread = hk::IndexSpread::Scattered;
  p.distribution.locality01 = 0.0F;

  EXPECT_EQ(hk::resolveGatherScatter(p).blocks.x, 40U);
  p.distribution.spread = hk::IndexSpread::Clustered;
  EXPECT_EQ(hk::resolveGatherScatter(p).blocks.x, 20U);
  p.distribution.spread = hk::IndexSpread::Unknown;
  p.distribution.locality01 = 0.9F;
  EXPECT_EQ(hk::resolveGatherScatter(p).blocks.x, 20U);
  p.distribution.spread = hk::IndexSpread::Scattered;
  p.distribution.locality01 = 0.0F;
  p.dedup = hk::WarpDedup::On;
  EXPECT_EQ(hk::resolveGatherScatter(p).blocks.x, 20U);
  p.dedup = hk::WarpDedup::Auto;
  p.presort = hk::IndexPresort::On;
  EXPECT_EQ(hk::resolveGatherScatter(p).blocks.x, 20U);
  p.dedup = hk::WarpDedup::On;
  EXPECT_EQ(hk::resolveGatherScatter(p).blocks.x, 20U);
}

/**
 * @brief Every gather hint that changes cost also changes cache identity.
 */
TEST(GatherSemantics, KeySeparatesGatherHints) {
  hk::GatherScatterParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numIndices = 10000ULL;
  p.segmentSize = 4ULL;
  p.dataItemSize = 4U;
  p.indexItemSize = 4U;
  p.indicesUnique = true;

  const auto fold = [](const hk::GatherScatterParams &q) {
    return hk::detail::fullKeyFor(q, q.device.archVersion).shapeBucket;
  };
  const uint64_t base = fold(p);

  hk::GatherScatterParams q = p;
  q.indicesUnique = false;
  EXPECT_NE(base, fold(q));
  q = p;
  q.expectIntraWarpDup = true;
  EXPECT_NE(base, fold(q));
  q = p;
  q.readWriteRatio = 3.0F;
  EXPECT_NE(base, fold(q));
  q = p;
  q.distribution.spread = hk::IndexSpread::Clustered;
  EXPECT_NE(base, fold(q));
  q = p;
  q.distribution.locality01 = 0.9F;
  EXPECT_NE(base, fold(q));
  q = p;
  q.dedup = hk::WarpDedup::On;
  EXPECT_NE(base, fold(q));
  q = p;
  q.presort = hk::IndexPresort::On;
  EXPECT_NE(base, fold(q));
}

namespace {

hk::ElementWiseParams routerStream(uint64_t bytes) {
  hk::ElementWiseParams p = stream(bytes / 8U);
  p.device.memBandwidthBytesPerSec = 900000000000ULL;
  return p;
}

} // namespace

/**
 * @brief Proven coalescing halves the memory estimate and can flip a
 * bandwidth-bound stream from Full to Fast.
 * @test 19 MB at 900 GB/s: 0.4 efficiency misses the 50 us launch budget,
 * 0.7 efficiency clears it.
 */
TEST(RouterHints, CoalescingChangesTier) {
  hk::ElementWiseParams p = routerStream(19000000ULL);
  p.inputAlignBytes = 4U;
  ASSERT_EQ(hk::fastTrafficBytes(p), 19000000ULL);
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  p.inputAlignBytes = 16U;
  p.outputAlignBytes = 16U;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
}

/**
 * @brief Expected divergence scales bandwidth efficiency down even for a
 * vectorized stream.
 */
TEST(RouterHints, DivergenceChangesTier) {
  hk::ElementWiseParams p = routerStream(19000000ULL);
  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  p.context.expectedDivergence = 0.5F;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief A stream that fits in L2 gets a higher effective bandwidth.
 */
TEST(RouterHints, L2FitChangesTier) {
  hk::ElementWiseParams p = routerStream(19000000ULL);
  p.inputAlignBytes = 4U;
  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  p.device.l2Size = 10000000ULL;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  p.device.l2Size = 19000000ULL;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
}

/**
 * @brief A persistent policy window boosts efficiency without L2
 * ownership of the whole working set.
 */
TEST(RouterHints, PersistentL2ChangesTier) {
  hk::ElementWiseParams p = routerStream(19000000ULL);
  p.inputAlignBytes = 4U;
  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  p.device.l2MaxWindow = 19000000ULL;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
}

/**
 * @brief GEMM coalescing is part of the same memory-time estimate.
 * @test 128x128x1 at 2.5 GB/s: 26 KiB misses scalar efficiency but clears
 * aligned efficiency.
 */
TEST(RouterHints, GemmCoalescingChangesTier) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.device.memBandwidthBytesPerSec = 2500000000ULL;
  p.kernel = testKernel();
  p.M = 128ULL;
  p.N = 128ULL;
  p.K = 1ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  ASSERT_EQ(hk::fastTrafficBytes(p), 66560ULL);
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  p.bRowMajor = false;
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief GEMM operand traffic, not just tile coverage, drives the tier.
 */
TEST(RouterHints, GemmTrafficChangesTier) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.device.memBandwidthBytesPerSec = 2500000000ULL;
  p.kernel = testKernel();
  p.M = 128ULL;
  p.N = 128ULL;
  p.K = 1ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  p.K = 128ULL;
  EXPECT_EQ(hk::fastTrafficBytes(p), 196608ULL);
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief Fused traffic is summed across stages before the router decides.
 * @test At 10 GB/s two 8K streams stay under the launch budget, while two
 * 16K streams cross it.
 */
TEST(RouterHints, FusedTrafficChangesTier) {
  const hk::ElementWiseParams a = stream(8192ULL, 4U, 1U);
  const hk::ElementWiseParams b = stream(8192ULL, 4U, 1U);
  hk::FusedOperatorParams small = fused(a, b);
  small.device.memBandwidthBytesPerSec = 10000000000ULL;
  ASSERT_EQ(hk::fastParallelWork(small), 16384ULL);
  ASSERT_EQ(hk::fastTrafficBytes(small), 131072ULL);
  EXPECT_TRUE(hk::resolveTier(small) == hk::HeuristicTier::Fast);

  const hk::ElementWiseParams c = stream(16384ULL, 4U, 1U);
  const hk::ElementWiseParams d = stream(16384ULL, 4U, 1U);
  hk::FusedOperatorParams large = fused(c, d);
  large.device.memBandwidthBytesPerSec = 10000000000ULL;
  ASSERT_EQ(hk::fastTrafficBytes(large), 262144ULL);
  EXPECT_TRUE(hk::resolveTier(large) == hk::HeuristicTier::Full);
}

/**
 * @brief Fused work below one wave routes Fast even with heavy traffic.
 */
TEST(RouterHints, FusedWorkBelowWaveRoutesFast) {
  const hk::ElementWiseParams a = stream(4096ULL, 1024U, 1U);
  const hk::ElementWiseParams b = stream(4096ULL, 1024U, 1U);
  hk::FusedOperatorParams p = fused(a, b);
  p.device.memBandwidthBytesPerSec = 10000000000ULL;
  ASSERT_EQ(hk::fastParallelWork(p), 8192ULL);
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
}

/**
 * @brief Recompute raises fused work past the wave gate and exposes the
 * traffic estimate to the router.
 */
TEST(RouterHints, FusedRecomputeChangesTier) {
  const hk::ElementWiseParams a = stream(10000ULL, 1024U, 1U);
  const hk::ElementWiseParams b = stream(0ULL, 1024U, 1U);
  hk::FusedOperatorParams p = fused(a, b);
  p.device.memBandwidthBytesPerSec = 10000000000ULL;
  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  p.recompute = hk::RecomputePolicy::Recompute;
  ASSERT_EQ(hk::fastParallelWork(p), 20000ULL);
  EXPECT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
}

/**
 * @brief Device identity includes the L2 knobs the router reads.
 */
TEST(RouterHints, DeviceTagSeparatesL2Fields) {
  hk::DeviceCaps base = testCaps();
  hk::DeviceCaps l2 = base;
  l2.l2Size = 19000000ULL;
  hk::DeviceCaps persistent = base;
  persistent.l2MaxWindow = 19000000ULL;
  hk::DeviceCaps optin = base;
  optin.smemPerBlockOptin = 65536U;

  EXPECT_NE(hk::detail::deviceTagFor(base), hk::detail::deviceTagFor(l2));
  EXPECT_NE(hk::detail::deviceTagFor(base),
            hk::detail::deviceTagFor(persistent));
  EXPECT_NE(hk::detail::deviceTagFor(base), hk::detail::deviceTagFor(optin));
}
