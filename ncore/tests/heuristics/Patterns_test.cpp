/**
 * @file Patterns_test.cpp
 * @brief Unit tests for per-pattern Fast/Full launch geometry.
 *
 * Host-side arithmetic with stub device caps; no GPU is required.
 * Each case pins one documented decision from its pattern SPEC.
 */

#include <array>

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

hk::KernelAttrs testKernel() {
  hk::KernelAttrs k{};
  k.regsPerThread = 16U;
  k.maxThreadsPerBlockKernel = 1024U;
  return k;
}

} // namespace

/**
 * @brief Aligned contiguous fp32 stream takes the capped Fast grid.
 * @test 1M elements: 256 threads, 1024-elem waves, 64-block cap.
 */
TEST(ElementWiseFast, VectorizedCappedGrid) {
  hk::ElementWiseParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 1ULL << 20U;
  p.inputItemSize = 4U;
  p.outputItemSize = 4U;
  p.inputAlignBytes = 16U;
  p.outputAlignBytes = 16U;
  p.packedElemsPerUnit = 1U;
  p.inputContiguous = true;
  p.outputContiguous = true;
  p.numInputs = 2U;
  p.arithmeticIntensity = 0.08F;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 64U);
}

/**
 * @brief Strided input stays correct through the same grid law.
 */
TEST(ElementWiseFast, StridedInputSameGrid) {
  hk::ElementWiseParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 1ULL << 20U;
  p.inputItemSize = 4U;
  p.outputItemSize = 4U;
  p.inputAlignBytes = 4U;
  p.outputAlignBytes = 16U;
  p.packedElemsPerUnit = 1U;
  p.inputContiguous = false;
  p.outputContiguous = true;
  p.numInputs = 1U;
  p.arithmeticIntensity = 0.08F;

  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 64U);
}

/**
 * @brief One block per reduction for a small batch.
 */
TEST(ReductionFast, OneBlockPerReduction) {
  hk::ReductionParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.reduceExtent = 512ULL;
  p.numReductions = 4ULL;
  p.reduceContiguous = true;
  p.op = hk::ReductionOp::Add;
  p.inputItemSize = 4U;
  p.accumItemSize = 4U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 4U);
}

/**
 * @brief Unknown caps still cap the Full thread budget.
 * @test 1M elements with zero caps: 256 threads, 256 blocks.
 */
TEST(ElementWiseFull, UnknownCapsCapped) {
  hk::ElementWiseParams p{};
  p.numElements = 1ULL << 20U;
  p.inputItemSize = 4U;
  p.outputItemSize = 4U;
  p.inputAlignBytes = 16U;
  p.outputAlignBytes = 16U;
  p.packedElemsPerUnit = 1U;
  p.inputContiguous = true;
  p.outputContiguous = true;
  p.numInputs = 1U;

  const hk::LaunchConfig cfg = hk::resolveElementWise(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 256U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief Wide axis splits across block rows under the SM cap.
 * @test 1M extent x batch 2: 256 splits capped to y=32.
 */
TEST(ReductionFull, SplitPiecesCapped) {
  hk::ReductionParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.reduceExtent = 1ULL << 20U;
  p.numReductions = 2ULL;
  p.reduceContiguous = true;
  p.op = hk::ReductionOp::Add;
  p.inputItemSize = 4U;
  p.accumItemSize = 4U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 2U);
  EXPECT_EQ(cfg.blocks.y, 32U);
}

/**
 * @brief Identity layout streams like element-wise.
 */
TEST(LayoutFast, IdentityStreams) {
  hk::LayoutTransformParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.layout.ndim = 2U;
  p.layout.shape[0] = 1024;
  p.layout.shape[1] = 1024;
  p.itemSize = 4U;
  p.perm[0] = 0U;
  p.perm[1] = 1U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 64U);
}

/**
 * @brief 2D transpose takes the fixed 32x8 tile.
 */
TEST(LayoutFast, TransposeFixedTile) {
  hk::LayoutTransformParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.layout.ndim = 2U;
  p.layout.shape[0] = 256;
  p.layout.shape[1] = 256;
  p.itemSize = 4U;
  p.perm[0] = 1U;
  p.perm[1] = 0U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 32U);
  EXPECT_EQ(cfg.threads.y, 8U);
  EXPECT_EQ(cfg.blocks.x, 8U);
  EXPECT_EQ(cfg.blocks.y, 32U);
}

/**
 * @brief Rank-3 permute fails Fast coverage.
 */
TEST(LayoutFast, Rank3FailsCoverage) {
  hk::LayoutTransformParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.layout.ndim = 3U;
  p.layout.shape[0] = 4;
  p.layout.shape[1] = 4;
  p.layout.shape[2] = 4;
  p.itemSize = 4U;
  p.perm[0] = 2U;
  p.perm[1] = 1U;
  p.perm[2] = 0U;

  EXPECT_FALSE(hk::coversLayoutTransformFast(p));
}

/**
 * @brief Oversized rank degrades to the floor without over-reading.
 */
TEST(LayoutFull, OversizedRankFloors) {
  hk::LayoutTransformParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.layout.ndim = 200U;
  p.itemSize = 4U;

  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 1U);
  EXPECT_EQ(cfg.blocks.x, 1U);
}

/**
 * @brief Small scan batch takes one block per scan.
 */
TEST(ScanFast, OneBlockPerScan) {
  hk::ParallelScanParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.scanExtent = 1024ULL;
  p.numScans = 8ULL;
  p.itemSize = 4U;
  p.inclusive = true;
  p.op = hk::ScanOp::Add;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.blocks.x, 8U);
}

/**
 * @brief Restricted float op fails Fast coverage.
 */
TEST(ScanFast, RestrictedOpFailsCoverage) {
  hk::ParallelScanParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.scanExtent = 64ULL;
  p.numScans = 1ULL;
  p.itemSize = 4U;
  p.op = hk::ScanOp::CustomRestricted;

  EXPECT_FALSE(hk::coversParallelScanFast(p));
}

/**
 * @brief Long axis splits into pieces across the batch grid.
 * @test 8192-wide axis, 2 scans: 5 pieces of 1792, grid 2x5.
 */
TEST(ScanFull, MultiPieceGrid) {
  hk::ParallelScanParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.scanExtent = 8192ULL;
  p.numScans = 2ULL;
  p.itemSize = 4U;
  p.inclusive = true;
  p.op = hk::ScanOp::Add;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.x, 2U);
  EXPECT_EQ(cfg.blocks.y, 5U);
  EXPECT_EQ(cfg.blocks.z, 1U);
}

/**
 * @brief Small histogram takes the shared-privatized grid.
 */
TEST(SortHistogramFast, SmallHistogram) {
  hk::SortHistogramParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 100000ULL;
  p.numBins = 256U;
  p.mode = hk::SortHistMode::Histogram;
  p.keyItemSize = 4U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 64U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief Oversized bins fail Fast coverage.
 */
TEST(SortHistogramFast, OversizedBinsFailCoverage) {
  hk::SortHistogramParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 1ULL << 20U;
  p.numBins = 1000000U;
  p.mode = hk::SortHistMode::Histogram;
  p.keyItemSize = 4U;

  EXPECT_FALSE(hk::coversSortHistogramFast(p));
}

/**
 * @brief Sort mode past the Fast element bound takes the Full grid.
 * @test 1M keys: 8-bit radix fits shared, 256 threads, capped at 64.
 */
TEST(SortFull, SortModeGrid) {
  hk::SortHistogramParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numElements = 1ULL << 20U;
  p.mode = hk::SortHistMode::Sort;
  p.keyItemSize = 4U;
  p.valueItemSize = 4U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 64U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief 1x1 spatial takes the streaming-equivalent grid.
 */
TEST(SpatialFast, PointwiseTile) {
  hk::SpatialNeighborhoodParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.filterH = 1U;
  p.filterW = 1U;
  p.filterD = 1U;
  p.inH = 64ULL;
  p.inW = 64ULL;
  p.inD = 1ULL;
  p.inChannels = 3U;
  p.outChannels = 3U;
  p.itemSize = 4U;
  p.strideH = 1U;
  p.strideW = 1U;
  p.strideD = 1U;
  p.dilationH = 1U;
  p.dilationW = 1U;
  p.dilationD = 1U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 16U);
  EXPECT_EQ(cfg.threads.y, 16U);
}

/**
 * @brief Strided spatial fails Fast coverage.
 */
TEST(SpatialFast, StridedFailsCoverage) {
  hk::SpatialNeighborhoodParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.filterH = 3U;
  p.filterW = 3U;
  p.filterD = 1U;
  p.inH = 64ULL;
  p.inW = 64ULL;
  p.inD = 1ULL;
  p.outChannels = 1U;
  p.strideH = 2U;
  p.strideW = 2U;
  p.strideD = 1U;
  p.dilationH = 1U;
  p.dilationW = 1U;
  p.dilationD = 1U;

  EXPECT_FALSE(hk::coversSpatialNeighborhoodFast(p));
}

/**
 * @brief Strided 3D convolution takes the scored Full grid.
 * @test 64-cubed, 8 channels, stride 2: D tiles live in z only.
 */
TEST(SpatialFull, StridedGrid) {
  hk::SpatialNeighborhoodParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.filterH = 3U;
  p.filterW = 3U;
  p.filterD = 1U;
  p.inH = 64ULL;
  p.inW = 64ULL;
  p.inD = 64ULL;
  p.inChannels = 8U;
  p.outChannels = 8U;
  p.itemSize = 4U;
  p.strideH = 2U;
  p.strideW = 1U;
  p.strideD = 1U;
  p.dilationH = 1U;
  p.dilationW = 1U;
  p.dilationD = 1U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 32U);
  EXPECT_EQ(cfg.threads.y, 8U);
  EXPECT_EQ(cfg.blocks.x, 8U);
  EXPECT_EQ(cfg.blocks.y, 1U);
  EXPECT_EQ(cfg.blocks.z, 2U);
}

/**
 * @brief 512-cubed GEMM takes block-per-tile without splits.
 * @test 16 tiles of 128, K too short to split.
 */
TEST(GemmFull, MediumBlockPerTile) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 512ULL;
  p.N = 512ULL;
  p.K = 512ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 16U);
}

/**
 * @brief Tuning-table hit overrides tiles and splits.
 * @test Persistent schedule collapses the grid to the SM count.
 */
TEST(GemmFull, TuningHitOverrides) {
  const std::array<hk::GemmTuningEntry, 1> entries{{{
      .mBucket = 1024ULL,
      .nBucket = 1024ULL,
      .kBucket = 1024ULL,
      .dtypeTag = hk::detail::foldDtypeTag(static_cast<uint64_t>(Float32),
                                           static_cast<uint64_t>(Float32)),
      .archTag = 90U,
      .tileM = 64U,
      .tileN = 64U,
      .tileK = 32U,
      .stages = 3U,
      .numSplits = 4U,
  }}};
  const hk::GemmTuningTable table{entries.data(), 1U};

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

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.blocks.x, 8U);
}

/**
 * @brief Tuning-table row tagged for another dtype must miss.
 * @test fp32-tagged row with an fp16 query falls back to the
 * analytic 64-block grid instead of the recorded 8.
 */
TEST(GemmFull, TuningWrongDtypeMisses) {
  const std::array<hk::GemmTuningEntry, 1> entries{{{
      .mBucket = 1024ULL,
      .nBucket = 1024ULL,
      .kBucket = 1024ULL,
      .dtypeTag = hk::detail::foldDtypeTag(static_cast<uint64_t>(Float32),
                                           static_cast<uint64_t>(Float32)),
      .archTag = 90U,
      .tileM = 64U,
      .tileN = 64U,
      .tileK = 32U,
      .stages = 3U,
      .numSplits = 4U,
  }}};
  const hk::GemmTuningTable table{entries.data(), 1U};

  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 1024ULL;
  p.N = 1024ULL;
  p.K = 1024ULL;
  p.inputDtype = Float16;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  p.tuningTable = &table;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 64U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief Analytic split-K refills idle waves on large K.
 * @test 256x256x4096 on 8 SMs: 4 tiles, last wave short, 2 splits.
 */
TEST(GemmFull, SplitKRefillsWaves) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 256ULL;
  p.N = 256ULL;
  p.K = 4096ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 8U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief Tile counts past eight waves switch to the persistent grid.
 * @test 1152x1152 output tiles dwarf 8 SMs: grid collapses to 8.
 */
TEST(GemmFull, PersistentCollapsesGrid) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 1152ULL;
  p.N = 1152ULL;
  p.K = 16ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 8U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief Small segments stream thread-per-element.
 */
TEST(GatherFast, SmallSegmentsStream) {
  hk::GatherScatterParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numIndices = 10000ULL;
  p.dataItemSize = 4U;
  p.indexItemSize = 4U;
  p.indicesUnique = true;
  p.segmentSize = 2ULL;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
}

/**
 * @brief Large segments fail Fast coverage.
 */
TEST(GatherFast, LargeSegmentsFailCoverage) {
  hk::GatherScatterParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numIndices = 100ULL;
  p.dataItemSize = 4U;
  p.segmentSize = 256ULL;

  EXPECT_FALSE(hk::coversGatherScatterFast(p));
}

/**
 * @brief Medium segments take one warp per segment group.
 * @test 128-byte segments: 3200 elements in 25 warp-sized groups.
 */
TEST(GatherFull, WarpPerSegment) {
  hk::GatherScatterParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numIndices = 100ULL;
  p.dataItemSize = 4U;
  p.indexItemSize = 4U;
  p.indicesUnique = true;
  p.segmentSize = 32ULL;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 32U);
  EXPECT_EQ(cfg.blocks.x, 25U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief Unlisted epilogue work never claims the cheap path.
 * @test 64-cubed is Fast-sized, but Custom forces Full; grid stays trivial.
 */
TEST(GemmFull, CustomEpilogueForcesFull) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 64ULL;
  p.N = 64ULL;
  p.K = 64ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  p.epilogue.activation = hk::ActivationKind::Custom;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 1U);
  EXPECT_EQ(cfg.threads.y, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
}

/**
 * @brief Forced StreamK takes the SM-proportional fixed grid.
 * @test 256x256x16: Auto sizes 4 block-per-tile, StreamK takes 8.
 */
TEST(GemmFull, StreamKFixesGrid) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 256ULL;
  p.N = 256ULL;
  p.K = 16ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  p.schedule = hk::CtaSchedule::StreamK;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig forced = hk::resolveLaunchConfig(p);
  EXPECT_EQ(forced.threads.x, 256U);
  EXPECT_EQ(forced.blocks.x, 8U);

  p.schedule = hk::CtaSchedule::Auto;
  const hk::LaunchConfig analytic = hk::resolveLaunchConfig(p);
  EXPECT_EQ(analytic.threads.x, 256U);
  EXPECT_EQ(analytic.blocks.x, 4U);
}

/**
 * @brief Grouped traversal is record-only for downstream pid mapping.
 * @test Same shape sizes the identical grid under both orders.
 */
TEST(GemmFull, TraversalGroupedRecordOnly) {
  hk::GemmParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.M = 256ULL;
  p.N = 256ULL;
  p.K = 16ULL;
  p.inputDtype = Float32;
  p.accumDtype = Float32;
  p.aRowMajor = true;
  p.bRowMajor = true;
  p.traversal = hk::TraversalOrder::RowMajor;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig row = hk::resolveLaunchConfig(p);
  p.traversal = hk::TraversalOrder::Grouped;
  const hk::LaunchConfig grouped = hk::resolveLaunchConfig(p);
  EXPECT_EQ(row.threads.x, grouped.threads.x);
  EXPECT_EQ(row.blocks.x, grouped.blocks.x);
  EXPECT_EQ(grouped.threads.x, 256U);
  EXPECT_EQ(grouped.blocks.x, 4U);
}

/**
 * @brief Single-stage fusion matches the stage Fast geometry.
 */
TEST(FusedFast, SingleStageMatchesStage) {
  hk::ElementWiseParams stage{};
  stage.device = testCaps();
  stage.kernel = testKernel();
  stage.numElements = 4096ULL;
  stage.inputItemSize = 4U;
  stage.outputItemSize = 4U;
  stage.inputAlignBytes = 16U;
  stage.outputAlignBytes = 16U;
  stage.packedElemsPerUnit = 1U;
  stage.inputContiguous = true;
  stage.outputContiguous = true;
  stage.numInputs = 1U;

  hk::FusedOperatorParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.stages[0].pattern = hk::ExecutionPattern::ElementWise;
  p.stages[0].params = &stage;
  p.numStages = 1U;
  p.smemReusable = true;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig stageCfg = hk::resolveLaunchConfig(stage);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, stageCfg.threads.x);
  EXPECT_EQ(cfg.blocks.x, stageCfg.blocks.x);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 4U);
}

/**
 * @brief Tiny fusion shrinks the grid to its stage work.
 * @test 64 elements need one 256-thread block, not the device-wide grid.
 */
TEST(FusedFast, TinyFusionShrinksGrid) {
  hk::ElementWiseParams stage{};
  stage.device = testCaps();
  stage.kernel = testKernel();
  stage.numElements = 64ULL;
  stage.inputItemSize = 4U;
  stage.outputItemSize = 4U;
  stage.inputAlignBytes = 16U;
  stage.outputAlignBytes = 16U;
  stage.packedElemsPerUnit = 1U;
  stage.inputContiguous = true;
  stage.outputContiguous = true;
  stage.numInputs = 1U;

  hk::FusedOperatorParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.stages[0].pattern = hk::ExecutionPattern::ElementWise;
  p.stages[0].params = &stage;
  p.numStages = 1U;
  p.smemReusable = true;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Fast);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 1U);
}

/**
 * @brief Rank-3 transpose takes the budgeted Full tile.
 * @test 4x4x4 reversed: 32-wide tile, 8 rows, outer dim in z.
 */
TEST(LayoutFull, Rank3Geometry) {
  hk::LayoutTransformParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.layout.ndim = 3U;
  p.layout.shape[0] = 4;
  p.layout.shape[1] = 4;
  p.layout.shape[2] = 4;
  p.itemSize = 4U;
  p.perm[0] = 2U;
  p.perm[1] = 1U;
  p.perm[2] = 0U;

  ASSERT_TRUE(hk::resolveTier(p) == hk::HeuristicTier::Full);
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 32U);
  EXPECT_EQ(cfg.threads.y, 8U);
  EXPECT_EQ(cfg.blocks.x, 1U);
  EXPECT_EQ(cfg.blocks.y, 1U);
  EXPECT_EQ(cfg.blocks.z, 4U);
}

/**
 * @brief Dangling stage pointer fails validation into the floor.
 */
TEST(FusedValidation, NullStagePointerFloors) {
  hk::FusedOperatorParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.stages[0].pattern = hk::ExecutionPattern::ElementWise;
  p.stages[0].params = nullptr;
  p.numStages = 1U;

  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 1U);
  EXPECT_EQ(cfg.blocks.x, 1U);
}

/**
 * @brief Oversized stage count degrades to the floor without over-reading.
 */
TEST(FusedValidation, OversizedStageCountFloors) {
  hk::FusedOperatorParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.numStages = 200U;

  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 1U);
  EXPECT_EQ(cfg.blocks.x, 1U);
}

/**
 * @brief Self-referential fusion terminates instead of recursing forever.
 * @test A stage pointing at its own params degrades to the device-wide
 * grid past the nesting cap.
 */
TEST(FusedValidation, SelfCycleTerminates) {
  hk::FusedOperatorParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.stages[0].pattern = hk::ExecutionPattern::FusedOperator;
  p.stages[0].params = &p;
  p.numStages = 1U;

  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.threads.x, 256U);
  EXPECT_EQ(cfg.blocks.x, 64U);
}

/**
 * @brief Stateful partials shrink the single-block cap.
 * @test Tight shared budget: Add fits 2048, ArgMax only 1024.
 */
TEST(ReductionOps, StatefulPartialsShrinkCap) {
  auto make = [](hk::ReductionOp op) {
    hk::ReductionParams p{};
    p.device = testCaps();
    p.device.smemPerBlock = 64U;
    p.kernel = testKernel();
    p.reduceExtent = 1500ULL;
    p.numReductions = 1ULL;
    p.reduceContiguous = true;
    p.op = op;
    p.inputItemSize = 4U;
    p.accumItemSize = 4U;
    return p;
  };
  EXPECT_TRUE(hk::coversReductionFast(make(hk::ReductionOp::Add)));
  EXPECT_FALSE(hk::coversReductionFast(make(hk::ReductionOp::ArgMax)));
  EXPECT_FALSE(hk::coversReductionFast(make(hk::ReductionOp::MinMax)));
}

/**
 * @brief Welford triples shrink the cap threefold.
 */
TEST(ReductionOps, MeanVarTriplesCap) {
  hk::ReductionParams p{};
  p.device = testCaps();
  p.device.smemPerBlock = 64U;
  p.kernel = testKernel();
  p.reduceExtent = 700ULL;
  p.numReductions = 1ULL;
  p.reduceContiguous = true;
  p.op = hk::ReductionOp::MeanVar;
  p.inputItemSize = 4U;
  p.accumItemSize = 4U;
  EXPECT_FALSE(hk::coversReductionFast(p));
  p.op = hk::ReductionOp::Add;
  EXPECT_TRUE(hk::coversReductionFast(p));
}

/**
 * @brief Non-combinable ops keep full thread blocks.
 * @test Median at extent 512 skips the single-warp shrink Add takes.
 */
TEST(ReductionOps, MedianSkipsShuffleShrink) {
  auto make = [](hk::ReductionOp op) {
    hk::ReductionParams p{};
    p.device = testCaps();
    p.kernel = testKernel();
    p.reduceExtent = 512ULL;
    p.numReductions = 1ULL;
    p.reduceContiguous = true;
    p.op = op;
    p.inputItemSize = 4U;
    p.accumItemSize = 4U;
    return p;
  };
  ASSERT_TRUE(hk::coversReductionFast(make(hk::ReductionOp::Median)));
  const hk::LaunchConfig addCfg =
      hk::resolveReduction(make(hk::ReductionOp::Add));
  const hk::LaunchConfig medCfg =
      hk::resolveReduction(make(hk::ReductionOp::Median));
  EXPECT_EQ(addCfg.threads.x, 32U);
  EXPECT_EQ(medCfg.threads.x, 256U);
}

/**
 * @brief Custom op stays covered on small shapes with safe floor.
 */
TEST(ReductionOps, CustomCoveredSmall) {
  hk::ReductionParams p{};
  p.device = testCaps();
  p.kernel = testKernel();
  p.reduceExtent = 100ULL;
  p.numReductions = 2ULL;
  p.reduceContiguous = true;
  p.op = hk::ReductionOp::Custom;
  p.inputItemSize = 4U;
  p.accumItemSize = 4U;
  EXPECT_TRUE(hk::coversReductionFast(p));
  const hk::LaunchConfig cfg = hk::resolveLaunchConfig(p);
  EXPECT_EQ(cfg.blocks.x, 2U);
}
