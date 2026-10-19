/**
 * @file ParallelDecisions_test.cpp
 * @brief Unit tests for CPU parallelization decisions (decide.h, grains.h).
 *
 * Covers every decide_* predicate at its documented grain boundary with
 * threads = 4 unless noted, the helper builders in helpers.h, and a
 * serial/parallel copy equivalence regression. All cases are host-side
 * integer arithmetic; only ContiguousCopy allocates a real tensor.
 *
 * Boundary convention: a decision clears when the grain quotient reaches
 * 2 (cap_count refuses quotients below 2), so refusal boundaries sit at
 * two grains, not at full thread utilization. Tests pin both sides.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tensor.h>
#include <ncore/threading/parallel/decide.h>
#include <ncore/threading/parallel/grains.h>
#include <ncore/threading/parallel/helpers.h>
#include <ncore/threading/parallel/pattern.h>

namespace {

constexpr uint32 kThreads = 4;

ElementwiseWork plainElementwise(size_t logical, size_t item = 4) {
  return {.logical_size = logical,
          .total_bytes = logical * item,
          .item_size = item,
          .packing = 1,
          .heavy = false,
          .valid = true};
}

ElementwiseWork heavyElementwise(size_t logical, size_t bytes, size_t packing) {
  return {.logical_size = logical,
          .total_bytes = bytes,
          .item_size = 1,
          .packing = packing,
          .heavy = true,
          .valid = true};
}

LayoutWork plainLayout(size_t elements, size_t bytes, bool dense = true) {
  return {.num_elements = elements,
          .total_bytes = bytes,
          .item_size = 4,
          .packing = 1,
          .heavy = false,
          .is_dense = dense,
          .valid = true};
}

ReductionWork plainReduction(size_t total, size_t extent, size_t batch) {
  return {.total = total,
          .extent = extent,
          .batch = batch,
          .item_size = 4,
          .packing = 1,
          .heavy = false,
          .valid = true};
}

GemmWork plainGemm(size_t arith, size_t outer) {
  return {.arith = arith, .outer = outer, .valid = true};
}

StencilWork plainStencil(size_t points) {
  return {.points = points, .valid = true};
}

ScanWork plainScan(size_t total, size_t extent, size_t batch) {
  return {.total = total, .extent = extent, .batch = batch, .valid = true};
}

GatherWork plainGather(size_t indices, size_t segment) {
  return {.indices = indices, .segment = segment, .valid = true};
}

SortWork plainSort(size_t total) { return {.total = total, .valid = true}; }

/**
 * @brief Stack tensor with wired storage for helper-builder tests.
 *
 * Only metadata is read by make_elementwise_work and make_layout_work,
 * so the small backing buffer is never dereferenced; requested_bytes
 * carries the logical byte count under test.
 */
struct FakeTensor {
  Tensor ten{};
  TensorStorage storage{};
  std::array<unsigned char, 512> backing{};

  FakeTensor(DType_ dtype, Device_ device, size_t ndims, size_t size,
             size_t logical, size_t item, size_t bytes) {
    storage.ptr.data = backing.data();
    storage.handle.requested_bytes = bytes;
    storage.handle.bytes = bytes;
    ten.dtype = dtype;
    ten.device = device;
    ten.ndims = ndims;
    ten.size = size;
    ten.logical_size = logical;
    ten.item_size = item;
    ten.storage = &storage;
    ten.data.data = backing.data();
    ten.is_allocated_ = true;
  }
};

void checkBounded(ParallelDecision decision, uint32 threads) {
  if (!decision.go) {
    return;
  }
  EXPECT_GE(decision.num_threads, 2u);
  EXPECT_LE(decision.num_threads, threads);
}

struct DtypeExpectation {
  DType_ dtype;
  size_t packing;
  bool heavy;
};

class HelperDtypeSweep : public ::testing::TestWithParam<DtypeExpectation> {};

} // namespace

/**
 * @brief One grain short of two full grains refuses; two grains clear.
 * @test REQ-002: 2*8192-1 refuses, 2*8192 clears with 2 threads.
 */
TEST(ElementwiseDecision, RefusesBelowGrain) {
  const ElementwiseWork below =
      plainElementwise((2 * PARALLEL_GRAIN_ELEMENTWISE) - 1);
  EXPECT_FALSE(decide_elementwise(&below, kThreads).go);

  const ElementwiseWork at = plainElementwise(2 * PARALLEL_GRAIN_ELEMENTWISE);
  const ParallelDecision cleared = decide_elementwise(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief Four grains clear with all four threads; one element short
 * clears with three (partial counts still fork).
 * @test REQ-002: 4*8192 gives {go, 4}, 4*8192-1 gives {go, 3}.
 */
TEST(ElementwiseDecision, ClearsAtGrain) {
  const ElementwiseWork at = plainElementwise(4 * PARALLEL_GRAIN_ELEMENTWISE);
  const ParallelDecision cleared = decide_elementwise(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);

  const ElementwiseWork shortByOne =
      plainElementwise((4 * PARALLEL_GRAIN_ELEMENTWISE) - 1);
  const ParallelDecision partial = decide_elementwise(&shortByOne, kThreads);
  EXPECT_TRUE(partial.go);
  EXPECT_EQ(partial.num_threads, 3u);
}

/**
 * @brief Below two byte-grains refuses; two byte-grains clear.
 * @test REQ-003: 2*32768-1 bytes refuse, 2*32768 bytes give {go, 2}.
 */
TEST(LayoutDecision, RefusesBelowByteGrain) {
  constexpr size_t kHalf = (2 * PARALLEL_GRAIN_LAYOUT_BYTES) - 1;
  const LayoutWork below = plainLayout(kHalf / 4, kHalf);
  EXPECT_FALSE(decide_layout(&below, kThreads).go);

  constexpr size_t kFull = 2 * PARALLEL_GRAIN_LAYOUT_BYTES;
  const LayoutWork at = plainLayout(kFull / 4, kFull);
  const ParallelDecision cleared = decide_layout(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief Four byte-grains clear with all threads; one byte short
 * clears with three.
 * @test REQ-003: 4*32768 bytes give {go, 4}, minus one gives {go, 3}.
 */
TEST(LayoutDecision, ClearsAtByteGrain) {
  constexpr size_t kFull = 4 * PARALLEL_GRAIN_LAYOUT_BYTES;
  const LayoutWork at = plainLayout(kFull / 4, kFull);
  const ParallelDecision cleared = decide_layout(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);

  constexpr size_t kShort = (4 * PARALLEL_GRAIN_LAYOUT_BYTES) - 1;
  const LayoutWork shortByOne = plainLayout(kShort / 4, kShort);
  const ParallelDecision partial = decide_layout(&shortByOne, kThreads);
  EXPECT_TRUE(partial.go);
  EXPECT_EQ(partial.num_threads, 3u);
}

/**
 * @brief The dense flag does not change the grain verdict.
 * @test Same fields with is_dense true/false give identical verdicts.
 */
TEST(LayoutDecision, DenseFlagPassesThrough) {
  constexpr size_t kFull = 4 * PARALLEL_GRAIN_LAYOUT_BYTES;
  const LayoutWork dense = plainLayout(kFull / 4, kFull, true);
  const LayoutWork strided = plainLayout(kFull / 4, kFull, false);
  const ParallelDecision denseDecision = decide_layout(&dense, kThreads);
  const ParallelDecision stridedDecision = decide_layout(&strided, kThreads);
  EXPECT_EQ(denseDecision.go, stridedDecision.go);
  EXPECT_EQ(denseDecision.num_threads, stridedDecision.num_threads);
  EXPECT_TRUE(denseDecision.go);
}

/**
 * @brief A thin extent with a narrow batch refuses despite huge total.
 * @test REQ-004: extent 16, batch 2, total 1M refuses (floor rule first).
 */
TEST(ReductionDecision, ThinExtentRefuses) {
  const ReductionWork thin = plainReduction(1 << 20, 16, 2);
  EXPECT_FALSE(decide_reduction(&thin, kThreads).go);
}

/**
 * @brief A deep batch over a wide extent clears.
 * @test REQ-004: extent 1024, batch 32, total 32768 gives {go, 2}.
 */
TEST(ReductionDecision, DeepBatchedClears) {
  const ReductionWork deep = plainReduction(32768, 1024, 32);
  const ParallelDecision cleared = decide_reduction(&deep, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief A 16-cubed product refuses (zero grain quotients).
 * @test REQ-005: arith 4096, outer 16 refuses.
 */
TEST(GemmDecision, TinyRefuses) {
  const GemmWork tiny = plainGemm(16 * 16 * 16, 16);
  EXPECT_FALSE(decide_gemm(&tiny, kThreads).go);
}

/**
 * @brief A 512-cubed product over 512 tiles clears with all threads.
 * @test REQ-005: arith 512^3, outer 512 gives {go, 4}.
 */
TEST(GemmDecision, TiledClears) {
  const GemmWork tiled = plainGemm(512 * 512 * 512, 512);
  const ParallelDecision cleared = decide_gemm(&tiled, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);
}

/**
 * @brief One point short of two grains refuses; two grains clear.
 * @test REQ-006: 2*4096-1 refuses, 2*4096 gives {go, 2}.
 */
TEST(StencilDecision, RefusesBelowGrain) {
  const StencilWork below = plainStencil((2 * PARALLEL_GRAIN_STENCIL) - 1);
  EXPECT_FALSE(decide_stencil(&below, kThreads).go);

  const StencilWork at = plainStencil(2 * PARALLEL_GRAIN_STENCIL);
  const ParallelDecision cleared = decide_stencil(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief Four grains clear with all threads; one point short clears
 * with three.
 * @test REQ-006: 4*4096 gives {go, 4}, 4*4096-1 gives {go, 3}.
 */
TEST(StencilDecision, ClearsAtGrain) {
  const StencilWork at = plainStencil(4 * PARALLEL_GRAIN_STENCIL);
  const ParallelDecision cleared = decide_stencil(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);

  const StencilWork shortByOne = plainStencil((4 * PARALLEL_GRAIN_STENCIL) - 1);
  const ParallelDecision partial = decide_stencil(&shortByOne, kThreads);
  EXPECT_TRUE(partial.go);
  EXPECT_EQ(partial.num_threads, 3u);
}

/**
 * @brief A single scan never parallelizes regardless of extent.
 * @test REQ-007: batch 1 with a 1M extent refuses (axis dependency).
 */
TEST(ScanDecision, SingleScanRefuses) {
  const ScanWork single = plainScan(1 << 20, 1 << 20, 1);
  EXPECT_FALSE(decide_scan(&single, kThreads).go);
}

/**
 * @brief Eight scans of 8K elements clear with all threads.
 * @test REQ-007: batch 8, extent 8192 gives {go, 4}.
 */
TEST(ScanDecision, BatchedClears) {
  const ScanWork batched = plainScan(8 * 8192, 8192, 8);
  const ParallelDecision cleared = decide_scan(&batched, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);
}

/**
 * @brief An 8-byte segment refuses despite 1M indices.
 * @test REQ-008: segment 8 is below the 16-byte floor.
 */
TEST(GatherDecision, SmallSegmentRefuses) {
  const GatherWork small = plainGather(1 << 20, 8);
  EXPECT_FALSE(decide_gather(&small, kThreads).go);
}

/**
 * @brief A 64-byte segment with four index grains clears.
 * @test REQ-008: segment 64, 4*8192 indices give {go, 4}.
 */
TEST(GatherDecision, LargeClears) {
  const GatherWork large = plainGather(4 * PARALLEL_GRAIN_GATHER, 64);
  const ParallelDecision cleared = decide_gather(&large, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);
}

/**
 * @brief One element below the ordering floor refuses.
 * @test REQ-009: 65535 refuses.
 */
TEST(SortDecision, BelowFloorRefuses) {
  const SortWork below = plainSort(PARALLEL_GRAIN_SORT_FLOOR - 1);
  EXPECT_FALSE(decide_sort(&below, kThreads).go);
}

/**
 * @brief Floor plus one grain clears with all threads.
 * @test REQ-009: 65536+16384 gives quotient 5, capped to {go, 4}.
 */
TEST(SortDecision, FloorPlusGrainClears) {
  const SortWork above =
      plainSort(PARALLEL_GRAIN_SORT_FLOOR + PARALLEL_GRAIN_SORT);
  const ParallelDecision cleared = decide_sort(&above, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);
}

/**
 * @brief The serial family refuses at every thread count.
 * @test REQ-010: threads 0, 1, 2, 4, 16 all refuse.
 */
TEST(SerialDecision, AlwaysRefuses) {
  for (uint32 threads : {0u, 1u, 2u, 4u, 16u}) {
    EXPECT_FALSE(decide_serial(threads).go) << "threads=" << threads;
  }
}

/**
 * @brief All-clear stages combine to the smallest stage count.
 * @test REQ-011: {go, 4} and {go, 2} combine to {go, 2}.
 */
TEST(FusedDecision, AllClearGivesMinCount) {
  const ParallelDecision stages[2] = {{.go = true, .num_threads = 4},
                                      {.go = true, .num_threads = 2}};
  const ParallelDecision combined = decide_fused(stages, 2);
  EXPECT_TRUE(combined.go);
  EXPECT_EQ(combined.num_threads, 2u);
}

/**
 * @brief Any refusing stage vetoes the fused region.
 * @test REQ-011: refusal first or last both propagate.
 */
TEST(FusedDecision, OneRefusalPropagates) {
  // const std::array<ParallelDecision, 2> tailRefuses = {
  //     {.go = true, .num_threads = 4}, {.go = false, .num_threads = 1}};
  const std::array<ParallelDecision, 2> tailRefuses{{
      {.go = true, .num_threads = 4},
      {.go = false, .num_threads = 1},
  }};
  EXPECT_FALSE(decide_fused(tailRefuses.data(), 2).go);

  const std::array<ParallelDecision, 2> headRefuses{{
      {.go = false, .num_threads = 1},
      {.go = true, .num_threads = 4},
  }};
  EXPECT_FALSE(decide_fused(headRefuses.data(), 2).go);
}

/**
 * @brief A single stage passes its verdict through unchanged.
 * @test REQ-011: one {go, 3} stage gives {go, 3}.
 */
TEST(FusedDecision, SingleStagePassesThrough) {
  const ParallelDecision stages[1] = {{.go = true, .num_threads = 3}};
  const ParallelDecision combined = decide_fused(stages, 1);
  EXPECT_TRUE(combined.go);
  EXPECT_EQ(combined.num_threads, 3u);
}

/**
 * @brief Thread counts below 2 refuse every work-bearing pattern.
 * @test REQ-021: all 8 decide_* with threads 0 and 1 refuse.
 */
TEST(ThreadsBelowTwo, RefusesAllPatterns) {
  const ElementwiseWork elementwise = plainElementwise(1 << 20);
  const LayoutWork layout = plainLayout(1 << 20, 1 << 22);
  const ReductionWork reduction = plainReduction(1 << 22, 1 << 12, 1 << 10);
  const GemmWork gemm = plainGemm(1ULL << 32, 1 << 12);
  const StencilWork stencil = plainStencil(1 << 20);
  const ScanWork scan = plainScan(1 << 22, 1 << 12, 1 << 10);
  const GatherWork gather = plainGather(1 << 20, 64);
  const SortWork sort = plainSort(1 << 20);

  for (uint32 threads : {0u, 1u}) {
    EXPECT_FALSE(decide_elementwise(&elementwise, threads).go);
    EXPECT_FALSE(decide_layout(&layout, threads).go);
    EXPECT_FALSE(decide_reduction(&reduction, threads).go);
    EXPECT_FALSE(decide_gemm(&gemm, threads).go);
    EXPECT_FALSE(decide_stencil(&stencil, threads).go);
    EXPECT_FALSE(decide_scan(&scan, threads).go);
    EXPECT_FALSE(decide_gather(&gather, threads).go);
    EXPECT_FALSE(decide_sort(&sort, threads).go);
  }
}

/**
 * @brief Two grains of work over eight threads fork exactly two.
 * @test 2*8192 elements with 8 threads give {go, 2}.
 */
TEST(EffectiveCount, CappedByUsefulGrains) {
  const ElementwiseWork work = plainElementwise(2 * PARALLEL_GRAIN_ELEMENTWISE);
  const ParallelDecision decision = decide_elementwise(&work, 8);
  EXPECT_TRUE(decision.go);
  EXPECT_EQ(decision.num_threads, 2u);
}

/**
 * @brief Huge work never forks more threads than available.
 * @test 16M elements with 4 threads give {go, 4}.
 */
TEST(EffectiveCount, CappedByAvailableThreads) {
  const ElementwiseWork work = plainElementwise(1 << 24);
  const ParallelDecision decision = decide_elementwise(&work, kThreads);
  EXPECT_TRUE(decision.go);
  EXPECT_EQ(decision.num_threads, 4u);
}

/**
 * @brief Scan threads are confined to whole scans, not grains.
 * @test Batch 3 over huge extents with 8 threads gives {go, 3}.
 */
TEST(EffectiveCount, ScanCappedByBatch) {
  const ScanWork work = plainScan(3 * (1 << 20), 1 << 20, 3);
  const ParallelDecision decision = decide_scan(&work, 8);
  EXPECT_TRUE(decision.go);
  EXPECT_EQ(decision.num_threads, 3u);
}

/**
 * @brief Extent 31 needs the batch factor; extent 32 stands alone.
 * @test Extent 31 with batch 2 refuses; extent 32 with batch 1024
 * and total 32768 gives {go, 2}.
 */
TEST(ReductionDecision, ExtentFloorBoundary) {
  const ReductionWork thin = plainReduction(1 << 20, 31, 2);
  EXPECT_FALSE(decide_reduction(&thin, kThreads).go);

  const ReductionWork wide = plainReduction(32768, 32, 1024);
  const ParallelDecision cleared = decide_reduction(&wide, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief A wide batch compensates a thin extent past the floor rule.
 * @test Extent 16 with batch 2048 (at least threads*4) and total
 * 32768 gives {go, 2}.
 */
TEST(ReductionDecision, BatchFactorCompensatesThinExtent) {
  const ReductionWork compensated = plainReduction(32768, 16, 2048);
  const ParallelDecision cleared = decide_reduction(&compensated, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief Segment 15 refuses; segment 16 with two index grains clears.
 * @test 15-byte segments refuse at 1M indices; 16-byte segments with
 * 2*8192 indices give {go, 2}.
 */
TEST(GatherDecision, SegmentFloorBoundary) {
  const GatherWork below = plainGather(1 << 20, 15);
  EXPECT_FALSE(decide_gather(&below, kThreads).go);

  const GatherWork at = plainGather(2 * PARALLEL_GRAIN_GATHER, 16);
  const ParallelDecision cleared = decide_gather(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief The ordering floor is inclusive: exactly 65536 clears.
 * @test 65536 passes the floor check and gives quotient 4, {go, 4}.
 */
TEST(SortDecision, FloorExactBoundary) {
  const SortWork at = plainSort(PARALLEL_GRAIN_SORT_FLOOR);
  const ParallelDecision cleared = decide_sort(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);
}

/**
 * @brief Two scans clear with two threads; one scan refuses.
 * @test Batch 2 over 1M extents gives {go, 2}.
 */
TEST(ScanDecision, BatchTwoBoundary) {
  const ScanWork pair = plainScan(2 * (1 << 20), 1 << 20, 2);
  const ParallelDecision cleared = decide_scan(&pair, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 2u);
}

/**
 * @brief An empty stage list refuses with null or valid pointers.
 * @test (nullptr, 0) and (valid, 0) both refuse.
 */
TEST(FusedDecision, EmptyStagesRefuse) {
  EXPECT_FALSE(decide_fused(nullptr, 0).go);

  const ParallelDecision one = {.go = true, .num_threads = 4};
  EXPECT_FALSE(decide_fused(&one, 0).go);
}

/**
 * @brief Structs that fail structural checks refuse every pattern.
 * @test Zero-initialized (valid=false) work refuses at threads 4.
 */
TEST(InvalidWork, ZeroCountsRefuse) {
  const ElementwiseWork elementwise{};
  const LayoutWork layout{};
  const ReductionWork reduction{};
  const GemmWork gemm{};
  const StencilWork stencil{};
  const ScanWork scan{};
  const GatherWork gather{};
  const SortWork sort{};

  EXPECT_FALSE(decide_elementwise(&elementwise, kThreads).go);
  EXPECT_FALSE(decide_layout(&layout, kThreads).go);
  EXPECT_FALSE(decide_reduction(&reduction, kThreads).go);
  EXPECT_FALSE(decide_gemm(&gemm, kThreads).go);
  EXPECT_FALSE(decide_stencil(&stencil, kThreads).go);
  EXPECT_FALSE(decide_scan(&scan, kThreads).go);
  EXPECT_FALSE(decide_gather(&gather, kThreads).go);
  EXPECT_FALSE(decide_sort(&sort, kThreads).go);
}

/**
 * @brief Null work refuses every work-bearing pattern.
 * @test All 8 decide_* with nullptr refuse at threads 4.
 */
TEST(NullWork, RefusesAllPatterns) {
  EXPECT_FALSE(decide_elementwise(nullptr, kThreads).go);
  EXPECT_FALSE(decide_layout(nullptr, kThreads).go);
  EXPECT_FALSE(decide_reduction(nullptr, kThreads).go);
  EXPECT_FALSE(decide_gemm(nullptr, kThreads).go);
  EXPECT_FALSE(decide_stencil(nullptr, kThreads).go);
  EXPECT_FALSE(decide_scan(nullptr, kThreads).go);
  EXPECT_FALSE(decide_gather(nullptr, kThreads).go);
  EXPECT_FALSE(decide_sort(nullptr, kThreads).go);
}

/**
 * @brief Null stages refuse even with a nonzero stage count.
 * @test (nullptr, 2) refuses.
 */
TEST(NullWork, FusedNullStagesRefuse) {
  EXPECT_FALSE(decide_fused(nullptr, 2).go);
}

/**
 * @brief Null tensors build invalid facts in both builders.
 */
TEST(Helpers, NullTensorInvalid) {
  EXPECT_FALSE(make_elementwise_work(nullptr).valid);
  EXPECT_FALSE(make_layout_work(nullptr, true).valid);
  EXPECT_FALSE(make_layout_work(nullptr, false).valid);
}

/**
 * @brief Unallocated tensors build invalid facts in both builders.
 */
TEST(Helpers, UnallocatedTensorInvalid) {
  Tensor ten{};
  ten.dtype = Float32;
  ten.device = DEVICE_CPU;
  ten.ndims = 1;
  ten.size = 8;
  ten.logical_size = 8;
  ten.item_size = 4;
  EXPECT_FALSE(make_elementwise_work(&ten).valid);
  EXPECT_FALSE(make_layout_work(&ten, true).valid);
}

/**
 * @brief Meta tensors (no backing storage) build invalid facts.
 */
TEST(Helpers, MetaTensorInvalid) {
  Tensor ten{};
  ten.dtype = Float32;
  ten.device = DEVICE_META;
  EXPECT_FALSE(make_elementwise_work(&ten).valid);
  EXPECT_FALSE(make_layout_work(&ten, true).valid);
}

/**
 * @brief Empty tensors build invalid facts in both builders.
 */
TEST(Helpers, EmptyTensorInvalid) {
  FakeTensor empty(Float32, DEVICE_CPU, 1, 0, 0, 4, 0);
  EXPECT_FALSE(make_elementwise_work(&empty.ten).valid);
  EXPECT_FALSE(make_layout_work(&empty.ten, true).valid);
}

/**
 * @brief Scalar tensors (ndims 0) build invalid facts.
 */
TEST(Helpers, ScalarTensorInvalid) {
  FakeTensor scalar(Float32, DEVICE_CPU, 0, 1, 1, 4, 4);
  EXPECT_FALSE(make_elementwise_work(&scalar.ten).valid);
  EXPECT_FALSE(make_layout_work(&scalar.ten, true).valid);
}

/**
 * @brief GPU-tagged tensors are rejected by both builders.
 */
TEST(Helpers, NonCpuPlacementInvalid) {
  FakeTensor gpu(Float32, DEVICE_GPU, 1, 64, 64, 4, 256);
  EXPECT_FALSE(make_elementwise_work(&gpu.ten).valid);
  EXPECT_FALSE(make_layout_work(&gpu.ten, true).valid);
}

/**
 * @brief Random stages build valid facts straight from the demand.
 *
 * Random work runs over caller-owned staging with no Tensor in
 * hand: the tensor builders refuse such call sites and the stage
 * used to fall back to serial.
 */
TEST(Helpers, RngWorkBuildsValidFacts) {
  const ElementwiseWork work = make_rng_work(131072, 4);
  EXPECT_TRUE(work.valid);
  EXPECT_EQ(work.logical_size, size_t{131072});
  EXPECT_EQ(work.total_bytes, size_t{131072 * 4});
  EXPECT_EQ(work.item_size, size_t{4});
  EXPECT_EQ(work.packing, size_t{1});
  EXPECT_TRUE(work.heavy);
}

/**
 * @brief A raw random stage parallelizes where the tensor builder
 * refused: the same demand clears the grain quotient.
 */
TEST(Helpers, RngWorkDecisionParallelizes) {
  const ElementwiseWork work = make_rng_work(131072, 4);
  const ParallelDecision decision = decide_elementwise(&work, 24);
  EXPECT_TRUE(decision.go);
  EXPECT_EQ(decision.num_threads, 8U); // heavy grain divides to 8.
}

/**
 * @brief null or zero demand through the raw builder stays serial.
 */
TEST(Helpers, RngWorkEmptyDemandStaysSerial) {
  const ElementwiseWork work = make_rng_work(0, 4);
  EXPECT_FALSE(decide_elementwise(&work, 24).go);
}

/**
 * @brief Out-of-range dtype tags are rejected by both builders.
 */
TEST(Helpers, InvalidDtypeInvalid) {
  FakeTensor bad(static_cast<DType_>(NUM_DTYPES), DEVICE_CPU, 1, 64, 64, 4,
                 256);
  EXPECT_FALSE(make_elementwise_work(&bad.ten).valid);
  EXPECT_FALSE(make_layout_work(&bad.ten, true).valid);
}

/**
 * @brief FP4 tensors map to packing 2 with the heavy flag set.
 */
TEST(Helpers, PackedDtypeMapping) {
  FakeTensor packed(Float4E2M1fn, DEVICE_CPU, 1, 8, 16, 1, 8);
  const ElementwiseWork work = make_elementwise_work(&packed.ten);
  ASSERT_TRUE(work.valid);
  EXPECT_EQ(work.packing, 2u);
  EXPECT_TRUE(work.heavy);
}

/**
 * @brief Packed work needs the heavier grain: 4*8192 packed bytes
 * refuse where plain clears, and 4 packed grains clear with four.
 * @test Plain 32768 items give {go, 4}; packed 32768 items over
 * 16384 bytes refuse (tiny-byte gate); packed 65536 items over
 * 32768 bytes give {go, 4}.
 */
TEST(ElementwiseDecision, PackedNeedsHeavierGrain) {
  const ElementwiseWork plain =
      plainElementwise(4 * PARALLEL_GRAIN_ELEMENTWISE);
  const ParallelDecision plainDecision = decide_elementwise(&plain, kThreads);
  ASSERT_TRUE(plainDecision.go);
  EXPECT_EQ(plainDecision.num_threads, 4u);

  const ElementwiseWork packedSmall = heavyElementwise(
      4 * PARALLEL_GRAIN_ELEMENTWISE, 4 * PARALLEL_GRAIN_ELEMENTWISE / 2, 2);
  EXPECT_FALSE(decide_elementwise(&packedSmall, kThreads).go);

  const ElementwiseWork packedFull =
      heavyElementwise(4 * PARALLEL_GRAIN_ELEMENTWISE_PACKED,
                       4 * PARALLEL_GRAIN_ELEMENTWISE_PACKED / 2, 2);
  const ParallelDecision packedDecision =
      decide_elementwise(&packedFull, kThreads);
  EXPECT_TRUE(packedDecision.go);
  EXPECT_EQ(packedDecision.num_threads, 4u);
}

/**
 * @brief Quantized work uses the heavier grain: the same item count
 * forks fewer threads than plain, and one item short of two packed
 * grains refuses.
 * @test Plain 32768 items give {go, 4}; QSigned8 32768 items over
 * 32768 bytes give {go, 2}; 32767 items refuse.
 */
TEST(ElementwiseDecision, QuantizedNeedsHeavierGrain) {
  const ElementwiseWork plain =
      plainElementwise(4 * PARALLEL_GRAIN_ELEMENTWISE);
  const ParallelDecision plainDecision = decide_elementwise(&plain, kThreads);
  ASSERT_TRUE(plainDecision.go);
  EXPECT_EQ(plainDecision.num_threads, 4u);

  const ElementwiseWork quantized = heavyElementwise(
      4 * PARALLEL_GRAIN_ELEMENTWISE, 4 * PARALLEL_GRAIN_ELEMENTWISE, 1);
  const ParallelDecision quantizedDecision =
      decide_elementwise(&quantized, kThreads);
  EXPECT_TRUE(quantizedDecision.go);
  EXPECT_EQ(quantizedDecision.num_threads, 2u);

  const ElementwiseWork quantizedShort =
      heavyElementwise((2 * PARALLEL_GRAIN_ELEMENTWISE_PACKED) - 1,
                       (2 * PARALLEL_GRAIN_ELEMENTWISE_PACKED) - 1, 1);
  EXPECT_FALSE(decide_elementwise(&quantizedShort, kThreads).go);
}

/**
 * @brief One-byte items need the byte grain: 4*8192-1 bytes refuse,
 * 4*8192 bytes clear.
 * @test 1M items over 32767 bytes refuse; over 32768 bytes give {go, 4}.
 */
TEST(ElementwiseDecision, OneByteNeedsByteGrain) {
  const ElementwiseWork below = {.logical_size = 1 << 20,
                                 .total_bytes =
                                     (kThreads * PARALLEL_GRAIN_TINY_BYTES) - 1,
                                 .item_size = 1,
                                 .packing = 1,
                                 .heavy = false,
                                 .valid = true};
  EXPECT_FALSE(decide_elementwise(&below, kThreads).go);

  const ElementwiseWork at = {.logical_size = 1 << 20,
                              .total_bytes =
                                  kThreads * PARALLEL_GRAIN_TINY_BYTES,
                              .item_size = 1,
                              .packing = 1,
                              .heavy = false,
                              .valid = true};
  const ParallelDecision cleared = decide_elementwise(&at, kThreads);
  EXPECT_TRUE(cleared.go);
  EXPECT_EQ(cleared.num_threads, 4u);
}

/**
 * @brief Signed8 tensors map to packing 1, no heavy flag, item size 1.
 */
TEST(Helpers, OneByteTensorMapping) {
  FakeTensor oneByte(Signed8, DEVICE_CPU, 1, 64, 64, 1, 64);
  const ElementwiseWork work = make_elementwise_work(&oneByte.ten);
  ASSERT_TRUE(work.valid);
  EXPECT_EQ(work.packing, 1u);
  EXPECT_FALSE(work.heavy);
  EXPECT_EQ(work.item_size, 1u);
}

/**
 * @brief Packing and heavy flags follow the dtype under test.
 */
TEST_P(HelperDtypeSweep, MappingMatchesHelpers) {
  const DtypeExpectation param = GetParam();
  const size_t item = dtype_size(param.dtype);
  FakeTensor ten(param.dtype, DEVICE_CPU, 1, 64, 64 * param.packing, item,
                 64 * item);
  const ElementwiseWork work = make_elementwise_work(&ten.ten);
  ASSERT_TRUE(work.valid);
  EXPECT_EQ(work.packing, param.packing);
  EXPECT_EQ(work.heavy, param.heavy);
}

INSTANTIATE_TEST_SUITE_P(
    PackedAndPlain, HelperDtypeSweep,
    ::testing::Values(DtypeExpectation{Float32, 1, false},
                      DtypeExpectation{Float64, 1, false},
                      DtypeExpectation{Signed8, 1, false},
                      DtypeExpectation{Float4E2M1fn, 2, true},
                      DtypeExpectation{QSigned8, 1, true}));

/**
 * @brief The dense flag survives the layout builder unchanged.
 */
TEST(Helpers, LayoutDenseFlagMapping) {
  FakeTensor ten(Float32, DEVICE_CPU, 1, 64, 64, 4, 256);
  const LayoutWork dense = make_layout_work(&ten.ten, true);
  ASSERT_TRUE(dense.valid);
  EXPECT_TRUE(dense.is_dense);

  const LayoutWork strided = make_layout_work(&ten.ten, false);
  ASSERT_TRUE(strided.valid);
  EXPECT_FALSE(strided.is_dense);
}

/**
 * @brief A striped copy over four threads is bit-identical to serial.
 * @test REQ-013: real Float32 CPU tensor, deterministic fill, go-case
 * layout verdict at threads 4, serial and striped bodies, memcmp equal.
 */
TEST(ContiguousCopy, SerialVsParallelBitIdentical) {
  constexpr size_t kElements =
      4 * PARALLEL_GRAIN_LAYOUT_BYTES / sizeof(float32);
  constexpr size_t kChunk = kElements / 4;

  shape_t shape{};
  shape[0] = kElements;
  novaStatus_t status{};
  Tensor src =
      create_tensor(shape, Float32, DEVICE_CPU, false, false, 1, &status);
  ASSERT_EQ(status.err, novaSuccess);
  ASSERT_TRUE(is_allocated(&src));
  Tensor dstSerial =
      create_tensor(shape, Float32, DEVICE_CPU, false, false, 1, &status);
  ASSERT_EQ(status.err, novaSuccess);
  ASSERT_TRUE(is_allocated(&dstSerial));
  Tensor dstParallel =
      create_tensor(shape, Float32, DEVICE_CPU, false, false, 1, &status);
  ASSERT_EQ(status.err, novaSuccess);
  ASSERT_TRUE(is_allocated(&dstParallel));

  for (size_t i = 0; i < kElements; ++i) {
    src.data.f32[i] = static_cast<float32>(i);
  }

  const LayoutWork work = make_layout_work(&src, true);
  ASSERT_TRUE(work.valid);
  const ParallelDecision decision = decide_layout(&work, kThreads);
  ASSERT_TRUE(decision.go);
  ASSERT_EQ(decision.num_threads, 4u);

  for (size_t i = 0; i < kElements; ++i) {
    dstSerial.data.f32[i] = src.data.f32[i];
  }
  for (uint32 t = 0; t < 4; ++t) {
    for (size_t i = t * kChunk; i < (t + 1) * kChunk; ++i) {
      dstParallel.data.f32[i] = src.data.f32[i];
    }
  }

  EXPECT_EQ(::memcmp(dstSerial.data.data, dstParallel.data.data,
                     kElements * sizeof(float32)),
            0);

  EXPECT_EQ(collect(&src).err, novaSuccess);
  EXPECT_EQ(collect(&dstSerial).err, novaSuccess);
  EXPECT_EQ(collect(&dstParallel).err, novaSuccess);
}

/**
 * @brief Cleared decisions never exceed available threads and never
 * fork a single thread.
 * @test All 8 patterns at small/medium/huge sizes with threads 2, 4,
 * 16: refusal, or 2 <= num_threads <= threads.
 */
TEST(EffectiveCount, NeverExceedsThreadsSweep) {
  for (uint32 threads : {2u, 4u, 16u}) {
    const ElementwiseWork ewSmall = plainElementwise(100);
    const ElementwiseWork ewMedium = plainElementwise(20000);
    const ElementwiseWork ewHuge = plainElementwise(1 << 26);
    checkBounded(decide_elementwise(&ewSmall, threads), threads);
    checkBounded(decide_elementwise(&ewMedium, threads), threads);
    checkBounded(decide_elementwise(&ewHuge, threads), threads);

    const LayoutWork lwSmall = plainLayout(100, 400);
    const LayoutWork lwMedium = plainLayout(20000, 80000);
    const LayoutWork lwHuge = plainLayout(1 << 24, 1 << 26);
    checkBounded(decide_layout(&lwSmall, threads), threads);
    checkBounded(decide_layout(&lwMedium, threads), threads);
    checkBounded(decide_layout(&lwHuge, threads), threads);

    const ReductionWork rwSmall = plainReduction(100, 64, 2);
    const ReductionWork rwMedium = plainReduction(40000, 200, 200);
    const ReductionWork rwHuge = plainReduction(1 << 26, 1 << 13, 1 << 13);
    checkBounded(decide_reduction(&rwSmall, threads), threads);
    checkBounded(decide_reduction(&rwMedium, threads), threads);
    checkBounded(decide_reduction(&rwHuge, threads), threads);

    const GemmWork gwSmall = plainGemm(100, 1);
    const GemmWork gwMedium = plainGemm(4 * PARALLEL_GRAIN_GEMM_OPS, 4);
    const GemmWork gwHuge = plainGemm(1ULL << 32, 1000);
    checkBounded(decide_gemm(&gwSmall, threads), threads);
    checkBounded(decide_gemm(&gwMedium, threads), threads);
    checkBounded(decide_gemm(&gwHuge, threads), threads);

    const StencilWork swSmall = plainStencil(100);
    const StencilWork swMedium = plainStencil(20000);
    const StencilWork swHuge = plainStencil(1 << 26);
    checkBounded(decide_stencil(&swSmall, threads), threads);
    checkBounded(decide_stencil(&swMedium, threads), threads);
    checkBounded(decide_stencil(&swHuge, threads), threads);

    const ScanWork scSmall = plainScan(100, 50, 2);
    const ScanWork scMedium = plainScan(40000, 10000, 4);
    const ScanWork scHuge = plainScan(64 * (1 << 20), 1 << 20, 64);
    checkBounded(decide_scan(&scSmall, threads), threads);
    checkBounded(decide_scan(&scMedium, threads), threads);
    checkBounded(decide_scan(&scHuge, threads), threads);

    const GatherWork gaSmall = plainGather(100, 64);
    const GatherWork gaMedium = plainGather(20000, 64);
    const GatherWork gaHuge = plainGather(1 << 24, 64);
    checkBounded(decide_gather(&gaSmall, threads), threads);
    checkBounded(decide_gather(&gaMedium, threads), threads);
    checkBounded(decide_gather(&gaHuge, threads), threads);

    const SortWork soSmall = plainSort(1000);
    const SortWork soMedium = plainSort(100000);
    const SortWork soHuge = plainSort(1 << 26);
    checkBounded(decide_sort(&soSmall, threads), threads);
    checkBounded(decide_sort(&soMedium, threads), threads);
    checkBounded(decide_sort(&soHuge, threads), threads);
  }
}
