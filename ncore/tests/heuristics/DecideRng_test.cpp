/**
 * @file DecideRng_test.cpp
 * @brief Unit tests for the counter-based fill decision (decide_rng).
 *
 * Same boundary convention as ParallelDecisions_test: a verdict
 * clears when the weighted-lane quotient reaches 2, so the u32 and
 * u64 splits sit at two weighted grains and the weight itself is
 * pinned by the equivalence between the widths.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <ncore/threading/parallel/decide.h>
#include <ncore/threading/parallel/grains.h>
#include <ncore/threading/parallel/pattern.h>

namespace {

constexpr uint32 kThreads = 4;

RngWork rngWork(size_t lanes, size_t cost) {
  return {.lanes = lanes, .cost_per_elem = cost, .valid = true};
}

TEST(DecideRng, RefusesTheSharedFloors) {
  EXPECT_FALSE(decide_rng(nullptr, kThreads).go);

  const RngWork empty = rngWork(0, 2);
  EXPECT_FALSE(decide_rng(&empty, kThreads).go);

  const RngWork small = rngWork(1, 2);
  const ParallelDecision oneThread = decide_rng(&small, 1);
  EXPECT_FALSE(oneThread.go);
  EXPECT_EQ(1, oneThread.num_threads);
}

TEST(DecideRng, SplitsAtTheWeightedGrainBoundary) {
  // u32 weight 2: the quotient hits 2 at 65536 lanes.
  const RngWork under = rngWork(65535, 2);
  EXPECT_FALSE(decide_rng(&under, kThreads).go);

  const RngWork atBound = rngWork(65536, 2);
  const ParallelDecision at = decide_rng(&atBound, kThreads);
  EXPECT_TRUE(at.go);
  EXPECT_EQ(2, at.num_threads);

  // u64 weight 4: the same quotient lands one width earlier.
  const RngWork u64Under = rngWork(32767, 4);
  EXPECT_FALSE(decide_rng(&u64Under, kThreads).go);

  const RngWork u64AtBound = rngWork(32768, 4);
  const ParallelDecision u64At = decide_rng(&u64AtBound, kThreads);
  EXPECT_TRUE(u64At.go);
  EXPECT_EQ(2, u64At.num_threads);
}

TEST(DecideRng, WeightsFoldWidthsOntoOneRule) {
  // One 64-bit pair costs two 32-bit lanes, so mirrored demands take
  // the same verdict at both widths.
  const RngWork u32Demand = rngWork(262144, 2);
  const RngWork u64Demand = rngWork(131072, 4);
  const ParallelDecision u32 = decide_rng(&u32Demand, kThreads);
  const ParallelDecision u64 = decide_rng(&u64Demand, kThreads);
  EXPECT_TRUE(u32.go);
  EXPECT_TRUE(u64.go);
  EXPECT_EQ(u32.num_threads, u64.num_threads);
}

TEST(DecideRng, CapsTheForkAtAvailableThreads) {
  const RngWork demand = rngWork(8 * PARALLEL_GRAIN_RNG, 2);
  const ParallelDecision capped = decide_rng(&demand, kThreads);
  EXPECT_TRUE(capped.go);
  EXPECT_EQ(kThreads, capped.num_threads);
}

} // namespace
