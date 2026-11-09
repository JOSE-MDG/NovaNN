/**
 * @file RngXoshiroJump_test.cpp
 * @brief Arbitrary-distance jumps of the xoshiro256++ engine.
 *
 * xoshiro256_jump_to() must land the state exactly where that many
 * xoshiro256_next() calls would, on both sides of the walk fast path
 * (steps <= 1024) and deep in the modular exponentiation path.
 * Composition and determinism pin the polynomial arithmetic, and the
 * null guard pins the status contract.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include <ncore/core/status.h>
#include <ncore/rand/xoshiro256.h>

namespace {

constexpr uint64_t kSeed = 0x0E5577ULL;

/// Distances around the walk fast path boundary (1024) and beyond.
const uint64_t kJumpSteps[] = {
    1, 2, 3, 17, 255, 1023, 1024, 1025, 4095, 4096, 4097, 65536,
};

/** Requires jump_to(steps) to reproduce a sequential walk state. */
void expectJumpLandsWhereWalkingDoes(uint64_t seed, uint64_t steps) {
  XoshiroState walked = {};
  XoshiroState jumped = {};
  ASSERT_EQ(xoshiro256_init(&walked, seed).err, novaSuccess);
  ASSERT_EQ(xoshiro256_init(&jumped, seed).err, novaSuccess);

  for (uint64_t i = 0; i < steps; ++i) {
    (void)xoshiro256_next(&walked);
  }
  const novaStatus_t st = xoshiro256_jump_to(&jumped, steps);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");

  EXPECT_EQ(0, std::memcmp(walked.s, jumped.s, sizeof walked.s))
      << "jump_to(" << steps << ") drifted from the sequential walk";
}

/** A zero distance must not touch the state at all. */
TEST(RngXoshiroJump, JumpToZeroLeavesTheStateIntact) {
  XoshiroState state = {};
  XoshiroState reference = {};
  ASSERT_EQ(xoshiro256_init(&state, kSeed).err, novaSuccess);
  ASSERT_EQ(xoshiro256_init(&reference, kSeed).err, novaSuccess);

  const novaStatus_t st = xoshiro256_jump_to(&state, 0);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  EXPECT_EQ(0, std::memcmp(reference.s, state.s, sizeof reference.s));
}

/** jump_to must equal stepping on the walk path, across the 1024
    boundary, and on the polynomial path. */
TEST(RngXoshiroJump, JumpToMatchesSequentialCallsAroundTheFastPath) {
  for (uint64_t steps : kJumpSteps) {
    SCOPED_TRACE(::testing::Message() << "steps=" << steps);
    expectJumpLandsWhereWalkingDoes(kSeed, steps);
    expectJumpLandsWhereWalkingDoes(0, steps);
    expectJumpLandsWhereWalkingDoes(0xDEADBEEFCAFEBABEULL, steps);
  }
}

/** A six figure distance exercises several ring multiplications. */
TEST(RngXoshiroJump, DeepJumpMatchesSequentialCalls) {
  expectJumpLandsWhereWalkingDoes(kSeed, 1ULL << 20);
}

/** Two jumps must compose into one jump of the summed distance. */
TEST(RngXoshiroJump, ConsecutiveJumpsComposeIntoOneJump) {
  constexpr uint64_t kA = 0x0000000F00000001ULL;
  constexpr uint64_t kB = 0x0000001000000007ULL;

  XoshiroState composed = {};
  XoshiroState single = {};
  ASSERT_EQ(xoshiro256_init(&composed, kSeed).err, novaSuccess);
  ASSERT_EQ(xoshiro256_init(&single, kSeed).err, novaSuccess);

  ASSERT_EQ(xoshiro256_jump_to(&composed, kA).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&composed, kB).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&single, kA + kB).err, novaSuccess);

  EXPECT_EQ(0, std::memcmp(single.s, composed.s, sizeof single.s))
      << "jump(a) then jump(b) must land at jump(a + b)";
}

/** Jump order must not matter, and a walk path jump must compose with
    a polynomial path jump. */
TEST(RngXoshiroJump, JumpOrderDoesNotMatterAroundTheFastPath) {
  constexpr uint64_t kSmall = 3;
  constexpr uint64_t kLarge = 4096;

  XoshiroState smallFirst = {};
  XoshiroState largeFirst = {};
  XoshiroState single = {};
  ASSERT_EQ(xoshiro256_init(&smallFirst, kSeed).err, novaSuccess);
  ASSERT_EQ(xoshiro256_init(&largeFirst, kSeed).err, novaSuccess);
  ASSERT_EQ(xoshiro256_init(&single, kSeed).err, novaSuccess);

  ASSERT_EQ(xoshiro256_jump_to(&smallFirst, kSmall).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&smallFirst, kLarge).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&largeFirst, kLarge).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&largeFirst, kSmall).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&single, kSmall + kLarge).err, novaSuccess);

  EXPECT_EQ(0, std::memcmp(single.s, smallFirst.s, sizeof single.s));
  EXPECT_EQ(0, std::memcmp(single.s, largeFirst.s, sizeof single.s));
}

/** The same seed and distance must always land on the same state. */
TEST(RngXoshiroJump, SameSeedReproducesTheJumpedState) {
  XoshiroState first = {};
  XoshiroState second = {};
  XoshiroState other = {};
  ASSERT_EQ(xoshiro256_init(&first, kSeed).err, novaSuccess);
  ASSERT_EQ(xoshiro256_init(&second, kSeed).err, novaSuccess);
  ASSERT_EQ(xoshiro256_init(&other, kSeed + 1).err, novaSuccess);

  ASSERT_EQ(xoshiro256_jump_to(&first, 1ULL << 40).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&second, 1ULL << 40).err, novaSuccess);
  ASSERT_EQ(xoshiro256_jump_to(&other, 1ULL << 40).err, novaSuccess);

  EXPECT_EQ(0, std::memcmp(first.s, second.s, sizeof first.s))
      << "the same seed must reproduce the jumped state";
  EXPECT_NE(0, std::memcmp(first.s, other.s, sizeof first.s))
      << "a different seed must land elsewhere";
}

/** A null state must be rejected, never dereferenced. */
TEST(RngXoshiroJump, JumpToRejectsNullState) {
  const novaStatus_t st = xoshiro256_jump_to(nullptr, 1);
  EXPECT_EQ(st.err, novaInvalidPointer)
      << (st.message != nullptr ? st.message : "");
}

} // namespace
