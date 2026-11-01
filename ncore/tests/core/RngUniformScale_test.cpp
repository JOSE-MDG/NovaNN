/**
 * @file RngUniformScale_test.cpp
 * @brief Canonical mapping of raw 32-bit lanes into (0, 1) uniforms.
 *
 * The Box-Muller consumer divides by a power of two through the
 * float mantissa: 24 raw bits, odd-forced so the result can never
 * reach 0.0f or 1.0f. The previous prototype mapping added 1.0f to
 * the overly rounded full 32-bit float first, and 256 inputs at the
 * top of the range (0xFFFFFF80 through 0xFFFFFFFF) collapsed onto
 * exactly 1.0f by double rounding, feeding logf a degraded boundary
 * sample. This pins the canonical mapper and documents the corner
 * that motivated replacing it.
 */

#include <gtest/gtest.h>

#include <cstdint>

namespace {

constexpr float kInv24 = 0x1p-24f;

/// Canonical uniform: 24 mantissa bits, odd-forced, strictly inside (0, 1).
float uniformUnit(uint32_t raw) {
  return static_cast<float>((raw >> 8) | 1U) * kInv24;
}

/// The retired prototype mapper, kept only to document its corner.
float uniformPlusOne(uint32_t raw) {
  return (static_cast<float>(raw) + 1.0f) / 4294967296.0f;
}

} // namespace

/// Every raw lane maps strictly inside (0, 1), at mantissa resolution.
TEST(RngUniformScale, StaysInOpenUnitInterval) {
  const uint32_t corners[] = {
      0U, 1U, 0x7FFFFFFFU, 0x80000000U, 0xFFFFFF80U, 0xFFFFFFFEU, 0xFFFFFFFFU,
  };
  for (const uint32_t raw : corners) {
    const float u = uniformUnit(raw);
    EXPECT_GT(u, 0.0f) << raw;
    EXPECT_LT(u, 1.0f) << raw;
    EXPECT_GE(u, kInv24) << raw;
    EXPECT_LE(u, 1.0f - kInv24) << raw;
  }
  // Sweep a deterministic stride through the whole range.
  for (uint64_t raw = 0; raw <= 0xFFFFFFFFULL; raw += 4099U) {
    const float u = uniformUnit(static_cast<uint32_t>(raw));
    EXPECT_GT(u, 0.0f);
    EXPECT_LT(u, 1.0f);
  }
}

/// Odd forcing: the scaled value is always an odd multiple of 2^-24.
TEST(RngUniformScale, AlwaysOddScaled) {
  for (uint64_t m = 0; m <= 0xFFFFFFULL; m += 1021U) {
    const uint32_t raw = static_cast<uint32_t>(m) << 8;
    // (2k+1) with k < 2^23 fits the 24-bit mantissa, so the scaled
    // odd integer survives the float round trip exactly.
    const auto scaled = static_cast<long>(uniformUnit(raw) / kInv24);
    EXPECT_EQ(scaled % 2, 1) << m;
    EXPECT_EQ(scaled, (static_cast<long>(m) | 1L)) << m;
  }
}

/**
 * @brief Documents why the plus-one mapper was retired.
 *
 * @test raw >= 0xFFFFFF80 collapses onto exactly 1.0f (double
 * rounding through (float)raw first, then +1.0f), giving logf a
 * boundary sample whose contribution to r is a zero draw dressed
 * as a tail-rate sample.
 */
TEST(RngUniformScale, RetiredPlusOneFormCollapsesOnOne) {
  EXPECT_EQ(uniformPlusOne(0xFFFFFFFFU), 1.0f);
  EXPECT_EQ(uniformPlusOne(0xFFFFFF80U), 1.0f);
  EXPECT_LT(uniformPlusOne(0xFFFFFF7FU), 1.0f);
}
