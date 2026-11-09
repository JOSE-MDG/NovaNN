/**
 * @file MagicDivision_test.cpp
 * @brief Certification for the multiply-shift division table.
 *
 * Validates ncore::magic helpers against exact 64-bit division for the
 * full gate the launchers apply: divisors under 2^32 and dividends
 * under 2^32. The scan is expected to resolve every such divisor, so
 * the fallback sentinel appears only for rank/divisor-zero inputs.
 * Exactness uses dividends around the two worst-case points the
 * header proof walks (the last dividend and the last fully traversed
 * quotient bucket), dense interior strides, randomized large
 * divisors, and the packed hot path the gather kernels execute.
 */

#include <array>
#include <cstdint>
#include <random>

#include <gtest/gtest.h>

#include <ncore/headeronly/magic_division.hh>

namespace nm = ncore::magic;

namespace {

/// Dividends that always include the exact-validation edges.
constexpr uint64_t kMaxDividend = 0xFFFFFFFFULL;

/// Largest divisor a packed set can craft from.
constexpr uint32_t kMaxDivisor = 0xFFFFFFFEU;

void checkExact(const uint32_t d, const uint64_t x, const char *what) {
  const nm::DivMagic m = nm::make_magic_divisor(d);
  ASSERT_EQ(m.valid, 1U) << "unresolved divisor " << d << " (" << what << ")";
  const uint64_t divVal = nm::magic_div32_bit(m, static_cast<uint32_t>(x));
  const uint32_t x32 = static_cast<uint32_t>(x);
  const uint32_t want = static_cast<uint32_t>(x / static_cast<uint64_t>(d));
  ASSERT_EQ(divVal, want) << "d=" << d << " x=" << x32 << " (" << what << ")";
}

/// Last dividend, the last fully traversed bucket top, and neighbors.
void checkWorstPoints(const uint32_t d, const char *what) {
  const uint64_t bucket = kMaxDividend + 1ULL;
  const uint64_t fullTop =
      ((bucket / static_cast<uint64_t>(d)) * static_cast<uint64_t>(d)) - 1ULL;
  const std::initializer_list<uint64_t> points = {
      0ULL,
      1ULL,
      fullTop + 1ULL,
      fullTop,
      fullTop ? fullTop - 1ULL : 0ULL,
      kMaxDividend};
  for (const uint64_t x : points) {
    checkExact(d, x, what);
  }
}

} // namespace

/**
 * @brief Every divisor up to 2^16 must resolve. Dense boundary checks cover
 *        the bucket-edge transitions the coordinate odometer walks, and the
 *        fused div+mod helper agrees with the native pair.
 */
TEST(MagicDivision, ExhaustiveSmallDivisors) {
  for (uint32_t d = 1U; d <= 65536U; ++d) {
    const nm::DivMagic m = nm::make_magic_divisor(d);
    ASSERT_EQ(m.valid, 1U) << "no pair for divisor " << d;
    if (m.wide != 0U) {
      ASSERT_LE(m.shift, 31U) << "wide shift range broken at d=" << d;
    } else if (d != 1U) {
      ASSERT_GE(m.shift, 31U) << "plain shift range broken at d=" << d;
      ASSERT_LE(m.shift, 63U) << "plain shift range broken at d=" << d;
    }

    const std::array<uint32_t, 8> seeds = {
        0U, 1U, 2U, d - 1U, d, d + 1U, 0x7FFFFFFFU, 0xFFFFFFFFU};
    for (const uint32_t x : seeds) {
      uint32_t divVal = 0U;
      uint32_t modVal = 0U;
      nm::magic_div_mod32(m, d, x, &divVal, &modVal);
      ASSERT_EQ(divVal, x / d) << "div mismatch d=" << d << " x=" << x;
      ASSERT_EQ(modVal, x % d) << "mod mismatch d=" << d << " x=" << x;
    }

    // Bucket-edge dividends: transitions where the quotient increments.
    // The start past the last bucket sits at 2^32 exactly for divisors
    // of 2^32, outside the pair's dividend range, so it only checks
    // when it lands inside.
    const uint64_t qTop = 4294967296ULL / d;
    if (qTop >= 2U) {
      checkExact(d, (qTop * d) - 1ULL, "full-bucket top");
      if (qTop * static_cast<uint64_t>(d) <= kMaxDividend) {
        checkExact(d, qTop * d, "next-bucket start");
      }
    }
  }
}

/**
 * @brief Interior sweep: every dividend in [0, 4 * d + 7] is validated for
 *        divisors up to 4096, catching carry behavior inside the smallest
 *        quotients (e.g. x = k * d + d - 1).
 */
TEST(MagicDivision, ExhaustiveSmallInterior) {
  for (uint32_t d = 2U; d <= 4096U; ++d) {
    const uint32_t span = (4U * d) + 7U;
    for (uint32_t x = 0U; x < span; ++x) {
      const nm::DivMagic m = nm::make_magic_divisor(d);
      ASSERT_EQ(nm::magic_div32_bit(m, x), x / d)
          << "interior mismatch d=" << d << " x=" << x;
    }
  }
}

/**
 * @brief Full large-divisor coverage: random divisors across [2, 2^32) all
 *        resolve (the coverage claim behind removing the plain-division
 *        coordinate path for supported extents), with exactness checked at
 *        the two worst-case residual points from the header proof plus
 *        randomized dividends.
 */
TEST(MagicDivision, UniversalLargeDivisorCoverage) {
  std::mt19937_64 rng(20260501U);
  std::uniform_int_distribution<uint32_t> divDist(2U, kMaxDivisor);
  std::uniform_int_distribution<uint32_t> xDist(0U, 0xFFFFFFFFU);
  for (int iter = 0; iter < 8000; ++iter) {
    const uint32_t d = divDist(rng);
    const nm::DivMagic m = nm::make_magic_divisor(d);
    ASSERT_EQ(m.valid, 1U) << "no pair for divisor " << d;
    checkWorstPoints(d, "random large");
    for (int j = 0; j < 8; ++j) {
      checkExact(d, xDist(rng), "random large");
    }
  }
}

/**
 * @brief Structured boundary divisors: powers of two and their neighbors.
 *        A power of two must resolve plain (bare shift), never take the wide
 *        network, and stay exact; adjacent divisors exercise the first
 *        multi-bit correction levels.
 */
TEST(MagicDivision, PowerOfTwoBoundaries) {
  for (uint32_t k = 1U; k <= 31U; ++k) {
    const uint32_t pot = 1U << k;
    const nm::DivMagic m = nm::make_magic_divisor(pot);
    ASSERT_EQ(m.valid, 1U) << "no pair for 2^" << k;
    ASSERT_EQ(m.wide, 0U) << "power of two must resolve plain, 2^" << k;
    ASSERT_EQ(m.mul, 1U << (31U - k)) << "bare shift multiplier, 2^" << k;
    ASSERT_EQ(m.shift, 31U) << "bare shift exponent, 2^" << k;

    for (const uint32_t x : {0U, 1U, 3U, 5U, 17U, 1023U, 1024U, 65535U,
                             0x7FFFFFFFU, 0xFFFFFFFFU}) {
      checkExact(pot, x, "power of two");
    }

    for (const uint32_t adj : {pot + 1U, pot - 1U}) {
      if (adj < 2U) {
        continue;
      }
      const nm::DivMagic j = nm::make_magic_divisor(adj);
      ASSERT_EQ(j.valid, 1U) << "no pair for 2^" << k << " +/- 1";
      checkWorstPoints(adj, "power of two neighbor");
      for (const uint32_t x : {0U, 1U, adj - 1U, adj, 0xFFFFFFFFU}) {
        checkExact(adj, x, "power of two neighbor");
      }
    }
  }
}

/**
 * @brief Encoding goldens pinned from the real scan and from the networks
 *        GNU compilers emit for these divisors: d = 3 and d = 5 are plain
 *        32-bit multipliers, d = 7 and d = 37 recover the 33rd bit through
 *        the wide class, and d = 2^31 - 1 uses the LEA-style M = 3 network.
 */
TEST(MagicDivision, ClassEncodingGoldens) {
  const nm::DivMagic three = nm::make_magic_divisor(3U);
  EXPECT_EQ(three.mul, 2863311531U);
  EXPECT_EQ(three.shift, 33U);
  EXPECT_EQ(three.wide, 0U);

  const nm::DivMagic five = nm::make_magic_divisor(5U);
  EXPECT_EQ(five.mul, 3435973837U);
  EXPECT_EQ(five.shift, 34U);
  EXPECT_EQ(five.wide, 0U);

  const nm::DivMagic seven = nm::make_magic_divisor(7U);
  EXPECT_EQ(seven.mul, 613566757U);
  EXPECT_EQ(seven.shift, 2U);
  EXPECT_EQ(seven.wide, 1U);

  const nm::DivMagic thirtySeven = nm::make_magic_divisor(37U);
  EXPECT_EQ(thirtySeven.mul, 3134165325U);
  EXPECT_EQ(thirtySeven.shift, 5U);
  EXPECT_EQ(thirtySeven.wide, 1U);

  const nm::DivMagic over31 = nm::make_magic_divisor(2147483647U);
  EXPECT_EQ(over31.mul, 3U);
  EXPECT_EQ(over31.shift, 30U);
  EXPECT_EQ(over31.wide, 1U);

  // Identity divisor: trivial pair, no network.
  const nm::DivMagic one = nm::make_magic_divisor(1U);
  EXPECT_EQ(one.mul, 1U);
  EXPECT_EQ(one.shift, 0U);
  EXPECT_EQ(one.wide, 0U);
  checkExact(1U, 0x7FFFFFFFU, "identity");
  checkExact(1U, 0xFFFFFFFFU, "identity");
}

/**
 * @brief Sentinel contract: zero is routed to the fallback and never reads
 *        like a resolved pair.
 */
TEST(MagicDivision, SentinelContract) {
  const nm::DivMagic zero = nm::make_magic_divisor(0U);
  EXPECT_EQ(zero.valid, 0U);
  EXPECT_EQ(zero.mul, 0U);
  EXPECT_TRUE(zero.wide == 0U);
}

/**
 * @brief Table-level gate: the launcher-facing validity check rejects an
 *        empty rank, a rank past the packed cap, and any invalid entry.
 */
TEST(MagicDivision, TableValidation) {
  nm::DivMagic table[nm::MAGIC_MAX_NDIMS] = {};
  for (uint32_t i = 0U; i < nm::MAGIC_MAX_NDIMS; ++i) {
    table[i] = nm::make_magic_divisor(i + 1U);
    ASSERT_EQ(table[i].valid, 1U);
  }

  EXPECT_TRUE(nm::magic_table_valid(table, 1U));
  EXPECT_TRUE(nm::magic_table_valid(table, 4U));
  EXPECT_TRUE(nm::magic_table_valid(table, nm::MAGIC_MAX_NDIMS));
  EXPECT_FALSE(nm::magic_table_valid(table, 0U));
  EXPECT_FALSE(nm::magic_table_valid(table, nm::MAGIC_MAX_NDIMS + 1U));

  table[2] = nm::kInvalidMagic;
  EXPECT_FALSE(nm::magic_table_valid(table, 4U));
}

/**
 * @brief Hot-path certification: pack a mixed wide/plain table and evaluate
 *        packed_div32 (the helper the gather kernels execute per element)
 *        against native division for randomized dividends, plus the packed
 *        failure contract for broken ranks.
 */
TEST(MagicDivision, PackedSetHotPath) {
  nm::DivMagic table[nm::MAGIC_MAX_NDIMS] = {};
  const uint32_t dims[] = {1U,     7U,         4096U,       65533U,
                           65537U, 3U,         37U,         1024U,
                           5U,     268435455U, 1073741823U, 2147483646U};
  constexpr uint32_t kRank = 12U;
  for (uint32_t i = 0U; i < kRank; ++i) {
    table[i] = nm::make_magic_divisor(dims[i]);
    ASSERT_EQ(table[i].valid, 1U);
  }
  ASSERT_EQ(kRank, 12U);

  nm::DivMagicSet set{};
  ASSERT_TRUE(nm::pack_magic_set(&set, table, kRank));
  for (uint32_t i = 0U; i < kRank; ++i) {
    EXPECT_EQ(set.mul[i], table[i].mul);
    EXPECT_EQ(set.shift[i], table[i].shift);
    EXPECT_EQ(set.wide[i], table[i].wide);
  }

  std::mt19937_64 rng(20261107U);
  std::uniform_int_distribution<uint32_t> xDist(1U, 0xFFFFFFFFU);
  for (uint32_t i = 0U; i < kRank; ++i) {
    for (int j = 0; j < 24; ++j) {
      const uint32_t x = xDist(rng);
      const uint32_t got =
          nm::packed_div32(set.mul[i], set.shift[i], set.wide[i], x);
      EXPECT_EQ(got, x / dims[i])
          << "packed miscompute dim=" << i << " x=" << x;
    }
  }

  // Fallback contract through the pack path.
  nm::DivMagic broken[nm::MAGIC_MAX_NDIMS] = {};
  broken[0] = nm::make_magic_divisor(3U);
  broken[0].valid = 0U;
  ASSERT_TRUE(nm::pack_magic_set(&set, broken, 4U) == false);
  EXPECT_EQ(set.mul[0], 0U);
  EXPECT_EQ(set.wide[3], 0U);
  EXPECT_TRUE(nm::pack_magic_set(&set, table, nm::MAGIC_MAX_NDIMS + 1U) ==
              false);
  EXPECT_EQ(set.shift[0], 0U);
}
