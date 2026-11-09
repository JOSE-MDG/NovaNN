/**
 * @file magic_division.hh
 * @brief Exact unsigned division by per-launch constants via multiply-shift.
 *
 * @details
 * The N-D gather kernels rebuild coordinates from a linear index, which
 * pays one integer division and one modulo per dimension per element.
 * On both device toolchains (nvcc and HIP-Clang) a 64-bit division
 * expands to a long instruction sequence, so the dominant per-element
 * cost is arithmetic, not memory traffic.
 *
 * This header precomputes, once per launch on the host, a reciprocal
 * pair for every shape extent and replaces each runtime division with
 * a fixed multiply-add-shift network. The device exactness statement
 * is
 *
 * @f[ \left\lfloor \frac{x m}{2^{s}} \right\rfloor =
 *     \left\lfloor \frac{x}{d} \right\rfloor \quad \forall x \le X
 * @f]
 *
 * with @f$X = 2^{32} - 1@f$. Writing @f$x = qd + r@f$ and
 * @f$m d = 2^{s} + \delta@f$ with @f$\delta \geq 0@f$ (the scan only
 * ever stores @f$m = \lceil 2^{s}/d \rceil@f$), the quotient estimate
 * becomes @f$q + \lfloor (q\delta + r m) / 2^{s} \rfloor@f$, so the
 * pair (m, s) is exact for every dividend when, and only when,
 *
 * @f[ q\delta + r m < 2^{s} \quad \text{for every reachable pair}
 *     \ (q, r),\ qd + r \le X @f$
 *
 * The error functional @f$f(x) = q(x)\delta + r(x)m@f$ rises by @f$m@f$
 * per step inside one quotient bucket and drops by
 * @f$(d-1)m - \delta@f$ at a bucket edge, where @f$(d-1)m \geq d-1 @f$
 * and @f$\delta \leq d-1@f$ (from @f$m = \lceil 2^{s}/d \rceil@f$,
 * @f$m d < 2^{s} + d@f$), so the maximum over the valid range is
 * attained at one of only two points:
 *
 * - @f$x = X@f$, the highest dividend (last, partial bucket).
 * - @f$x = \lfloor (X+1)/d \rfloor \cdot d - 1@f$, the highest
 *   quotient bucket fully traversed (remainder @f$d-1@f$).
 *
 * @ref make_magic_divisor() scans @f$s \in [31, 64]@f$ accepting the
 * first pair that passes both checks. The multiplier is capped at
 * @f$2^{33} - 1@f$: pairs with @f$m < 2^{32}@f$ compute
 * @f$\lfloor x m / 2^{s} \rfloor@f$ from one 64-bit multiply; pairs
 * with @f$m \geq 2^{32}@f$ (the @c wide class) recover the implicit
 * 33rd bit on the device with
 *
 * @code
 * t = (x * M) >> 32;                  // M = m - 2^32
 * q = (t + ((x - t) >> 1)) >> (s - 33)
 * @endcode
 *
 * which equals @f$\lfloor x m / 2^{s} \rfloor@f$ exactly, since
 * @f$t = \lfloor xM/2^{32} \rfloor@f$ satisfies
 * @f$x + t = \lfloor x m / 2^{32} \rfloor@f$
 * (@c floor of an integer sum) and nested floors compose through the
 * halving. Both forms need no 64-bit division and no multiply wider
 * than 64 bits.
 *
 * With the @c wide class the scan covers every divisor below
 * @f$2^{32}@f$, so no launch ever needs the plain-division N-D
 * kernels for supported extents (validation corners are still the
 * caller's structural gates: rank and dividend fit). Coverage and
 * exactness are additionally verified empirically against native
 * division by exhaustive and randomized sweeps in
 * @c MagicDivision_test.cpp.
 *
 * The validation arithmetic stays in unsigned 64 bits per term
 * (@f$q\delta < X@f$ because @f$q \leq X/d@f$;
 * @f$(d-1)m < md = 2^{s} + \delta \leq 2^{64}@f$), and the
 * @f$s = 64@f$ probes resolve their scale through the overflow route
 * described at @ref make_magic_divisor(). The multiply on the device
 * side is exact in the widths shown above.
 */

#pragma once

#include <cstdint>

#include <ncore/headeronly/macros.h>

namespace ncore::magic {

/// Rank cap for the packed reciprocal table passed as a kernel
/// parameter. Deeper collapsed views keep the plain division kernels.
inline constexpr uint32_t MAGIC_MAX_NDIMS = 16;

/**
 * @struct DivMagic
 * @brief One reciprocal pair with its device form.
 *
 * @details
 * @c mul stores the low 32 bits of the multiplier for both classes:
 * the full multiplier itself when @c wide is zero, and
 * @f$M = m - 2^{32}@f$ when @c wide is one (the 33rd bit lives in the
 * flag). @c shift is the total scale exponent @f$s@f$ for the plain
 * class and @f$s - 33@f$ for the wide class, matching the device
 * forms shown in the file documentation.
 */
struct DivMagic {
  uint32_t mul;   ///< Low 32 bits of the multiplier.
  uint32_t shift; ///< Scale exponent or post-network shift, see above.
  uint8_t wide;   ///< One when the multiplier needs its 33rd bit.
  uint8_t valid;  ///< Zero when no pair was found; route to fallback.
};

/** @brief Sentinel used by the fallback launchers, never a real pair. */
inline constexpr DivMagic kInvalidMagic{
    .mul = 0U,
    .shift = 0U,
    .wide = 0U,
    .valid = 0U,
};

/**
 * @brief Precompute the multiply-shift pair for one divisor.
 *
 * @details
 * Host-side only. Scans the scale exponent and accepts the first pair
 * passing the two worst-case checks. Powers of two resolve on the
 * first probe with @f$\delta = 0@f$, collapsing to bare shifts.
 *
 * The check runs on unsigned 64-bit arithmetic with overflow resolved
 * as rejection: the @f$2^{64}@f$ scale cannot be stored, so a wrapped
 * error functional is by definition past the scale. @f$s = 64@f$
 * derives its @f$\delta@f$ from modular remainders of @f$2^{64}-1@f$
 * instead of forming
 * @f$m d@f$, whose product would itself wrap. Overflow builtins are a
 * GNU/Clang extension; other providers answer the sentinel and every
 * launch routes to the plain-division fallback kernels.
 *
 * @param[in]  d  Divisor; must satisfy @f$d < 2^{32}@f$.
 *
 * @return The reciprocal pair, or @ref kInvalidMagic when the scan
 *         finds none (callers route that launch to the fallback).
 */
inline DivMagic make_magic_divisor(const uint32_t d) noexcept {
  if (d == 0U) {
    return kInvalidMagic;
  }
  if (d == 1U) {
    // floor(x/1) = x: the identity pair short-circuits modulo zero.
    return DivMagic{
        .mul = 1U,
        .shift = 0U,
        .wide = 0U,
        .valid = 1U,
    };
  }
#if defined(__GNUC__) || defined(__clang__)
  const uint64_t dd = d;
  const uint64_t X = 0xFFFFFFFFULL; // Largest valid dividend.
  const uint64_t qA = X / dd;       // Quotient of the last.
  const uint64_t rA = X % dd;
  const uint64_t qB = ((X + 1U) / dd) - 1U; // Fully traversed bucket.
  // 2^64 - 1 backs the s = 64 arithmetic: ceil(2^64/dd) and its delta
  // both come from this value's decomposition.
  const uint64_t w = ~0ULL;
  for (uint32_t s = 31U; s <= 64U; ++s) {
    uint64_t m = 0U;
    uint64_t delta = 0U;
    if (s == 64U) {
      m = (w / dd) + 1U;
      const uint64_t rem64 = (w % dd) + 1U;
      delta = rem64 == dd ? 0U : dd - rem64;
    } else {
      const uint64_t p = 1ULL << s;
      m = (p + dd - 1U) / dd;
      delta = (m * dd) - p;
    }
    if (m > 0x1FFFFFFFFULL) {
      continue;
    }
    // q * delta stays under 2^32 (q <= X/dd, delta <= dd - 1); the
    // r * m term carries the only overflow risk and a wrap already
    // proves failure, since the functional would exceed the scale.
    const auto exact = [&](const uint64_t q, const uint64_t r) noexcept {
      uint64_t rm = 0U;
      if (__builtin_mul_overflow(r, m, &rm)) {
        return false;
      }
      uint64_t total = 0U;
      if (__builtin_add_overflow(rm, q * delta, &total)) {
        return false;
      }
      return s == 64U ? true : total < (1ULL << s);
    };
    if (!exact(qA, rA) || !exact(qB, dd - 1U)) {
      continue;
    }
    const bool isWide = m >= 0x100000000ULL;
    const uint32_t low = static_cast<uint32_t>(m);
    return DivMagic{
        .mul = low,
        .shift = isWide ? static_cast<uint32_t>(s) - 33U : s,
        .wide = static_cast<uint8_t>(isWide ? 1U : 0U),
        .valid = 1U,
    };
  }
#endif
  return kInvalidMagic;
}

/**
 * @brief Unsigned @c x / divisor with the precomputed pair.
 *
 * @details
 * Device-or-host reference form of the two network classes; the
 * packed hot path is @ref packed_div32(). Applies the pair from
 * @ref make_magic_divisor only when the caller proved the dividend
 * range (@f$x < 2^{32}@f$); invalid pairs must never reach it.
 *
 * @param[in]  m  Reciprocal pair with @ref DivMagic.valid set.
 * @param[in]  x  Dividend below @f$2^{32}@f$.
 *
 * @return @f$\lfloor x / d \rfloor@f$ for the divisor @p m was made
 *         from.
 */
NCORE_HOST_DEVICE inline uint32_t magic_div32_bit(const DivMagic m,
                                                  const uint32_t x) noexcept {
  if (m.wide != 0U) {
    const uint64_t t =
        (static_cast<uint64_t>(x) * static_cast<uint64_t>(m.mul)) >> 32U;
    return static_cast<uint32_t>((t + ((x - t) >> 1U)) >> m.shift);
  }
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(x) * static_cast<uint64_t>(m.mul)) >> m.shift);
}

/**
 * @brief Quotient and remainder for one dividend, fusing both halves.
 *
 * @param[in]  m       Reciprocal pair for the divisor.
 * @param[in]  d       Divisor @p m was made from.
 * @param[in]  x       Dividend below @f$2^{32}@f$.
 * @param[out] div_val Quotient @f$\lfloor x / d \rfloor@f$.
 * @param[out] mod_val Remainder @f$x \bmod d@f$.
 */
NCORE_HOST_DEVICE inline void
magic_div_mod32(const DivMagic m, const uint32_t d, const uint32_t x,
                uint32_t *const div_val, uint32_t *const mod_val) noexcept {
  *div_val = magic_div32_bit(m, x);
  *mod_val = x - (*div_val * d);
}

/**
 * @brief True when every entry in a packed reciprocal table resolved.
 *
 * @param[in]  table  Entries for the active dimensions.
 * @param[in]  ndims  Active entry count.
 *
 * @return True when the launch can take the multiply-shift kernels;
 *         false routes the call to the division fallbacks.
 */
inline bool magic_table_valid(const DivMagic (&table)[MAGIC_MAX_NDIMS],
                              const uint32_t ndims) noexcept {
  if (ndims == 0U || ndims > MAGIC_MAX_NDIMS) {
    return false;
  }
  for (uint32_t i = 0U; i < ndims; ++i) {
    if (table[i].valid == 0U) {
      return false;
    }
  }
  return true;
}

/**
 * @struct DivMagicSet
 * @brief Kernel-parameter form of one reciprocal table.
 *
 * @details
 * Both launchers (CUDA and HIP) pass this by value, sized for
 * @ref MAGIC_MAX_NDIMS entries so deeper collapsed views fall back to
 * the plain division kernels instead of paying a large parameter
 * block. The packed layout (separate multiplier, shift, and class
 * arrays) keeps the block small; device code indexes it exactly as
 * the collapsed dims are walked, innermost first.
 */
struct DivMagicSet {
  uint32_t mul[MAGIC_MAX_NDIMS];  ///< Low 32 multiplier bits per dim.
  uint8_t shift[MAGIC_MAX_NDIMS]; ///< Scale or post-network shift.
  uint8_t wide[MAGIC_MAX_NDIMS];  ///< 33rd-bit class flag per dim.
};

/**
 * @brief Pack a resolved table into the kernel parameter form.
 *
 * @param[out] dst  Kernel parameter block; zeroed on any failure.
 * @param[in]  src  Host table with one entry per collapsed dim.
 * @param[in]  ndims  Active entry count, nonzero and at most
 *                    @ref MAGIC_MAX_NDIMS.
 *
 * @return True when all entries are resolved and the rank fits; false
 *         leaves @p dst zeroed for the fallback launch path.
 */
inline bool pack_magic_set(DivMagicSet *const dst, const DivMagic *src,
                           const uint32_t ndims) noexcept {
  *dst = DivMagicSet{};
  if (ndims == 0U || ndims > MAGIC_MAX_NDIMS) {
    return false;
  }
  for (uint32_t i = 0U; i < ndims; ++i) {
    if (src[i].valid == 0U) {
      *dst = DivMagicSet{};
      return false;
    }
    dst->mul[i] = src[i].mul;
    dst->shift[i] = static_cast<uint8_t>(src[i].shift);
    dst->wide[i] = src[i].wide;
  }
  return true;
}

/**
 * @brief Unsigned quotient for one dividend with a packed entry.
 *
 * @details
 * Hot path for the N-D gather kernels: one class branch, then either
 * a single 64-bit multiply and shift (plain) or the 33-bit recovery
 * network (wide, see the file documentation). Applies a packed pair
 * from @ref pack_magic_set only when the caller proved the dividend
 * range (@f$x < 2^{32}@f$); invalid pairs must never reach it.
 *
 * @param[in]  mul   Packed low 32 multiplier bits from the set.
 * @param[in]  shift Packed shift from the set.
 * @param[in]  wide  Packed class flag from the set.
 * @param[in]  x     Dividend below @f$2^{32}@f$.
 *
 * @return @f$\lfloor x / d \rfloor@f$ for the packed divisor.
 */
NCORE_HOST_DEVICE inline uint32_t packed_div32(const uint32_t mul,
                                               const uint8_t shift,
                                               const uint8_t wide,
                                               const uint32_t x) noexcept {
  if (wide != 0U) {
    const uint64_t t =
        (static_cast<uint64_t>(x) * static_cast<uint64_t>(mul)) >> 32U;
    return static_cast<uint32_t>((t + ((static_cast<uint64_t>(x) - t) >> 1U)) >>
                                 static_cast<uint32_t>(shift));
  }
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(x) * static_cast<uint64_t>(mul)) >>
      static_cast<uint32_t>(shift));
}

} // namespace ncore::magic
