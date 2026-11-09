/**
 * @file xoshiro256.c
 * @brief Host boundaries for xoshiro256++ state handling.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <ncore/core/status.h>
#include <ncore/rand/splitmix64.h>
#include <ncore/rand/xoshiro256.h>

/**
 * @def XOSHIRO_CHARPOLY_WORDS
 * @brief Words holding the low coefficients of the transition polynomial.
 */
#define XOSHIRO_CHARPOLY_WORDS 4U

/**
 * @brief Low 256 coefficients of the xoshiro256++ characteristic polynomial.
 *
 * @details
 * Coefficient @c k is bit @c (k & 63) of word @c (k >> 6); the leading
 * x^256 term is implicit. Both the value and the square-and-multiply
 * jump below reproduce the reference arbitrary-jump code published
 * with the generator:
 * https://prng.di.unimi.it/xoshiro256plusplus.c and
 * https://prng.di.unimi.it/f2x.c
 */
static const uint64_t xoshiro_charpoly[XOSHIRO_CHARPOLY_WORDS] = {
    0x9D116F2BB0F0F001ULL,
    0x0280002BCEFD1A5EULL,
    0x04B4EDCF26259F85ULL,
    0x0003C03C3F3ECB19ULL,
};

/**
 * @brief Multiply a ring element by x and reduce it modulo the polynomial.
 *
 * @param[in,out] a  Element to shift, reduced in place.
 */
static void xoshiro_mul_x(uint64_t a[XOSHIRO_CHARPOLY_WORDS]) {
  uint64_t carry = 0;
  for (size_t i = 0; i < XOSHIRO_CHARPOLY_WORDS; ++i) {
    const uint64_t next_carry = a[i] >> 63;
    a[i] = (a[i] << 1) | carry;
    carry = next_carry;
  }
  /* 256 is a whole number of words, so the shifted-out bit is x^256. */
  if (carry != 0) {
    for (size_t i = 0; i < XOSHIRO_CHARPOLY_WORDS; ++i) {
      a[i] ^= xoshiro_charpoly[i];
    }
  }
}

/**
 * @brief Multiply two ring elements and reduce modulo the polynomial.
 *
 * @details
 * Horner over the bits of @p a from the top. @p a may alias @p b: both
 * are read before the result is written back.
 *
 * @param[in,out] a  Left factor, replaced by the product.
 * @param[in]     b  Right factor.
 */
static void xoshiro_mul_mod(uint64_t a[XOSHIRO_CHARPOLY_WORDS],
                            const uint64_t b[XOSHIRO_CHARPOLY_WORDS]) {
  uint64_t r[XOSHIRO_CHARPOLY_WORDS] = {};
  for (int k = 255; k >= 0; --k) {
    xoshiro_mul_x(r);
    if (((a[(size_t)k >> 6] >> ((size_t)k & 63U)) & 1U) != 0U) {
      for (size_t i = 0; i < XOSHIRO_CHARPOLY_WORDS; ++i) {
        r[i] ^= b[i];
      }
    }
  }
  memcpy(a, r, sizeof r);
}

novaStatus_t xoshiro256_jump_to(XoshiroState *state, uint64_t steps) {
  if (state == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Jumping needs a live XoshiroState. Got null.\n",
    };
  }
  /* Walking a handful of calls beats the ring exponentiation below, and
     a zero distance must not square the polynomial at all. */
  if (steps <= 1024U) {
    for (uint64_t i = 0; i < steps; ++i) {
      (void)xoshiro256_next(state);
    }
    return (novaStatus_t){
        .err = novaSuccess,
        .message = nova_get_error_msg(novaSuccess, nullptr),
    };
  }
  uint64_t poly[XOSHIRO_CHARPOLY_WORDS] = {1, 0, 0, 0};
  int top = 63;
  while (top > 0 && ((steps >> top) & 1U) == 0U) {
    --top;
  }
  for (int k = top; k >= 0; --k) {
    xoshiro_mul_mod(poly, poly);
    if (((steps >> k) & 1U) != 0U) {
      xoshiro_mul_x(poly);
    }
  }
  xoshiro256_jump_table(state, poly);
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}

novaStatus_t xoshiro256_init(XoshiroState *state, uint64_t seed) {
  if (state == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Seeding needs a live XoshiroState. Got null.\n",
    };
  }
  uint64_t expanded[4] = {};
  auto fill = splitmix64_fill(seed, expanded);
  if (fill.err != novaSuccess) {
    return fill;
  }
  state->s[0] = expanded[0];
  state->s[1] = expanded[1];
  state->s[2] = expanded[2];
  state->s[3] = expanded[3];
  if ((state->s[0] | state->s[1] | state->s[2] | state->s[3]) == 0) {
    state->s[0] = 1;
  }
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}

novaStatus_t xoshiro256_get_state(const XoshiroState *state,
                                  uint64_t snapshot[4]) {
  if (state == nullptr || snapshot == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Snapshotting needs a live state and room for 4 entries. "
                   "Got null.\n",
    };
  }
  snapshot[0] = state->s[0];
  snapshot[1] = state->s[1];
  snapshot[2] = state->s[2];
  snapshot[3] = state->s[3];
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}

novaStatus_t xoshiro256_set_state(XoshiroState *state,
                                  const uint64_t snapshot[4]) {
  if (state == nullptr || snapshot == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Restoring needs a live state plus a 4-entry snapshot. "
                   "Got null.\n",
    };
  }
  if ((snapshot[0] | snapshot[1] | snapshot[2] | snapshot[3]) == 0) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "All-zero snapshots never advance. Restore one taken by "
                   "xoshiro256_get_state.\n",
    };
  }
  state->s[0] = snapshot[0];
  state->s[1] = snapshot[1];
  state->s[2] = snapshot[2];
  state->s[3] = snapshot[3];
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}
