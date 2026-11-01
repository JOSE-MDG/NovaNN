/**
 * @file xoshiro256.h
 * @brief xoshiro256++ engine for CPU generation.
 *
 * @details
 * 256 bits of state, 64-bit output through an add-rotate-add
 * scrambler. Shift/rotate path, no multiplication. The four entries
 * exclude zero as a group. A single zero entry is fine.
 */

#pragma once

#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/rand/splitmix64.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct XoshiroState
 * @brief 256-bit xoshiro state as four 64-bit entries.
 */
typedef struct {
  ALIGN(32) uint64_t s[4]; ///< State entries. Never all zero as a group.
} XoshiroState;

/**
 * @brief Rotate a 64-bit value left.
 *
 * @param[in] x  Input value.
 * @param[in] k  Rotation distance in bits, in @c [1, 63).
 *
 * @return @p x rotated left by @p k.
 */
NCORE_HOST_DEVICE static inline uint64_t xoshiro_rotl_u64(uint64_t x,
                                                          unsigned int k) {
  return (x << k) | (x >> (64 - k));
}

/**
 * @brief Emit the next value and advance the state.
 *
 * @details
 * Output rotation 23, shift 17, state rotation 45: the published
 * xoshiro256++ update. Upper output bits carry the scrambler
 * strength. Bulk 32-bit consumers take them first.
 *
 * @param[in,out] state  Engine state, advanced in place. Must not be
 *                     nullptr and must not hold all-zero entries.
 *
 * @return Next 64-bit value.
 *
 * @pre @p state is valid with at least one nonzero entry.
 */
NCORE_HOST_DEVICE static inline uint64_t xoshiro256_next(XoshiroState *state) {
  const uint64_t result =
      xoshiro_rotl_u64(state->s[0] + state->s[3], 23) + state->s[0];
  const uint64_t t = state->s[1] << 17;
  state->s[2] ^= state->s[0];
  state->s[3] ^= state->s[1];
  state->s[1] ^= state->s[2];
  state->s[0] ^= state->s[3];
  state->s[2] ^= t;
  state->s[3] = xoshiro_rotl_u64(state->s[3], 45);
  return result;
}

/**
 * @brief Accumulate-and-step helper shared by jump functions.
 *
 * @param[in,out] state   Engine state, advanced in place.
 * @param[in]     table Four-entry jump polynomial.
 */
NCORE_HOST_DEVICE static inline void
xoshiro256_jump_table(XoshiroState *state, const uint64_t table[4]) {
  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (int i = 0; i < 4; i++) {
    for (int b = 0; b < 64; b++) {
      if (table[i] & (UINT64_C(1) << b)) {
        s0 ^= state->s[0];
        s1 ^= state->s[1];
        s2 ^= state->s[2];
        s3 ^= state->s[3];
      }
      (void)xoshiro256_next(state);
    }
  }
  state->s[0] = s0;
  state->s[1] = s1;
  state->s[2] = s2;
  state->s[3] = s3;
}

/**
 * @brief Jump ahead as 2 to the 128 next calls.
 *
 * @details
 * Partitions one seeded sequence into non-overlapping worker streams:
 * worker @c w applies this @c w times from the seeded state.
 *
 * @param[in,out] state  Engine state. Must not be nullptr.
 */
NCORE_HOST_DEVICE static inline void xoshiro256_jump(XoshiroState *state) {
  static const uint64_t table[4] = {
      0x180EC6D33CFD0ABA,
      0xD5A61266F0C9392C,
      0xA9582618E03FC9AA,
      0x39ABDC4529B1661C,
  };
  xoshiro256_jump_table(state, table);
}

/**
 * @brief Jump ahead as 2 to the 192 next calls.
 *
 * @details
 * Coarse partitioning for groups of workers. Each group start then
 * fans out with @ref xoshiro256_jump.
 *
 * @param[in,out] state  Engine state. Must not be nullptr.
 */
NCORE_HOST_DEVICE static inline void xoshiro256_long_jump(XoshiroState *state) {
  static const uint64_t table[4] = {
      0x76E15D3EFEFDCBBF,
      0xC5004E441C522FB3,
      0x77710069854EE241,
      0x39109BB02ACBE635,
  };
  xoshiro256_jump_table(state, table);
}

/**
 * @brief Initialize state from a 64-bit seed.
 *
 * @param[out] state   Destination state. Must not be nullptr.
 * @param[in]  seed  Seed value. Zero is valid.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer when @p state is nullptr.
 */
novaStatus_t xoshiro256_init(XoshiroState *state, uint64_t seed);

/**
 * @brief Copy the engine state out.
 *
 * @param[in]  state       Source state. Must not be nullptr.
 * @param[out] snapshot  Destination for the four state entries. Must not
 *                       be nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer on null arguments.
 */
novaStatus_t xoshiro256_get_state(const XoshiroState *state,
                                  uint64_t snapshot[4]);

/**
 * @brief Restore the engine state.
 *
 * @details
 * All-zero input is rejected so the engine can never resume from its
 * single invalid state. Prior state is left untouched on rejection.
 *
 * @param[out] state       Destination state. Must not be nullptr.
 * @param[in]  snapshot  Four state entries. Must not be nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidPointer on null arguments, or
 *         @c novaInvalidValue on all-zero input.
 */
novaStatus_t xoshiro256_set_state(XoshiroState *state,
                                  const uint64_t snapshot[4]);

#ifdef __cplusplus
}
#endif
