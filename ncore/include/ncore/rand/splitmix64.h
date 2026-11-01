/**
 * @file splitmix64.h
 * @brief SplitMix64 seeder for RNG engine initialization.
 *
 * @details
 * Expands one 64-bit seed into engine state via splitmix64_next.
 */

#pragma once

#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Advance a SplitMix64 state and return the next value.
 *
 * @details
 * Published xor-shift-multiply stirring over its fixed increment.
 * Zero seed maps to a fixed nonzero stream, never to all-zero state.
 *
 * @param[in,out] state  Running state, updated in place. Must not be
 *                       nullptr.
 *
 * @return Next 64-bit value of the stream.
 *
 * @pre @p state is a valid pointer.
 */
NCORE_HOST_DEVICE static inline uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
  return z ^ (z >> 31);
}

/**
 * @brief Fill four outputs from a 64-bit seed.
 *
 * @param[in]  seed  Seed value.
 * @param[out] out   Destination for four entries. Must not be nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer when @p out is nullptr.
 */
novaStatus_t splitmix64_fill(uint64_t seed, uint64_t out[4]);

#ifdef __cplusplus
}
#endif
