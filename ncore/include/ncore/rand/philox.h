/**
 * @file philox.h
 * @brief Philox4x32-10 block function for GPU-friendly generation.
 *
 * @details
 * Philox maps a 128-bit counter plus a 64-bit key to 128 bits of
 * output through 10 fixed rounds. It holds no evolving state: every
 * draw is a pure function of counter and key, so parallel workers
 * partition draws by counter without communicating.
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
 * @struct PhiloxState
 * @brief Counter plus key selecting one Philox stream position.
 */
typedef struct {
  uint32_t ctr[4]; ///< 128-bit counter, little end first.
  uint32_t key[2]; ///< 64-bit key selecting the stream.
} PhiloxState;

/**
 * @brief Multiply with split high and low halves.
 *
 * @details
 * Device passes use the native multiply-high intrinsic. Host passes
 * use one widening product. Both compute the exact halves, so host,
 * CUDA, and HIP agree bit-for-bit. The check covers the nvcc and the
 * HIP device-pass macros together, never the CUDA-only one alone.
 *
 * @param[in]  a           First operand.
 * @param[in]  b           Second operand.
 * @param[out] result_high Receives the high 32 bits. Must not be
 *                         nullptr.
 *
 * @return Low 32 bits of the product.
 */
NCORE_HOST_DEVICE static inline uint32_t
philox_mulhilo32(uint32_t a, uint32_t b, uint32_t *result_high) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  *result_high = __umulhi(a, b);
  return a * b;
#else
  const uint64_t product = STATIC_CAST(uint64_t, a) * STATIC_CAST(uint64_t, b);
  *result_high = STATIC_CAST(uint32_t, product >> 32);
  return STATIC_CAST(uint32_t, product);
#endif
}

/**
 * @brief Apply one Philox round to a counter.
 *
 * @param[in] ctr  Input counter entries.
 * @param[in] key  Round key entries.
 * @param[out] out Round output entries. Must not be nullptr.
 */
NCORE_HOST_DEVICE static inline void philox_single_round(const uint32_t ctr[4],
                                                         const uint32_t key[2],
                                                         uint32_t out[4]) {
  uint32_t hi0 = 0;
  uint32_t hi1 = 0;
  const uint32_t lo0 = philox_mulhilo32(0xD2511F53, ctr[0], &hi0);
  const uint32_t lo1 = philox_mulhilo32(0xCD9E8D57, ctr[2], &hi1);
  out[0] = hi1 ^ ctr[1] ^ key[0];
  out[1] = lo1;
  out[2] = hi0 ^ ctr[3] ^ key[1];
  out[3] = lo0;
}

/**
 * @brief Compute one Philox4x32-10 block.
 *
 * @details
 * Ten rounds with the frozen multipliers. The Weyl schedule advances
 * the key between rounds. Ten keeps margin over the seven-round
 * statistical minimum.
 *
 * @param[in] ctr  Counter entries selecting the block.
 * @param[in] key  Key entries selecting the stream.
 * @param[out] out Four output entries in lane order. Must not be
 *                 nullptr.
 */
NCORE_HOST_DEVICE static inline void
philox4x32_10(const uint32_t ctr[4], const uint32_t key[2], uint32_t out[4]) {
  uint32_t counter[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
  uint32_t round_key[2] = {key[0], key[1]};
  for (uint32_t round = 0; round < 9; ++round) {
    philox_single_round(counter, round_key, counter);
    round_key[0] += 0x9E3779B9;
    round_key[1] += 0xBB67AE85;
  }
  philox_single_round(counter, round_key, out);
}

/**
 * @brief Add a block index to a 128-bit Philox counter.
 *
 * @details
 * The subsequence lives in the key, so the whole counter tracks
 * position with carry across all four entries. Safe with @p out
 * aliasing @p base: every read completes before any write.
 *
 * @param[in] base   Base counter entries.
 * @param[in] block  Block index to add.
 * @param[out] out   Sum entries. Must not be nullptr.
 */
NCORE_HOST_DEVICE static inline void
philox_ctr_add(const uint32_t base[4], uint64_t block, uint32_t out[4]) {
  const uint32_t lo = STATIC_CAST(uint32_t, block);
  const uint32_t hi = STATIC_CAST(uint32_t, block >> 32);
  const uint64_t c0 = STATIC_CAST(uint64_t, base[0]) + lo;
  const uint64_t c1 = STATIC_CAST(uint64_t, base[1]) + hi + (c0 >> 32);
  const uint64_t c2 = STATIC_CAST(uint64_t, base[2]) + (c1 >> 32);
  const uint64_t c3 = STATIC_CAST(uint64_t, base[3]) + (c2 >> 32);
  out[0] = STATIC_CAST(uint32_t, c0);
  out[1] = STATIC_CAST(uint32_t, c1);
  out[2] = STATIC_CAST(uint32_t, c2);
  out[3] = STATIC_CAST(uint32_t, c3);
}

/**
 * @brief Initialize counter and key from seed and subsequence.
 *
 * @details
 * The key mixes the SplitMix64 expansion of @p seed with @p
 * subsequence, giving each subsequence a disjoint key stream. The
 * counter starts at zero.
 *
 * @param[out] state       Destination. Must not be nullptr.
 * @param[in]  seed        Seed value. Zero is valid.
 * @param[in]  subsequence Stream index selecting the key.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer when @p state is nullptr.
 */
novaStatus_t philox_init(PhiloxState *state, uint64_t seed,
                         uint64_t subsequence);

/**
 * @brief Copy counter and key out.
 *
 * @param[in]  state  Source state. Must not be nullptr.
 * @param[out] ctr    Receives the four counter entries. Must not be
 *                    nullptr.
 * @param[out] key    Receives the two key entries. Must not be
 *                    nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer on null arguments.
 */
novaStatus_t philox_get_state(const PhiloxState *state, uint32_t ctr[4],
                              uint32_t key[2]);

/**
 * @brief Restore counter and key.
 *
 * @param[out] state Destination state. Must not be nullptr.
 * @param[in]  ctr   Four counter entries. Must not be nullptr.
 * @param[in]  key   Two key entries. Must not be nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer on null arguments.
 */
novaStatus_t philox_set_state(PhiloxState *state, const uint32_t ctr[4],
                              const uint32_t key[2]);

#ifdef __cplusplus
}
#endif
