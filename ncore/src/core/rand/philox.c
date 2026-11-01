/**
 * @file philox.c
 * @brief Host boundaries for Philox4x32-10 state handling.
 */

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/rand/philox.h>
#include <ncore/rand/splitmix64.h>

novaStatus_t philox_init(PhiloxState *state, uint64_t seed,
                         uint64_t subsequence) {
  if (state == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Seeding needs a live PhiloxState. Got null.\n",
    };
  }
  uint64_t expanded[4] = {};
  auto fill = splitmix64_fill(seed, expanded);
  if (fill.err != novaSuccess) {
    return fill;
  }
  state->key[0] = (uint32_t)expanded[0] ^ (uint32_t)subsequence;
  state->key[1] = (uint32_t)expanded[1] ^ (uint32_t)(subsequence >> 32);
  state->ctr[0] = 0;
  state->ctr[1] = 0;
  state->ctr[2] = 0;
  state->ctr[3] = 0;
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}

novaStatus_t philox_get_state(const PhiloxState *state, uint32_t ctr[4],
                              uint32_t key[2]) {
  if (state == nullptr || ctr == nullptr || key == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Snapshotting needs room for 4 counter plus 2 key entries. "
                   "Got null.\n",
    };
  }
  ctr[0] = state->ctr[0];
  ctr[1] = state->ctr[1];
  ctr[2] = state->ctr[2];
  ctr[3] = state->ctr[3];
  key[0] = state->key[0];
  key[1] = state->key[1];
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}

novaStatus_t philox_set_state(PhiloxState *state, const uint32_t ctr[4],
                              const uint32_t key[2]) {
  if (state == nullptr || ctr == nullptr || key == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Restoring needs 4 counter plus 2 key entries. Got null.\n",
    };
  }
  state->ctr[0] = ctr[0];
  state->ctr[1] = ctr[1];
  state->ctr[2] = ctr[2];
  state->ctr[3] = ctr[3];
  state->key[0] = key[0];
  state->key[1] = key[1];
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}
