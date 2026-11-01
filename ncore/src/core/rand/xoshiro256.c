/**
 * @file xoshiro256.c
 * @brief Host boundaries for xoshiro256++ state handling.
 */

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/rand/splitmix64.h>
#include <ncore/rand/xoshiro256.h>

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
