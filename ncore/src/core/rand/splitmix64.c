/**
 * @file splitmix64.c
 * @brief Host boundary for SplitMix64 seeding.
 */

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/status.h>
#include <ncore/rand/splitmix64.h>

novaStatus_t splitmix64_fill(uint64_t seed, uint64_t out[4]) {
  if (out == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Expansion needs room for 4 entries but out is null.\n",
    };
  }
  uint64_t state = seed;
  for (size_t i = 0; i < 4; ++i) {
    out[i] = splitmix64_next(&state);
  }
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}
