/**
 * @file Fill.hpp
 * @brief One-shot deterministic fills for test data over the generator
 *        wrapper.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/generator.hh>
#include <ncore/tensor.h>

namespace tests::rng {

/// Default fill seed ("NOVA" in ASCII), shared by the whole test tree.
constexpr uint64_t kDefaultSeed = UINT64_C(0x4E4F5641);

/**
 * @brief One-shot fill of a whole tensor payload from a fresh stream.
 */
[[nodiscard]] inline novaStatus_t fillTensor(Tensor &ten,
                                             uint64_t seed = kDefaultSeed,
                                             uint64_t subsequence = 0) {
  ncore::wrappers::GeneratorCXX fill(Philox4x32, ten.device, seed, subsequence);
  return fill.fill(ten);
}

/**
 * @brief One-shot byte fill of a host buffer from a fresh stream.
 *
 * Byte-precise: whole lanes go through the 32-bit fill, then a peeled
 * tail takes the low bytes of one more lane on little-endian hosts.
 */
[[nodiscard]] inline novaStatus_t fillBytes(void *dst, size_t bytes,
                                            uint64_t seed = kDefaultSeed,
                                            uint64_t subsequence = 0) {
  ncore::wrappers::GeneratorCXX fill(Philox4x32, DEVICE_CPU, seed, subsequence);
  return fill.fillBytes(dst, bytes);
}

} // namespace tests::rng
