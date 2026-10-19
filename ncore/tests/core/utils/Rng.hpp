/**
 * @file Rng.hpp
 * @brief Reproducible fixture fills over the generator handle API.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/rand/generator.h>
#include <ncore/tensor.h>

namespace tests::rng {

/// Default fill seed ("NOVA" in ASCII), shared by the whole test tree.
constexpr uint64_t kDefaultSeed = UINT64_C(0x4E4F5641);

/**
 * @class Generator
 * @brief One reproducible fixture stream.
 *
 * A failed init is reported by the next call, not by the constructor.
 * Copies fork the stream and each copy advances on its own. If a
 * device drain fails the fill already landed: regard the handle as
 * spent.
 */
class Generator {
public:
  /**
   * @brief Initializes the stream.
   *
   * Xoshiro runs host side only.
   */
  explicit Generator(Device_ device, uint64_t seed = kDefaultSeed,
                     uint64_t subsequence = 0, RngEngine engine = Philox4x32)
      : init_{generator_init(&handle_, engine, device, seed, subsequence)} {}

  /**
   * @brief Fills @p n 32-bit lanes.
   *
   * A device placement drains the null path before returning.
   */
  [[nodiscard]] novaStatus_t fillU32(uint32_t *dst, size_t n) {
    if (init_.err != novaSuccess) {
      return init_;
    }
    const novaStatus_t filled = generator_fill_u32(&handle_, dst, n);
    if (filled.err == novaSuccess && n > 0 && handle_.device == DEVICE_GPU) {
      return drain(dst);
    }
    return filled;
  }

  /**
   * @brief Fills @p n 64-bit lanes.
   */
  [[nodiscard]] novaStatus_t fillU64(uint64_t *dst, size_t n) {
    if (init_.err != novaSuccess) {
      return init_;
    }
    const novaStatus_t filled = generator_fill_u64(&handle_, dst, n);
    if (filled.err == novaSuccess && n > 0 && handle_.device == DEVICE_GPU) {
      return drain(dst);
    }
    return filled;
  }

  /**
   * @brief Fills @p bytes bytes.
   *
   * Whole lanes go through the 32-bit fill and a host tail peels the
   * low bytes of one more lane on little-endian hosts. Device handles
   * expect device memory (fill() is the checked path) and lane
   * multiple byte counts.
   */
  [[nodiscard]] novaStatus_t fillBytes(void *dst, size_t bytes) {
    if (handle_.device == DEVICE_GPU && bytes % 4 != 0) {
      return novaStatus_t{
          .err = novaInvalidValue,
          .message = "Device fills go by lanes of four bytes, the tail "
                     "must be zero.\n"};
    }
    const novaStatus_t head = fillU32(static_cast<uint32_t *>(dst), bytes / 4);
    if (head.err != novaSuccess) {
      return head;
    }
    const size_t tail = bytes % 4;
    if (tail == 0) {
      return head;
    }
    uint32_t lane = 0;
    const novaStatus_t consumed = fillU32(&lane, 1);
    if (consumed.err != novaSuccess) {
      return consumed;
    }
    auto *at = static_cast<uint8_t *>(dst) + ((bytes / 4) * 4);
    std::memcpy(at, &lane, tail);
    return head;
  }

  /**
   * @brief Fills the whole payload of an allocated tensor.
   *
   * The tensor must sit on the handle's placement.
   */
  [[nodiscard]] novaStatus_t fill(Tensor &ten) {
    if (init_.err != novaSuccess) {
      return init_;
    }
    if (ten.device != handle_.device) {
      return novaStatus_t{.err = novaInvalidDevice,
                          .message = "The tensor placement must match the "
                                     "handle placement.\n"};
    }
    return fillBytes(ten.data.data, ten.size * ten.item_size);
  }

  /**
   * @brief Entries a state snapshot needs (5 for Xoshiro, 7 for Philox).
   */
  [[nodiscard]] size_t snapshotWords() const {
    return handle_.engine == Xoshiro256 ? 5U : 7U;
  }

  /**
   * @brief Writes the engine state into @p dst.
   *
   * Reproduction works only from a block boundary, parked lanes do not
   * travel with the snapshot.
   */
  [[nodiscard]] novaStatus_t snapshot(uint64_t *dst) {
    if (init_.err != novaSuccess) {
      return init_;
    }
    size_t n = snapshotWords();
    return generator_get_state(&handle_, dst, &n);
  }

  /**
   * @brief Restores the engine state from @p src.
   *
   * Parked lanes restart empty.
   */
  [[nodiscard]] novaStatus_t restore(const uint64_t *src) {
    if (init_.err != novaSuccess) {
      return init_;
    }
    return generator_set_state(&handle_, src, snapshotWords());
  }

  /**
   * @brief Returns a fork at the current stream position.
   */
  [[nodiscard]] Generator clone() const {
    Generator forked;
    if (init_.err != novaSuccess) {
      forked.init_ = init_;
      return forked;
    }
    forked.init_ = generator_clone(&handle_, &forked.handle_);
    return forked;
  }

private:
  Generator() = default;

  /**
   * @brief Waits for the queued device fills.
   *
   * The null path drops a freshly compiled kernel when the next
   * null-stream op waits host side behind it, so a one byte readback
   * copy shares that queue and orders it out before the caller
   * continues.
   */
  [[nodiscard]] static novaStatus_t drain(const void *device) {
    uint8_t probe = 0;
    return transfer_to(DEVICE_GPU, DEVICE_CPU, device, &probe, 1);
  }

  Generator_ handle_{};
  novaStatus_t init_{};
};

/**
 * @brief One-shot fill of a whole tensor payload from a fresh stream.
 */
[[nodiscard]] inline novaStatus_t fillTensor(Tensor &ten,
                                             uint64_t seed = kDefaultSeed,
                                             uint64_t subsequence = 0) {
  Generator fill(ten.device, seed, subsequence);
  return fill.fill(ten);
}

/**
 * @brief One-shot byte fill of a host buffer from a fresh stream.
 *
 * Byte-precise, whole lanes and a peeled tail like the class method.
 */
[[nodiscard]] inline novaStatus_t fillBytes(void *dst, size_t bytes,
                                            uint64_t seed = kDefaultSeed,
                                            uint64_t subsequence = 0) {
  Generator fill(DEVICE_CPU, seed, subsequence);
  return fill.fillBytes(dst, bytes);
}

} // namespace tests::rng
