/**
 * @file generator.hh
 * @brief C++ wrapper for the NovaNN generator handle, providing a
 *        reproducible random stream over the C @ref Generator_ API.
 *
 * @details
 * Defines the @ref ncore::wrappers::GeneratorCXX class that wraps the
 * low-level C @ref Generator_ handle with value semantics, deterministic
 * fills, and state snapshots.
 *
 * A handle owns no resources: the C representation is plain state that
 * the caller keeps on the stack, so the wrapper needs no destructor and
 * copying a handle forks the stream instead of sharing it.
 *
 * @see generator.h  C generator definition.
 * @see device.h     Device enumeration (@ref Device_).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/rand/generator.h>
#include <ncore/tensor.h>

/**
 * @namespace ncore::wrappers
 * @brief High-level C++ wrappers over the ncore C core.
 */
namespace ncore::wrappers {

/**
 * @class GeneratorCXX
 * @brief RAII-style wrapper around the C @ref Generator_, exposing a safe
 *        C++ interface.
 *
 * A copy forks the stream: both handles keep their own state and advance
 * on their own. A failed initialization is reported through the
 * constructor status and through every later call, so the handle never
 * silently produces values it could not seed. A moved-from handle reports
 * @c novaInvalidHandle on every later call, because its engine state left
 * with the destination.
 *
 * A device fill is enqueued on the device engine stream and drained
 * before the call returns, so the caller observes the written values
 * without a separate synchronization step.
 *
 * A single instance must not be mutated concurrently from multiple
 * threads without external synchronization.
 */
class GeneratorCXX {

public:
  /**
   * @brief Initializes the stream.
   *
   * Xoshiro reaches a subsequence by walking one jump from a fresh state
   * per unit, so initialization costs O(subsequence) time; worker indices
   * should stay small, or prefer Philox4x32 whose subsequence lands
   * directly in the key.
   *
   * @param[in]  engine      Engine backing the handle.
   * @param[in]  device      Placement the handle fills.
   * @param[in]  seed        Seed value. Zero is valid.
   * @param[in]  subsequence Stream index. Worker @c w uses @c w.
   * @param[out] st          Optional status output; @ref novaSuccess on
   *                         success. A failure is also reported by every
   *                         later call on this handle.
   */
  explicit GeneratorCXX(RngEngine engine, Device_ device, uint64_t seed = 0,
                        uint64_t subsequence = 0, novaStatus_t *st = nullptr);

  /**
   * @brief Copy constructor. Forks the stream at the current position.
   */
  GeneratorCXX(const GeneratorCXX &other);

  /**
   * @brief Copy assignment. Forks the stream at the current position.
   */
  GeneratorCXX &operator=(const GeneratorCXX &other);

  /**
   * @brief Move constructor.
   */
  GeneratorCXX(GeneratorCXX &&other) noexcept;

  /**
   * @brief Move assignment.
   */
  GeneratorCXX &operator=(GeneratorCXX &&other) noexcept;

  /**
   * @brief Fills @p n 32-bit lanes.
   *
   * @param[out] dst Destination buffer. Must not be nullptr when @p n is
   *                 nonzero.
   * @param[in]  n   Values to emit.
   *
   * @return Status with @c err set to @c novaSuccess on success.
   */
  [[nodiscard]] novaStatus_t fillU32(uint32_t *dst, size_t n);

  /**
   * @brief Fills @p n 64-bit lanes.
   *
   * @param[out] dst Destination buffer. Must not be nullptr when @p n is
   *                 nonzero.
   * @param[in]  n   Values to emit.
   *
   * @return Status with @c err set to @c novaSuccess on success.
   */
  [[nodiscard]] novaStatus_t fillU64(uint64_t *dst, size_t n);

  /**
   * @brief Fills @p bytes bytes.
   *
   * Whole lanes go through the 32-bit fill and a host tail peels the low
   * bytes of one more lane on little-endian hosts. A device handle expects
   * device memory and a lane multiple byte count.
   *
   * @param[out] dst   Destination buffer.
   * @param[in]  bytes Bytes to write.
   *
   * @return Status with @c err set to @c novaSuccess on success, or
   *         @c novaInvalidValue for a device fill whose byte count is not
   *         a multiple of four.
   */
  [[nodiscard]] novaStatus_t fillBytes(void *dst, size_t bytes);

  /**
   * @brief Fills the whole payload of an allocated tensor.
   *
   * @param[in,out] ten Tensor to fill. Its placement must match the
   *                   handle placement.
   *
   * @return Status with @c err set to @c novaSuccess on success, or
   *         @c novaInvalidDevice when the placements differ.
   */
  [[nodiscard]] novaStatus_t fill(Tensor &ten);

  /**
   * @brief Entries a state snapshot needs (5 for Xoshiro, 7 for Philox).
   */
  [[nodiscard]] size_t snapshotWords() const;

  /**
   * @brief Writes the engine state into @p dst.
   *
   * Reproduction works only from a block boundary; parked lanes do not
   * travel with the snapshot.
   *
   * @param[out] dst Buffer holding at least @ref snapshotWords() entries.
   *
   * @return Status with @c err set to @c novaSuccess on success.
   */
  [[nodiscard]] novaStatus_t snapshot(uint64_t *dst) const;

  /**
   * @brief Restores the engine state from @p src.
   *
   * Parked lanes restart empty.
   *
   * @param[in] src Serialized state to restore.
   *
   * @return Status with @c err set to @c novaSuccess on success.
   */
  [[nodiscard]] novaStatus_t restore(const uint64_t *src);

  /**
   * @brief Returns a fork at the current stream position.
   */
  [[nodiscard]] GeneratorCXX clone() const;

  /**
   * @brief Placement this handle fills.
   */
  [[nodiscard]] Device_ device() const { return handle_.device; }

  /**
   * @brief Engine backing this handle.
   */
  [[nodiscard]] RngEngine engine() const { return handle_.engine; }

  /**
   * @brief Values emitted so far.
   */
  [[nodiscard]] uint64_t offset() const { return handle_.offset; }

private:
  /**
   * @brief Default constructor. Builds an unseeded handle that reports
   *        the initialization failure of the handle it is assigned from.
   */
  GeneratorCXX() = default;

  /**
   * @brief Waits for the queued device fills.
   *
   * A one byte readback copy shares the engine stream queue with the
   * fills, so it orders them out before the caller continues.
   */
  [[nodiscard]] static novaStatus_t drain(const void *device);

  Generator_ handle_{};
  novaStatus_t init_{};
};

// ================================================================
// Inline definitions
// ================================================================

inline GeneratorCXX::GeneratorCXX(RngEngine engine, Device_ device,
                                  uint64_t seed, uint64_t subsequence,
                                  novaStatus_t *st) {
  init_ = generator_init(&handle_, engine, device, seed, subsequence);
  if (st != nullptr) {
    *st = init_;
  }
}

inline GeneratorCXX::GeneratorCXX(const GeneratorCXX &other)
    : init_(other.init_) {
  if (init_.err == novaSuccess) {
    init_ = generator_clone(&other.handle_, &handle_);
  }
}

inline GeneratorCXX &GeneratorCXX::operator=(const GeneratorCXX &other) {
  if (this != &other) {
    handle_ = Generator_{};
    init_ = other.init_;
    if (init_.err == novaSuccess) {
      init_ = generator_clone(&other.handle_, &handle_);
    }
  }
  return *this;
}

inline GeneratorCXX::GeneratorCXX(GeneratorCXX &&other) noexcept
    : handle_(other.handle_), init_(other.init_) {
  other = GeneratorCXX();
}

inline GeneratorCXX &GeneratorCXX::operator=(GeneratorCXX &&other) noexcept {
  if (this != &other) {
    handle_ = other.handle_;
    init_ = other.init_;
    other = GeneratorCXX();
  }
  return *this;
}

inline novaStatus_t GeneratorCXX::fillU32(uint32_t *dst, size_t n) {
  if (init_.err != novaSuccess) {
    return init_;
  }
  const novaStatus_t filled = generator_fill_u32(&handle_, dst, n);
  if (filled.err == novaSuccess && n > 0 && handle_.device == DEVICE_GPU) {
    return drain(dst);
  }
  return filled;
}

inline novaStatus_t GeneratorCXX::fillU64(uint64_t *dst, size_t n) {
  if (init_.err != novaSuccess) {
    return init_;
  }
  const novaStatus_t filled = generator_fill_u64(&handle_, dst, n);
  if (filled.err == novaSuccess && n > 0 && handle_.device == DEVICE_GPU) {
    return drain(dst);
  }
  return filled;
}

inline novaStatus_t GeneratorCXX::fillBytes(void *dst, size_t bytes) {
  if (handle_.device == DEVICE_GPU && bytes % 4 != 0) {
    return novaStatus_t{.err = novaInvalidValue,
                        .message =
                            "Device fills go by lanes of four bytes, the tail "
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

inline novaStatus_t GeneratorCXX::fill(Tensor &ten) {
  if (init_.err != novaSuccess) {
    return init_;
  }
  if (ten.device != handle_.device) {
    return novaStatus_t{.err = novaInvalidDevice,
                        .message = "The tensor placement must match the "
                                   "handle placement.\n"};
  }
  return fillBytes(ten.data.data + ten.offset,
                   ten.storage->handle.requested_bytes);
}

inline size_t GeneratorCXX::snapshotWords() const {
  return handle_.engine == Xoshiro256 ? 5U : 7U;
}

inline novaStatus_t GeneratorCXX::snapshot(uint64_t *dst) const {
  if (init_.err != novaSuccess) {
    return init_;
  }
  size_t words = snapshotWords();
  return generator_get_state(&handle_, dst, &words);
}

inline novaStatus_t GeneratorCXX::restore(const uint64_t *src) {
  if (init_.err != novaSuccess) {
    return init_;
  }
  return generator_set_state(&handle_, src, snapshotWords());
}

inline GeneratorCXX GeneratorCXX::clone() const {
  GeneratorCXX forked;
  if (init_.err != novaSuccess) {
    forked.init_ = init_;
    return forked;
  }
  forked.init_ = generator_clone(&handle_, &forked.handle_);
  return forked;
}

inline novaStatus_t GeneratorCXX::drain(const void *device) {
  uint8_t probe = 0;
  return transfer_to(DEVICE_GPU, DEVICE_CPU, device, &probe, 1);
}

} // namespace ncore::wrappers
