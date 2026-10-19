/**
 * @file generator.h
 * @brief Generator handle and bulk fill facade over engine state.
 *
 * @details
 * A @ref Generator_ names one reproducible stream.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/rand/philox.h>
#include <ncore/rand/xoshiro256.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum RngEngine
 * @brief Engine backing a handle.
 */
typedef enum : uint8_t {
  Philox4x32, ///< Counter blocks. Offset is exact random access.
  Xoshiro256, ///< Sequential recurrence. Offset counts emitted values.
} RngEngine;

/**
 * @def RAND_LANE_CACHE_CAP
 * @brief Lanes a fill may park in the handle cache between fills.
 */
#define RAND_LANE_CACHE_CAP 3U

/**
 * @struct Generator_
 * @brief Caller-owned handle naming one reproducible stream.
 */
typedef struct {
  uint64_t seed;        ///< Run seed selecting the key material.
  uint64_t subsequence; ///< Stream index within the seed.
  uint64_t
      offset; ///< Values emitted so far. The Philox counter tracks blocks separately.
  Device_ device;   ///< Placement the stream fills.
  RngEngine engine; ///< Engine backing this handle.
  union {
    PhiloxState philox;   ///< Live state when @c engine is Philox4x32.
    XoshiroState xoshiro; ///< Live state when @c engine is Xoshiro256.
  } state;
  uint32_t lane_cache[RAND_LANE_CACHE_CAP];
  uint32_t lane_count; ///< Parked lanes in @c lane_cache (0 to cap).
} Generator_;

/**
 * @brief Initialize a handle from seed and subsequence.
 *
 * @details
 * Xoshiro reaches a subsequence by walking one jump from a fresh
 * state per unit, so initialization costs O(subsequence) time; worker
 * indices should stay small, or prefer Philox4x32 whose subsequence
 * lands directly in the key. Impossible engine/device pairs are
 * rejected before the handle is touched. Initialization performs no
 * device probe.
 *
 * @param[out] gen         Destination handle. Must not be nullptr.
 * @param[in]  engine      Engine backing the handle.
 * @param[in]  device      Placement the handle fills.
 * @param[in]  seed        Seed value. Zero is valid.
 * @param[in]  subsequence Stream index. Worker @c w uses @c w.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidPointer when @p gen is nullptr,
 *         @c novaInvalidValue for an unknown engine or Xoshiro on
 *         device, or @c novaInvalidDevice for an unknown placement.
 */
novaStatus_t generator_init(Generator_ *gen, RngEngine engine, Device_ device,
                            uint64_t seed, uint64_t subsequence);

/**
 * @brief Fill with 64-bit values, advancing the handle.
 *
 * @details
 * Consecutive fills never repeat. Philox packs lane pairs little
 * end first. GPU placements fill device memory through the native
 * bridge. Offset advances only on success. Zero demand succeeds
 * without touching state or validating the engine.
 *
 * @param[in,out] gen  Handle, advanced in place. Must not be nullptr.
 * @param[out]    out  Caller-owned buffer. Must not be nullptr when
 *                     @p n is nonzero.
 * @param[in]     n    Values to emit.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidPointer on null arguments,
 *         @c novaInvalidValue for xoshiro on device, or the bridge
 *         status for GPU placements.
 *
 * @pre @p gen is exclusively owned by the caller for the call.
 */
novaStatus_t generator_fill_u64(Generator_ *gen, uint64_t *out, size_t n);

/**
 * @brief Fill with 32-bit values, advancing the handle.
 *
 * @details
 * The xoshiro path emits the upper half first, where the scrambler
 * is strongest. Philox emits blocks in lane order. GPU placements
 * fill device memory through the native bridge. Offset advances
 * only on success. Zero demand succeeds without touching state or
 * validating the engine.
 *
 * @param[in,out] gen  Handle, advanced in place. Must not be nullptr.
 * @param[out]    out  Caller-owned buffer. Must not be nullptr when
 *                     @p n is nonzero.
 * @param[in]     n    Values to emit.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidPointer on null arguments,
 *         @c novaInvalidValue for xoshiro on device, or the bridge
 *         status for GPU placements.
 *
 * @pre @p gen is exclusively owned by the caller for the call.
 */
novaStatus_t generator_fill_u32(Generator_ *gen, uint32_t *out, size_t n);

/**
 * @brief Copy the engine state and offset out of a handle.
 *
 * @details
 * The snapshot serializes live state plus the emitted offset: five
 * entries for Xoshiro256 (state then offset), seven for Philox4x32
 * (counter, key, then offset). @p n_snapshot travels both ways: pass
 * the buffer's capacity in; on success it holds the entries written,
 * and for an undersized buffer nothing is written, the call fails
 * with @c novaInvalidValue and it holds the required count. Parked
 * lanes of a mid-block fill do not travel: a restored handle resumes
 * on the next block boundary, so a snapshot taken mid-block cannot
 * reproduce its in-flight lanes.
 *
 * @param[in]     gen         Source handle. Must not be nullptr.
 * @param[out]    snapshot    Destination for the serialized state. Must
 *                            not be nullptr and hold @c *n_snapshot
 *                            entries.
 * @param[in,out] n_snapshot  Capacity in, entries written or required
 *                            out. Must not be nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidPointer on null arguments, or
 *         @c novaInvalidValue when the buffer is undersized.
 */
novaStatus_t generator_get_state(const Generator_ *gen, uint64_t *snapshot,
                                 size_t *n_snapshot);

/**
 * @brief Restore the engine state and offset of a handle.
 *
 * @details
 * Passing the snapshot back reproduces the stream only when the
 * snapshot was taken at a block boundary: five entries for
 * Xoshiro256 (state then offset), seven for Philox4x32 (counter,
 * key, then offset). Philox counter and key entries beyond the 32-bit
 * range are rejected instead of silently truncated. A restore lands
 * on a block boundary: mid-block parked lanes do not travel, so they
 * are dropped and the next fill continues from the restored counter.
 * Failure leaves prior state untouched.
 *
 * @param[in,out] gen         Destination handle. Must not be nullptr.
 * @param[in]     snapshot    Serialized state to restore. Must not be
 *                            nullptr.
 * @param[in]     n_snapshot  5 for Xoshiro256 or 7 for Philox4x32,
 *                            matching @p gen engine.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidPointer on null arguments, or
 *         @c novaInvalidValue on wrong size or out-of-range Philox
 *         entries (also for all-zero xoshiro state).
 */
novaStatus_t generator_set_state(Generator_ *gen, const uint64_t *snapshot,
                                 size_t n_snapshot);

/**
 * @brief Duplicate a handle including its live state.
 *
 * @param[in]  src  Source handle. Must not be nullptr.
 * @param[out] dst  Destination handle. Must not be nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer on null arguments.
 */
novaStatus_t generator_clone(const Generator_ *src, Generator_ *dst);

/**
 * @brief Copy the default handle for a device.
 *
 * @details
 * The copy is a frozen snapshot: filling it advances the copy, never
 * the underlying default. Two copies of the default therefore draw
 * identical values, and successive fills through @ref default_fill_u32()
 * or @ref default_fill_u64() are the way to consume the shared stream.
 *
 * @param[in]  device Placement selecting which default to copy.
 * @param[out] gen    Destination handle. Must not be nullptr.
 *
 * @return Status with @c err set to @c novaSuccess on success or
 *         @c novaInvalidPointer when @p gen is nullptr.
 */
novaStatus_t default_generator(Device_ device, Generator_ *gen);

/**
 * @brief Seed the CPU and GPU defaults together.
 *
 * @details
 * Reseeds both defaults. The GPU default is plain state, no device
 * needed. Per-device separation is the caller's job through
 * @c subsequence.
 *
 * @param[in] seed Seed value. Zero is valid.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
novaStatus_t manual_seed(uint64_t seed);

/**
 * @brief Fill @p out with 32-bit draws flowing through the process
 *        default.
 *
 * @details
 * Runs a full fill on the live default for @p device under its mutex,
 * so the shared state advances across calls and repeated calls hand
 * out fresh values. Prefer @c subsequence via @ref generator_init()
 * when a private stream is needed; this entry point is the shared
 * one.
 *
 * @param[in]  device Placement of @p out (DEVICE_CPU or DEVICE_GPU).
 * @param[out] out    Destination values. Must not be nullptr when
 *                    @p n is greater than zero.
 * @param[in]  n      Values to draw. Zero draws nothing.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidDevice for other placements, or the fill
 *         error.
 */
novaStatus_t default_fill_u32(Device_ device, uint32_t *out, size_t n);

/**
 * @brief Fill @p out with 64-bit draws flowing through the process
 *        default.
 *
 * @copydetails default_fill_u32()
 *
 * @param[in]  device Placement of @p out (DEVICE_CPU or DEVICE_GPU).
 * @param[out] out    Destination values. Must not be nullptr when
 *                    @p n is greater than zero.
 * @param[in]  n      Values to draw. Zero draws nothing.
 *
 * @return Status with @c err set to @c novaSuccess on success,
 *         @c novaInvalidDevice for other placements, or the fill
 *         error.
 */
novaStatus_t default_fill_u64(Device_ device, uint64_t *out, size_t n);

#ifdef __cplusplus
}
#endif
