/**
 * @file Uniform.c
 * @brief CPU uniform sampling operation.
 *
 * @details
 * Fills a tensor that already exists with samples drawn uniformly from
 * the half-open interval [low, high). The destination storage is
 * neither replaced nor resized.
 *
 * The fill is fused and chunked: one chunk of raw words is drawn into a
 * small thread-local buffer, mapped into the destination, and the same
 * buffer serves the next chunk. The raw words never occupy a buffer the
 * size of the destination, and the chunk stays in cache between the draw
 * and the map. The buffer is per-thread and capped at 256 KB.
 *
 * A serial fill walks the stream chunk after chunk. A parallel fill
 * gives every worker its own cursor into the same stream, positioned at
 * the worker's offset, so contiguous chunks reproduce the serial
 * sequence bit for bit. The handle advances once per fill, never once
 * per chunk.
 *
 * A draw keeps 24 mantissa bits and forces the low bit, so the unit
 * sample can never reach zero or one. The 64-bit mapper keeps 53 bits
 * for the same reason.
 *
 * The sample is computed with a fused multiply-add, so the scalar and
 * the vector variants agree bit for bit and the result does not depend
 * on the rounding the compiler picks for a plain multiply-add.
 *
 * The reduced-precision formats snap the interval to the values they
 * can hold before the sampling starts, so rounding a sample into the
 * format cannot escape the requested range. Packed formats draw two raw
 * words per storage byte and write one byte per pair.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/native/cpu/random/common.h>
#include <ncore/native/cpu/random/uniform.h>
#include <ncore/rand/generator.h>
#include <ncore/tensor.h>
#include <ncore/threading/parallel.h>
#include <ncore/threading/threads.h>

#ifdef NOVA_OPENMP
#include <omp.h>
#endif

/**
 * @def UNIFORM_GRID_MASK_16
 * @brief Magnitude mask of the 16-bit floating point formats.
 */
#define UNIFORM_GRID_MASK_16 0x7FFFU

/**
 * @def UNIFORM_GRID_MASK_8
 * @brief Magnitude mask of the 8-bit floating point formats.
 */
#define UNIFORM_GRID_MASK_8 0x7FU

/**
 * @def UNIFORM_GRID_MASK_4
 * @brief Magnitude mask of the 4-bit floating point format.
 */
#define UNIFORM_GRID_MASK_4 0x07U

/**
 * @def UNIFORM_FP16_MAX_FINITE
 * @brief Largest finite float16 magnitude pattern.
 */
#define UNIFORM_FP16_MAX_FINITE 0x7BFFU

/**
 * @def UNIFORM_BF16_MAX_FINITE
 * @brief Largest finite bfloat16 magnitude pattern.
 */
#define UNIFORM_BF16_MAX_FINITE 0x7F7FU

/**
 * @def UNIFORM_FP8E4M3_MAX_FINITE
 * @brief Largest finite float8_e4m3fn magnitude pattern.
 */
#define UNIFORM_FP8E4M3_MAX_FINITE 0x7EU

/**
 * @def UNIFORM_FP8E5M2_MAX_FINITE
 * @brief Largest finite float8_e5m2 magnitude pattern.
 */
#define UNIFORM_FP8E5M2_MAX_FINITE 0x7BU

/**
 * @def UNIFORM_FP4E2M1_MAX_FINITE
 * @brief Largest finite float4_e2m1fn_x2 magnitude pattern.
 */
#define UNIFORM_FP4E2M1_MAX_FINITE 0x07U

/**
 * @def UNIFORM_CHUNK_U32
 * @brief Raw 32-bit words held by the thread-local draw buffer.
 */
#define UNIFORM_CHUNK_U32 65536U

/**
 * @def UNIFORM_CHUNK_U64
 * @brief Raw 64-bit words held by the thread-local draw buffer.
 */
#define UNIFORM_CHUNK_U64 32768U

/**
 * @def UNIFORM_PARALLEL_FLOOR_CHEAP_MAP
 * @brief Storage units below which a fill whose map is cheaper than its
 *        draw stays serial. Covers f32, f16 and bf16.
 *
 * @details
 * Every extra worker costs one arbitrary jump, so a fill has to be long
 * enough for the fork to repay the jump before the split pays off. Maps
 * this cheap only win from the parallel draw itself, which needs millions
 * of units to amortize the fork. Measured on Zen 4, the split wins 6x
 * at the floor.
 */
#define UNIFORM_PARALLEL_FLOOR_CHEAP_MAP (1U << 20)

/**
 * @def UNIFORM_PARALLEL_FLOOR_F64
 * @brief Same floor for the 64-bit path, whose draw covers twice the
 *        samples per call.
 *
 * @details
 * Measured on Zen 4, the split wins 13x at twice the floor.
 */
#define UNIFORM_PARALLEL_FLOOR_F64 (1U << 19)

/**
 * @def UNIFORM_PARALLEL_FLOOR_FP8
 * @brief Storage units below which a byte-wide fill stays serial.
 *
 * @details
 * The fp8 maps cost a few scalar nanoseconds per unit. Measured on
 * Zen 4, the split wins 3.5x at the floor and 7x at six times it.
 */
#define UNIFORM_PARALLEL_FLOOR_FP8 (1U << 17)

/**
 * @def UNIFORM_PARALLEL_FLOOR_FP4
 * @brief Storage units below which a packed-nibble fill stays serial.
 *
 * @details
 * The fp4 map is the costliest scalar map in the file. Measured on
 * Zen 4, the split breaks even at the floor and wins 1.9x at twice
 * the floor.
 */
#define UNIFORM_PARALLEL_FLOOR_FP4 (1U << 15)

/**
 * @brief Unit-interval sample built from a raw 32-bit draw.
 *
 * @return A value strictly inside (0, 1) at mantissa resolution.
 */
static inline float uniform_unit_f32(uint32_t raw) {
  return (float)((raw >> 8) | 1U) * 0x1p-24f;
}

/**
 * @brief Unit-interval sample built from a raw 64-bit draw.
 *
 * @return A value strictly inside (0, 1) at mantissa resolution.
 */
static inline double uniform_unit_f64(uint64_t raw) {
  return (double)((raw >> 11) | 1ULL) * 0x1p-53;
}

/**
 * @brief Status describing a successful call.
 */
static inline novaStatus_t uniform_ok() {
  return (novaStatus_t){.err = novaSuccess,
                        .message = nova_get_error_msg(novaSuccess, nullptr)};
}

/**
 * @typedef UniformToFloat
 * @brief Decoder of a reduced-precision format.
 *
 * Takes the raw pattern and returns the value it encodes.
 */
typedef float (*UniformToFloat)(uint16_t bits);

/**
 * @typedef UniformFromFloat
 * @brief Encoder of a reduced-precision format.
 *
 * Takes a value and returns the raw pattern it takes, rounded to
 * nearest even.
 */
typedef uint16_t (*UniformFromFloat)(float value);

/**
 * @brief Next representable value above @p bits in the same format.
 *
 * @param[in] bits        Raw pattern of the current value.
 * @param[in] mask        Magnitude mask of the format.
 * @param[in] max_finite  Largest finite magnitude pattern.
 *
 * @return The successor pattern, or @p bits when the format has none.
 */
static uint16_t uniform_grid_next_up(uint16_t bits, uint16_t mask,
                                     uint16_t max_finite) {
  const uint16_t sign = (uint16_t)(bits & (uint16_t)(mask + 1U));
  const uint16_t mag = (uint16_t)(bits & mask);
  if (sign == 0U) {
    if (mag >= max_finite) {
      return bits;
    }
    return (uint16_t)(mag + 1U);
  }
  if (mag == 0U) {
    return 0U;
  }
  return (uint16_t)(sign | (uint16_t)(mag - 1U));
}

/**
 * @brief Next representable value below @p bits in the same format.
 *
 * @param[in] bits        Raw pattern of the current value.
 * @param[in] mask        Magnitude mask of the format.
 * @param[in] max_finite  Largest finite magnitude pattern.
 *
 * @return The predecessor pattern, or @p bits when the format has none.
 */
static uint16_t uniform_grid_next_down(uint16_t bits, uint16_t mask,
                                       uint16_t max_finite) {
  const uint16_t sign = (uint16_t)(bits & (uint16_t)(mask + 1U));
  const uint16_t mag = (uint16_t)(bits & mask);
  if (sign == 0U) {
    if (mag == 0U) {
      /* Negative zero duplicates positive zero, so the next distinct
         value below zero is the smallest negative subnormal. */
      return (uint16_t)((mask + 1U) | 1U);
    }
    return (uint16_t)(mag - 1U);
  }
  if (mag >= max_finite) {
    return bits;
  }
  return (uint16_t)(sign | (uint16_t)(mag + 1U));
}

/**
 * @brief Value carried by a float16 raw pattern.
 *
 * @details
 * The raw pattern is the object representation of the float16 value,
 * so it is read back as one. The format does not share its layout
 * with float32, so no shift is involved.
 *
 * @return The value the pattern encodes.
 */
static float uniform_fp16_value(uint16_t bits) {
  float16 half;
  memcpy(&half, &bits, sizeof half);
  return fp16_to_float(half);
}

/**
 * @brief Raw pattern a value takes in float16.
 *
 * @return The raw pattern, rounded to nearest even.
 */
static uint16_t uniform_fp16_pattern(float value) {
  return fp16_to_bits(fp16_from_float(value));
}

/**
 * @brief Value carried by a bfloat16 raw pattern.
 *
 * @details
 * The raw pattern is the object representation of the bfloat16 value,
 * so it is read back as one.
 *
 * @return The value the pattern encodes.
 */
static float uniform_bf16_value(uint16_t bits) {
  bfloat16 brain;
  memcpy(&brain, &bits, sizeof brain);
  return bf16_to_float(brain);
}

/**
 * @brief Raw pattern a value takes in bfloat16.
 *
 * @return The raw pattern, rounded to nearest even.
 */
static uint16_t uniform_bf16_pattern(float value) {
  return bf16_to_bits(bf16_from_float(value));
}

/**
 * @brief Value carried by a float8_e4m3fn raw pattern.
 *
 * @return The value the pattern encodes.
 */
static float uniform_fp8e4m3_value(uint16_t bits) {
  const uint32_t wide = fp8e4m3fn_to_f32_bits((float8_e4m3fn)(bits & 0xFFU));
  float value = 0.0f;
  memcpy(&value, &wide, sizeof value);
  return value;
}

/**
 * @brief Raw pattern a value takes in float8_e4m3fn.
 *
 * @return The raw pattern, rounded to nearest even.
 */
static uint16_t uniform_fp8e4m3_pattern(float value) {
  return fp8e4m3fn_from_float(value);
}

/**
 * @brief Value carried by a float8_e5m2 raw pattern.
 *
 * @return The value the pattern encodes.
 */
static float uniform_fp8e5m2_value(uint16_t bits) {
  const uint32_t wide = fp8e5m2_to_f32_bits((float8_e5m2)(bits & 0xFFU));
  float value = 0.0f;
  memcpy(&value, &wide, sizeof value);
  return value;
}

/**
 * @brief Raw pattern a value takes in float8_e5m2.
 *
 * @return The raw pattern, rounded to nearest even.
 */
static uint16_t uniform_fp8e5m2_pattern(float value) {
  return fp8e5m2_from_float(value);
}

/**
 * @brief Value carried by a float4_e2m1fn_x2 raw pattern.
 *
 * @details
 * The format has a single nibble per value, so the pattern is duplicated
 * into both lanes of a pair byte and the low lane is decoded.
 *
 * @return The value the pattern encodes.
 */
static float uniform_fp4e2m1_value(uint16_t bits) {
  const uint8_t nibble = (uint8_t)(bits & 0x0FU);
  const float4_e2m1fn_x2 pair = (float4_e2m1fn_x2)((nibble << 4) | nibble);
  float lo = 0.0f;
  float hi = 0.0f;
  fp4e2m1x2_to_floats(pair, &lo, &hi);
  return lo;
}

/**
 * @brief Raw pattern a value takes in float4_e2m1fn_x2.
 *
 * @return The raw pattern, rounded to nearest even.
 */
static uint16_t uniform_fp4e2m1_pattern(float value) {
  return (uint16_t)(fp4e2m1x2_from_floats(value, value) & 0x0FU);
}

/**
 * @brief Sampling interval of a float32 fill.
 *
 * @details
 * The upper bound moves one step inside the interval, so a sample can
 * never round up to @p high.
 *
 * @param[in]  low   Lower bound of the requested interval.
 * @param[in]  high  Exclusive upper bound of the requested interval.
 * @param[out] base  Lower end of the sampling interval.
 * @param[out] span  Width of the sampling interval.
 *
 * @return @c true when the interval can be sampled.
 */
static bool uniform_range_f32(float low, float high, float *base, float *span) {
  if (!(low < high)) {
    return false;
  }
  const float hi = nextafterf(high, low);
  const float width = hi - low;
  if (!isfinite(width)) {
    return false;
  }
  *base = low;
  *span = width;
  return true;
}

/**
 * @brief Sampling interval of a float64 fill.
 *
 * @copydetails uniform_range_f32()
 *
 * @param[in]  low   Lower bound of the requested interval.
 * @param[in]  high  Exclusive upper bound of the requested interval.
 * @param[out] base  Lower end of the sampling interval.
 * @param[out] span  Width of the sampling interval.
 *
 * @return @c true when the interval can be sampled.
 */
static bool uniform_range_f64(float low, float high, double *base,
                              double *span) {
  if (!(low < high)) {
    return false;
  }
  const double lo = (double)low;
  const double hi = nextafter((double)high, lo);
  const double width = hi - lo;
  if (!isfinite(width)) {
    return false;
  }
  *base = lo;
  *span = width;
  return true;
}

/**
 * @brief Sampling interval of a reduced-precision fill.
 *
 * @details
 * The interval is snapped to the values the format can hold: the
 * samples start at the smallest representable value at or above
 * @p low and stop below the smallest representable value at or above
 * @p high. Rounding the samples into the format therefore cannot
 * escape the requested range.
 *
 * The two 8-bit formats saturate differently: E4M3FN clamps to its
 * largest finite magnitude, E5M2 reaches infinity first. A bound
 * beyond the format therefore lands on a different effective edge.
 *
 * @param[in]  low         Lower bound of the requested interval.
 * @param[in]  high        Exclusive upper bound of the requested interval.
 * @param[in]  to_float    Decoder of the format.
 * @param[in]  from_float  Encoder of the format.
 * @param[in]  mask        Magnitude mask of the format.
 * @param[in]  max_finite  Largest finite magnitude pattern.
 * @param[out] base        Lower end of the sampling interval.
 * @param[out] span        Width of the sampling interval.
 *
 * @return @c true when the interval can be sampled.
 */
static bool uniform_range_narrow(float low, float high, UniformToFloat to_float,
                                 UniformFromFloat from_float, uint16_t mask,
                                 uint16_t max_finite, float *base,
                                 float *span) {
  if (!(low < high)) {
    return false;
  }

  uint16_t lo_bits = from_float(low);
  float lo_value = to_float(lo_bits);
  if (lo_value < low) {
    const uint16_t up = uniform_grid_next_up(lo_bits, mask, max_finite);
    if (up == lo_bits) {
      return false;
    }
    lo_bits = up;
    lo_value = to_float(lo_bits);
  }

  uint16_t hi_bits = from_float(high);
  float hi_value = to_float(hi_bits);
  if (hi_value >= high) {
    const uint16_t down = uniform_grid_next_down(hi_bits, mask, max_finite);
    if (down == hi_bits) {
      return false;
    }
    hi_bits = down;
    hi_value = to_float(hi_bits);
  }

  if (!(lo_value <= hi_value)) {
    return false;
  }

  *base = lo_value;
  *span = hi_value - lo_value;
  return true;
}

/**
 * @union UniformScratch
 * @brief Thread-local draw buffer backing every chunked fill.
 *
 * @details
 * The 32-bit and 64-bit views cover the same 256 KB, so a fill uses
 * whichever width its engine emits without reserving two buffers. The
 * union is aligned to a cache line so the vector maps read whole lines.
 * Native TLS backs the buffer; its 256 KB per-thread footprint is the
 * accepted cost on toolchains with small static TLS reserves.
 */
typedef union {
  ALIGN(64) uint32_t u32[UNIFORM_CHUNK_U32]; ///< 32-bit engine words.
  ALIGN(64) uint64_t u64[UNIFORM_CHUNK_U64]; ///< 64-bit engine words.
} UniformScratch;

static thread_local UniformScratch uniform_scratch;

/**
 * @typedef UniformMapU32
 * @brief Maps a chunk of raw 32-bit words into destination units.
 *
 * @return The destination pointer advanced past the written units.
 */
typedef void *(*UniformMapU32)(const uint32_t *raw, void *out, size_t units,
                               float base, float span);

/**
 * @typedef UniformMapU64
 * @brief Maps a chunk of raw 64-bit words into destination samples.
 *
 * @return The destination pointer advanced past the written samples.
 */
typedef void *(*UniformMapU64)(const uint64_t *raw, void *out, size_t units,
                               double base, double span);

/**
 * @struct UniformPlanU32
 * @brief Chunk schedule of a 32-bit-sourced fill.
 */
typedef struct {
  UniformMapU32 map;   ///< Turns raw words into destination units.
  size_t raw_per_unit; ///< Raw words consumed by one destination unit.
  size_t out_bytes;    ///< Destination bytes written by one unit.
  size_t parallel_floor; ///< Xoshiro units below which the fill stays serial.
} UniformPlanU32;

/**
 * @struct UniformPlanU64
 * @brief Chunk schedule of a 64-bit-sourced fill.
 */
typedef struct {
  UniformMapU64 map;   ///< Turns raw words into destination samples.
  size_t out_bytes;    ///< Destination bytes written by one sample.
  size_t parallel_floor; ///< Xoshiro units below which the fill stays serial.
} UniformPlanU64;

/**
 * @struct UniformGenerator
 * @brief Live handle backing one fill.
 *
 * @details
 * A private handle is used as is. A null handle leases the process
 * default for the whole fill, so another thread cannot interleave
 * between two chunks.
 */
typedef struct {
  Generator_ *gen;             ///< Handle the draw advances.
  DefaultGeneratorLease lease; ///< Held default, null for a private handle.
} UniformGenerator;

/**
 * @brief Resolve the handle a fill draws from.
 *
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] held       Resolved handle and lease storage.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
static novaStatus_t uniform_generator_acquire(Generator_ *generator,
                                              UniformGenerator *held) {
  if (generator != nullptr) {
    held->gen = generator;
    held->lease.gen = nullptr;
    return uniform_ok();
  }
  const novaStatus_t leased =
      default_generator_lease(DEVICE_CPU, &held->lease);
  if (leased.err != novaSuccess) {
    return leased;
  }
  held->gen = held->lease.gen;
  return uniform_ok();
}

/**
 * @brief Release the handle a fill drew from.
 *
 * @param[in,out] held  Handle and lease resolved by
 *                      @ref uniform_generator_acquire().
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
static novaStatus_t uniform_generator_release(UniformGenerator *held) {
  return default_generator_return(&held->lease);
}

#ifdef NOVA_OPENMP
/**
 * @brief Number of workers a fused fill should fork.
 *
 * @param[in] gen           Handle the fill draws from.
 * @param[in] units         Destination units in the fill.
 * @param[in] threads       Compute threads available to the caller.
 * @param[in] xoshiro_floor Demand below which a Xoshiro fill stays serial.
 * @param[in] cost_per_unit Draw cost of one unit.
 *
 * @return Worker count, at least one.
 */
static size_t uniform_worker_count(const Generator_ *gen, size_t units,
                                   uint32 threads, size_t xoshiro_floor,
                                   uint32 cost_per_unit) {
  if (threads < 2U) {
    return 1U;
  }
  const RngWork work = {
      .lanes = units,
      .cost_per_elem = cost_per_unit,
      .valid = true,
  };
  const ParallelDecision decision = decide_rng(&work, threads);
  if (!decision.go) {
    return 1U;
  }
  if (gen->engine == Xoshiro256 && units < xoshiro_floor) {
    return 1U;
  }
  return (size_t)decision.num_threads;
}
#endif

/**
 * @brief Draw and map a 32-bit-sourced fill.
 *
 * @details
 * A serial fill draws one chunk into the thread-local buffer, maps it,
 * and repeats. A parallel fill seats one cursor per worker at the
 * worker's stream offset; the last worker's cursor carries the final
 * state back to the handle.
 *
 * @param[in,out] gen      Live handle. Advanced by the fill.
 * @param[in]     plan     Chunk schedule.
 * @param[out]    out      Destination storage.
 * @param[in]     units    Destination units to produce.
 * @param[in]     base     Lower end of the sampling interval.
 * @param[in]     span     Width of the sampling interval.
 * @param[in]     threads  Compute threads available to the caller.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
static novaStatus_t uniform_draw_map_u32(Generator_ *gen,
                                         const UniformPlanU32 *plan, void *out,
                                         size_t units, float base, float span,
                                         uint32 threads) {
  const size_t chunk = UNIFORM_CHUNK_U32 / plan->raw_per_unit;
  size_t workers = 1U;
#ifdef NOVA_OPENMP
  workers = uniform_worker_count(gen, units, threads, plan->parallel_floor,
                                 2U);
#endif
  if (workers <= 1U) {
    size_t done = 0U;
    void *dst = out;
    while (done < units) {
      const size_t take = units - done < chunk ? units - done : chunk;
      const size_t raws = take * plan->raw_per_unit;
      const novaStatus_t drawn =
          generator_fill_u32(gen, uniform_scratch.u32, raws);
      if (drawn.err != novaSuccess) {
        return drawn;
      }
      dst = plan->map(uniform_scratch.u32, dst, take, base, span);
      done += take;
    }
    return uniform_ok();
  }
#ifdef NOVA_OPENMP
  const Generator_ source = *gen;
  Generator_ tail = {};
  int failed = 0;
  novaStatus_t failure = {};
#pragma omp parallel for num_threads(workers) schedule(static)
  for (size_t w = 0; w < workers; ++w) {
    const size_t begin = (units * w) / workers;
    const size_t end = (units * (w + 1U)) / workers;
    Generator_ cursor = {};
    const novaStatus_t seated =
        generator_u32_stream_at(&source, begin * plan->raw_per_unit, &cursor);
    if (seated.err != novaSuccess) {
#pragma omp critical(uniform_stream_error)
      {
        if (failed == 0) {
          failed = 1;
          failure = seated;
        }
      }
      continue;
    }
    void *dst = (char *)out + (begin * plan->out_bytes);
    size_t done = begin;
    while (done < end) {
      const size_t take = end - done < chunk ? end - done : chunk;
      const size_t raws = take * plan->raw_per_unit;
      const novaStatus_t drawn =
          generator_fill_u32(&cursor, uniform_scratch.u32, raws);
      if (drawn.err != novaSuccess) {
#pragma omp critical(uniform_stream_error)
        {
          if (failed == 0) {
            failed = 1;
            failure = drawn;
          }
        }
        break;
      }
      dst = plan->map(uniform_scratch.u32, dst, take, base, span);
      done += take;
    }
    if (w + 1U == workers) {
      tail = cursor;
    }
  }
  if (failed != 0) {
    return failure;
  }
  *gen = tail;
  return uniform_ok();
#else
  return uniform_ok();
#endif
}

/**
 * @brief Draw and map a 64-bit-sourced fill.
 *
 * @copydetails uniform_draw_map_u32()
 *
 * @param[in,out] gen      Live handle. Advanced by the fill.
 * @param[in]     plan     Chunk schedule.
 * @param[out]    out      Destination storage.
 * @param[in]     units    Destination samples to produce.
 * @param[in]     base     Lower end of the sampling interval.
 * @param[in]     span     Width of the sampling interval.
 * @param[in]     threads  Compute threads available to the caller.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
static novaStatus_t uniform_draw_map_u64(Generator_ *gen,
                                         const UniformPlanU64 *plan, void *out,
                                         size_t units, double base, double span,
                                         uint32 threads) {
  const size_t chunk = UNIFORM_CHUNK_U64;
  size_t workers = 1U;
#ifdef NOVA_OPENMP
  workers = uniform_worker_count(gen, units, threads, plan->parallel_floor,
                                 4U);
#endif
  if (workers <= 1U) {
    size_t done = 0U;
    void *dst = out;
    while (done < units) {
      const size_t take = units - done < chunk ? units - done : chunk;
      const novaStatus_t drawn =
          generator_fill_u64(gen, uniform_scratch.u64, take);
      if (drawn.err != novaSuccess) {
        return drawn;
      }
      dst = plan->map(uniform_scratch.u64, dst, take, base, span);
      done += take;
    }
    return uniform_ok();
  }
#ifdef NOVA_OPENMP
  const Generator_ source = *gen;
  Generator_ tail = {};
  int failed = 0;
  novaStatus_t failure = {};
#pragma omp parallel for num_threads(workers) schedule(static)
  for (size_t w = 0; w < workers; ++w) {
    const size_t begin = (units * w) / workers;
    const size_t end = (units * (w + 1U)) / workers;
    Generator_ cursor = {};
    const novaStatus_t seated =
        generator_u64_stream_at(&source, begin, &cursor);
    if (seated.err != novaSuccess) {
#pragma omp critical(uniform_stream_error)
      {
        if (failed == 0) {
          failed = 1;
          failure = seated;
        }
      }
      continue;
    }
    void *dst = (char *)out + (begin * plan->out_bytes);
    size_t done = begin;
    while (done < end) {
      const size_t take = end - done < chunk ? end - done : chunk;
      const novaStatus_t drawn =
          generator_fill_u64(&cursor, uniform_scratch.u64, take);
      if (drawn.err != novaSuccess) {
#pragma omp critical(uniform_stream_error)
        {
          if (failed == 0) {
            failed = 1;
            failure = drawn;
          }
        }
        break;
      }
      dst = plan->map(uniform_scratch.u64, dst, take, base, span);
      done += take;
    }
    if (w + 1U == workers) {
      tail = cursor;
    }
  }
  if (failed != 0) {
    return failure;
  }
  *gen = tail;
  return uniform_ok();
#else
  return uniform_ok();
#endif
}

/**
 * @brief Resolve the handle, run a 32-bit fill, and release the handle.
 *
 * @param[out] out        Destination storage.
 * @param[in]  units      Destination units to produce.
 * @param[in]  plan       Chunk schedule.
 * @param[in]  base       Lower end of the sampling interval.
 * @param[in]  span       Width of the sampling interval.
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] st         Caller-owned status.
 */
static void uniform_run_u32(void *out, size_t units,
                            const UniformPlanU32 *plan, float base, float span,
                            Generator_ *generator, novaStatus_t *st) {
  UniformGenerator held = {};
  const novaStatus_t acquired = uniform_generator_acquire(generator, &held);
  if (acquired.err != novaSuccess) {
    *st = acquired;
    return;
  }
  uint32 threads = 1U;
#ifdef NOVA_OPENMP
  threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    (void)uniform_generator_release(&held);
    return;
  }
#endif
  const novaStatus_t drawn =
      uniform_draw_map_u32(held.gen, plan, out, units, base, span, threads);
  const novaStatus_t released = uniform_generator_release(&held);
  *st = drawn.err != novaSuccess ? drawn : released;
}

/**
 * @brief Resolve the handle, run a 64-bit fill, and release the handle.
 *
 * @copydetails uniform_run_u32()
 *
 * @param[out] out        Destination storage.
 * @param[in]  units      Destination samples to produce.
 * @param[in]  plan       Chunk schedule.
 * @param[in]  base       Lower end of the sampling interval.
 * @param[in]  span       Width of the sampling interval.
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] st         Caller-owned status.
 */
static void uniform_run_u64(void *out, size_t units,
                            const UniformPlanU64 *plan, double base,
                            double span, Generator_ *generator,
                            novaStatus_t *st) {
  UniformGenerator held = {};
  const novaStatus_t acquired = uniform_generator_acquire(generator, &held);
  if (acquired.err != novaSuccess) {
    *st = acquired;
    return;
  }
  uint32 threads = 1U;
#ifdef NOVA_OPENMP
  threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    (void)uniform_generator_release(&held);
    return;
  }
#endif
  const novaStatus_t drawn =
      uniform_draw_map_u64(held.gen, plan, out, units, base, span, threads);
  const novaStatus_t released = uniform_generator_release(&held);
  *st = drawn.err != novaSuccess ? drawn : released;
}

/* =========================================================================
 * Scalar maps
 * ========================================================================= */

/**
 * @brief Map raw words into float32 samples using portable code.
 *
 * @return The destination pointer advanced past @p units samples.
 */
static void *uniform_map_f32_scalar(const uint32_t *raw, void *out, size_t units,
                                    float base, float span) {
  float *dst = out;
  for (size_t i = 0; i < units; ++i) {
    dst[i] = fmaf(uniform_unit_f32(raw[i]), span, base);
  }
  return dst + units;
}

/**
 * @brief Map raw words into float64 samples using portable code.
 *
 * @return The destination pointer advanced past @p units samples.
 */
static void *uniform_map_f64_scalar(const uint64_t *raw, void *out, size_t units,
                                    double base, double span) {
  double *dst = out;
  for (size_t i = 0; i < units; ++i) {
    dst[i] = fma(uniform_unit_f64(raw[i]), span, base);
  }
  return dst + units;
}

/**
 * @brief Map raw words into float16 samples using portable code.
 *
 * @return The destination pointer advanced past @p units samples.
 */
static void *uniform_map_fp16_scalar(const uint32_t *raw, void *out,
                                     size_t units, float base, float span) {
  float16 *dst = out;
  for (size_t i = 0; i < units; ++i) {
    dst[i] = fp16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Map raw words into bfloat16 samples using portable code.
 *
 * @return The destination pointer advanced past @p units samples.
 */
static void *uniform_map_bf16_scalar(const uint32_t *raw, void *out,
                                     size_t units, float base, float span) {
  bfloat16 *dst = out;
  for (size_t i = 0; i < units; ++i) {
    dst[i] = bf16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Map raw words into float8_e4m3fn samples using portable code.
 *
 * @return The destination pointer advanced past @p units samples.
 */
static void *uniform_map_fp8e4m3_scalar(const uint32_t *raw, void *out,
                                        size_t units, float base, float span) {
  float8_e4m3fn *dst = out;
  for (size_t i = 0; i < units; ++i) {
    dst[i] = fp8e4m3fn_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Map raw words into float8_e5m2 samples using portable code.
 *
 * @return The destination pointer advanced past @p units samples.
 */
static void *uniform_map_fp8e5m2_scalar(const uint32_t *raw, void *out,
                                        size_t units, float base, float span) {
  float8_e5m2 *dst = out;
  for (size_t i = 0; i < units; ++i) {
    dst[i] = fp8e5m2_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Map raw word pairs into float4_e2m1fn_x2 storage bytes.
 *
 * @details
 * One destination unit is one storage byte holding two samples, so the
 * map consumes two raw words per unit.
 *
 * @return The destination pointer advanced past @p units bytes.
 */
static void *uniform_map_fp4e2m1_scalar(const uint32_t *raw, void *out,
                                        size_t units, float base, float span) {
  float4_e2m1fn_x2 *dst = out;
  for (size_t i = 0; i < units; ++i) {
    const float lo = fmaf(uniform_unit_f32(raw[2U * i]), span, base);
    const float hi = fmaf(uniform_unit_f32(raw[(2U * i) + 1U]), span, base);
    dst[i] = fp4e2m1x2_from_floats(lo, hi);
  }
  return dst + units;
}

/* =========================================================================
 * Scalar variants
 * ========================================================================= */

/**
 * @brief Fill @p ten with uniform float32 samples using portable code.
 *
 * @param[in]  ten        Destination tensor, filled in place.
 * @param[in]  low        Lower bound of the sampling interval.
 * @param[in]  high       Exclusive upper bound of the sampling interval.
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] st         Caller-owned status. Must not be @c nullptr.
 */
void uniform_cpu_impl_f32_scalar(Tensor *ten, float low, float high,
                                 Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }

  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_f32(low, high, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float32. Use "
                                    "bounds float32 can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_f32_scalar,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.f32, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float64 samples using portable code.
 */
void uniform_cpu_impl_f64_scalar(Tensor *ten, float low, float high,
                                 Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }

  double base = 0.0;
  double span = 0.0;
  if (!uniform_range_f64(low, high, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float64. Use "
                                    "bounds float64 can hold.\n"};
    return;
  }
  static const UniformPlanU64 plan = {
      .map = uniform_map_f64_scalar,
      .out_bytes = sizeof(double),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_F64,
  };
  uniform_run_u64(ten->data.f64, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float16 samples using portable code.
 */
void uniform_cpu_impl_fp16_scalar(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }

  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_fp16_value,
                            uniform_fp16_pattern, UNIFORM_GRID_MASK_16,
                            UNIFORM_FP16_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the float16 "
                                    "format. Use bounds the format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_fp16_scalar,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float16),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.half, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform bfloat16 samples using portable code.
 */
void uniform_cpu_impl_bf16_scalar(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }

  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_bf16_value,
                            uniform_bf16_pattern, UNIFORM_GRID_MASK_16,
                            UNIFORM_BF16_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the bfloat16 "
                                    "format. Use bounds the format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_bf16_scalar,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(bfloat16),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.bf16, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float8_e4m3fn samples using portable code.
 */
void uniform_cpu_impl_fp8e4m3_scalar(Tensor *ten, float low, float high,
                                     Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }

  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_fp8e4m3_value,
                            uniform_fp8e4m3_pattern, UNIFORM_GRID_MASK_8,
                            UNIFORM_FP8E4M3_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the "
                                    "float8_e4m3fn format. Use bounds the "
                                    "format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_fp8e4m3_scalar,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float8_e4m3fn),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_FP8,
  };
  uniform_run_u32(ten->data.fp8e4m3fn, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float8_e5m2 samples using portable code.
 */
void uniform_cpu_impl_fp8e5m2_scalar(Tensor *ten, float low, float high,
                                     Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }

  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_fp8e5m2_value,
                            uniform_fp8e5m2_pattern, UNIFORM_GRID_MASK_8,
                            UNIFORM_FP8E5M2_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the "
                                    "float8_e5m2 format. Use bounds the "
                                    "format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_fp8e5m2_scalar,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float8_e5m2),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_FP8,
  };
  uniform_run_u32(ten->data.fp8e5m2, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float4_e2m1fn_x2 samples using portable
 *        code.
 *
 * @details
 * Two logical samples share each storage byte, so one destination unit is
 * one byte and consumes two raw words.
 */
void uniform_cpu_impl_fp4e2m1_scalar(Tensor *ten, float low, float high,
                                     Generator_ *generator, novaStatus_t *st) {
  const size_t units = ten->size;
  if (units == 0U) {
    *st = uniform_ok();
    return;
  }

  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_fp4e2m1_value,
                            uniform_fp4e2m1_pattern, UNIFORM_GRID_MASK_4,
                            UNIFORM_FP4E2M1_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the "
                                    "float4_e2m1fn_x2 format. Use bounds the "
                                    "format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_fp4e2m1_scalar,
      .raw_per_unit = 2U,
      .out_bytes = sizeof(float4_e2m1fn_x2),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_FP4,
  };
  uniform_run_u32(ten->data.fp4e2m1fn_x2, units, &plan, base, span, generator,
                  st);
}

/* =========================================================================
 * SIMD maps and variants
 * ========================================================================= */

#ifdef _GNUC_CLANG_

/**
 * @brief Map raw words into float32 samples using AVX-512.
 *
 * Requires: AVX512F
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx512f")]] static void *
uniform_map_f32_avx512(const uint32_t *raw, void *out, size_t units, float base,
                       float span) {
  float *dst = out;
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  const size_t rem = units - (units % step);
  const __m512i v_one = _mm512_set1_epi32(1);
  const __m512 v_scale = _mm512_set1_ps(0x1p-24f);
  const __m512 v_base = _mm512_set1_ps(base);
  const __m512 v_span = _mm512_set1_ps(span);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi32(v_raw, 8), v_one);
    const __m512 v_unit = _mm512_mul_ps(_mm512_cvtepi32_ps(v_mant), v_scale);
    _mm512_storeu_ps(&dst[i], _mm512_fmadd_ps(v_unit, v_span, v_base));
  }
  for (; i < units; ++i) {
    dst[i] = fmaf(uniform_unit_f32(raw[i]), span, base);
  }
  return dst + units;
}

/**
 * @brief Map raw words into float32 samples using AVX/AVX2.
 *
 * Requires: AVX/AVX2, FMA
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx,avx2,fma")]] static void *
uniform_map_f32_avx_avx2_fma(const uint32_t *raw, void *out, size_t units,
                             float base, float span) {
  float *dst = out;
  const size_t step = NOVA_SIMD_U32_WITH_AVX_AVX2;
  const size_t rem = units - (units % step);
  const __m256i v_one = _mm256_set1_epi32(1);
  const __m256 v_scale = _mm256_set1_ps(0x1p-24f);
  const __m256 v_base = _mm256_set1_ps(base);
  const __m256 v_span = _mm256_set1_ps(span);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m256i v_raw = _mm256_loadu_si256((const __m256i_u *)&raw[i]);
    const __m256i v_mant = _mm256_or_si256(_mm256_srli_epi32(v_raw, 8), v_one);
    const __m256 v_unit = _mm256_mul_ps(_mm256_cvtepi32_ps(v_mant), v_scale);
    _mm256_storeu_ps(&dst[i], _mm256_fmadd_ps(v_unit, v_span, v_base));
  }
  for (; i < units; ++i) {
    dst[i] = fmaf(uniform_unit_f32(raw[i]), span, base);
  }
  return dst + units;
}

/**
 * @brief Map raw words into float64 samples using AVX-512.
 *
 * Requires: AVX512F, AVX512DQ
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx512f,avx512dq")]] static void *
uniform_map_f64_avx512dq(const uint64_t *raw, void *out, size_t units,
                         double base, double span) {
  double *dst = out;
  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  const size_t rem = units - (units % step);
  const __m512i v_one = _mm512_set1_epi64(1);
  const __m512d v_scale = _mm512_set1_pd(0x1p-53);
  const __m512d v_base = _mm512_set1_pd(base);
  const __m512d v_span = _mm512_set1_pd(span);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi64(v_raw, 11), v_one);
    const __m512d v_unit = _mm512_mul_pd(_mm512_cvtepu64_pd(v_mant), v_scale);
    _mm512_storeu_pd(&dst[i], _mm512_fmadd_pd(v_unit, v_span, v_base));
  }
  for (; i < units; ++i) {
    dst[i] = fma(uniform_unit_f64(raw[i]), span, base);
  }
  return dst + units;
}

/**
 * @brief Map raw words into float64 samples using AVX/AVX2.
 *
 * Requires: AVX/AVX2, FMA
 *
 * @details
 * AVX2 has no unsigned quadword to double conversion, so the 53-bit
 * mantissa is rebuilt from its two halves. The split is exact: the
 * high part holds 21 bits, the low part 32 bits, and every integer
 * below 2^53 is representable as a double.
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx,avx2,fma")]] static void *
uniform_map_f64_avx_avx2_fma(const uint64_t *raw, void *out, size_t units,
                             double base, double span) {
  double *dst = out;
  const size_t step = NOVA_SIMD_U64_WITH_AVX_AVX2;
  const size_t rem = units - (units % step);
  const __m256i v_one = _mm256_set1_epi64x(1);
  const __m256i v_low32 = _mm256_set1_epi64x(0xFFFFFFFFLL);
  const __m256i v_pick = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
  const __m128i v_flip = _mm_set1_epi32(INT32_MIN);
  const __m256d v_magic = _mm256_set1_pd(2147483648.0);
  const __m256d v_two32 = _mm256_set1_pd(4294967296.0);
  const __m256d v_one_pd = _mm256_set1_pd(1.0);
  const __m256d v_scale = _mm256_set1_pd(0x1p-53);
  const __m256d v_base = _mm256_set1_pd(base);
  const __m256d v_span = _mm256_set1_pd(span);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m256i v_raw = _mm256_loadu_si256((const __m256i_u *)&raw[i]);
    const __m256i v_mant = _mm256_or_si256(_mm256_srli_epi64(v_raw, 11), v_one);
    const __m256i v_high = _mm256_srli_epi64(v_mant, 32);
    const __m256i v_low = _mm256_and_si256(v_mant, v_low32);
    const __m128i v_high_words =
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(v_high, v_pick));
    const __m128i v_low_words =
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(v_low, v_pick));
    /* The unsigned low half is read as signed and repaired by adding
       2^31, which keeps the value exact. */
    const __m256d v_high_d = _mm256_cvtepi32_pd(v_high_words);
    const __m256d v_low_d =
        _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm_xor_si128(v_low_words, v_flip)),
                        v_one_pd, v_magic);
    const __m256d v_value = _mm256_fmadd_pd(v_high_d, v_two32, v_low_d);
    const __m256d v_unit = _mm256_mul_pd(v_value, v_scale);
    _mm256_storeu_pd(&dst[i], _mm256_fmadd_pd(v_unit, v_span, v_base));
  }
  for (; i < units; ++i) {
    dst[i] = fma(uniform_unit_f64(raw[i]), span, base);
  }
  return dst + units;
}

/**
 * @brief Map raw words into float16 samples using AVX-512.
 *
 * Requires: AVX512F
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx512f")]] static void *
uniform_map_fp16_avx512(const uint32_t *raw, void *out, size_t units,
                        float base, float span) {
  float16 *dst = out;
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  const size_t rem = units - (units % step);
  const __m512i v_one = _mm512_set1_epi32(1);
  const __m512 v_scale = _mm512_set1_ps(0x1p-24f);
  const __m512 v_base = _mm512_set1_ps(base);
  const __m512 v_span = _mm512_set1_ps(span);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi32(v_raw, 8), v_one);
    const __m512 v_unit = _mm512_mul_ps(_mm512_cvtepi32_ps(v_mant), v_scale);
    const __m512 v_out = _mm512_fmadd_ps(v_unit, v_span, v_base);
    const __m256i v_half =
        _mm512_cvtps_ph(v_out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm256_storeu_si256((__m256i_u *)&dst[i], v_half);
  }
  for (; i < units; ++i) {
    dst[i] = fp16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Map raw words into float16 samples using AVX/AVX2 and F16C.
 *
 * Requires: AVX/AVX2, FMA, F16C
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx,avx2,fma,f16c")]] static void *
uniform_map_fp16_avx_avx2_fma_f16c(const uint32_t *raw, void *out, size_t units,
                                   float base, float span) {
  float16 *dst = out;
  const size_t step = NOVA_SIMD_U32_WITH_AVX_AVX2;
  const size_t rem = units - (units % step);
  const __m256i v_one = _mm256_set1_epi32(1);
  const __m256 v_scale = _mm256_set1_ps(0x1p-24f);
  const __m256 v_base = _mm256_set1_ps(base);
  const __m256 v_span = _mm256_set1_ps(span);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m256i v_raw = _mm256_loadu_si256((const __m256i_u *)&raw[i]);
    const __m256i v_mant = _mm256_or_si256(_mm256_srli_epi32(v_raw, 8), v_one);
    const __m256 v_unit = _mm256_mul_ps(_mm256_cvtepi32_ps(v_mant), v_scale);
    const __m256 v_out = _mm256_fmadd_ps(v_unit, v_span, v_base);
    const __m128i v_half =
        _mm256_cvtps_ph(v_out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i_u *)&dst[i], v_half);
  }
  for (; i < units; ++i) {
    dst[i] = fp16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Map raw words into bfloat16 samples using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx512f,avx512bf16")]] static void *
uniform_map_bf16_avx512bf16(const uint32_t *raw, void *out, size_t units,
                            float base, float span) {
  bfloat16 *dst = out;
  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  const size_t rem = units - (units % step);
  const __m512i v_one = _mm512_set1_epi32(1);
  const __m512 v_scale = _mm512_set1_ps(0x1p-24f);
  const __m512 v_base = _mm512_set1_ps(base);
  const __m512 v_span = _mm512_set1_ps(span);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi32(v_raw, 8), v_one);
    const __m512 v_unit = _mm512_mul_ps(_mm512_cvtepi32_ps(v_mant), v_scale);
    const __m512 v_out = _mm512_fmadd_ps(v_unit, v_span, v_base);
    const __m256bh v_brain = _mm512_cvtneps_pbh(v_out);
    memcpy(&dst[i], &v_brain, sizeof v_brain);
  }
  for (; i < units; ++i) {
    dst[i] = bf16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Map raw words into bfloat16 samples using AVX/AVX2.
 *
 * Requires: AVX/AVX2, FMA
 *
 * @details
 * AVX2 has no bfloat16 conversion, so the rounding is integer: add the
 * nearest-even bias to the float pattern, shift the low half away, and
 * pack the eight upper halves into eight 16-bit samples. NaNs are
 * canonicalized to match the scalar encoder.
 *
 * @return The destination pointer advanced past @p units samples.
 */
[[gnu::target("avx,avx2,fma")]] static void *
uniform_map_bf16_avx_avx2_fma(const uint32_t *raw, void *out, size_t units,
                              float base, float span) {
  bfloat16 *dst = out;
  const size_t step = NOVA_SIMD_U32_WITH_AVX_AVX2;
  const size_t rem = units - (units % step);
  const __m256i v_one = _mm256_set1_epi32(1);
  const __m256 v_scale = _mm256_set1_ps(0x1p-24f);
  const __m256 v_base = _mm256_set1_ps(base);
  const __m256 v_span = _mm256_set1_ps(span);
  const __m256i v_bias = _mm256_set1_epi32(0x7FFF);
  const __m256i v_lsb = _mm256_set1_epi32(1);
  const __m256i v_nan = _mm256_set1_epi32(0x7FC0);
  size_t i = 0;
  for (; i < rem; i += step) {
    const __m256i v_raw = _mm256_loadu_si256((const __m256i_u *)&raw[i]);
    const __m256i v_mant = _mm256_or_si256(_mm256_srli_epi32(v_raw, 8), v_one);
    const __m256 v_unit = _mm256_mul_ps(_mm256_cvtepi32_ps(v_mant), v_scale);
    const __m256 v_out = _mm256_fmadd_ps(v_unit, v_span, v_base);
    const __m256i v_bits = _mm256_castps_si256(v_out);
    const __m256i v_tie =
        _mm256_and_si256(_mm256_srli_epi32(v_bits, 16), v_lsb);
    const __m256i v_round =
        _mm256_add_epi32(v_bits, _mm256_add_epi32(v_bias, v_tie));
    const __m256i v_upper = _mm256_srli_epi32(v_round, 16);
    const __m256 v_isnan = _mm256_cmp_ps(v_out, v_out, _CMP_UNORD_Q);
    const __m256i v_fixed =
        _mm256_blendv_epi8(v_upper, v_nan, _mm256_castps_si256(v_isnan));
    const __m256i v_packed = _mm256_packus_epi32(v_fixed, v_fixed);
    const __m256i v_ordered = _mm256_permute4x64_epi64(v_packed, 0xD8);
    _mm_storeu_si128((__m128i_u *)&dst[i], _mm256_castsi256_si128(v_ordered));
  }
  for (; i < units; ++i) {
    dst[i] = bf16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }
  return dst + units;
}

/**
 * @brief Fill @p ten with uniform float32 samples using AVX-512.
 *
 * Requires: AVX512F
 */
[[gnu::target("avx512f")]] void
uniform_cpu_impl_f32_avx512(Tensor *ten, float low, float high,
                            Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_f32(low, high, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float32. Use "
                                    "bounds float32 can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_f32_avx512,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.f32, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float32 samples using AVX/AVX2.
 *
 * Requires: AVX/AVX2, FMA
 */
[[gnu::target("avx,avx2,fma")]] void
uniform_cpu_impl_f32_avx_avx2_fma(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_f32(low, high, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float32. Use "
                                    "bounds float32 can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_f32_avx_avx2_fma,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.f32, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float64 samples using AVX-512.
 *
 * Requires: AVX512F, AVX512DQ
 */
[[gnu::target("avx512f,avx512dq")]] void
uniform_cpu_impl_f64_avx512dq(Tensor *ten, float low, float high,
                              Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  double base = 0.0;
  double span = 0.0;
  if (!uniform_range_f64(low, high, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float64. Use "
                                    "bounds float64 can hold.\n"};
    return;
  }
  static const UniformPlanU64 plan = {
      .map = uniform_map_f64_avx512dq,
      .out_bytes = sizeof(double),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_F64,
  };
  uniform_run_u64(ten->data.f64, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float64 samples using AVX/AVX2.
 *
 * Requires: AVX/AVX2, FMA
 */
[[gnu::target("avx,avx2,fma")]] void
uniform_cpu_impl_f64_avx_avx2_fma(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  double base = 0.0;
  double span = 0.0;
  if (!uniform_range_f64(low, high, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float64. Use "
                                    "bounds float64 can hold.\n"};
    return;
  }
  static const UniformPlanU64 plan = {
      .map = uniform_map_f64_avx_avx2_fma,
      .out_bytes = sizeof(double),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_F64,
  };
  uniform_run_u64(ten->data.f64, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float16 samples using AVX-512.
 *
 * Requires: AVX512F
 */
[[gnu::target("avx512f")]] void
uniform_cpu_impl_fp16_avx512(Tensor *ten, float low, float high,
                             Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_fp16_value,
                            uniform_fp16_pattern, UNIFORM_GRID_MASK_16,
                            UNIFORM_FP16_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the float16 "
                                    "format. Use bounds the format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_fp16_avx512,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float16),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.half, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform float16 samples using AVX/AVX2 and F16C.
 *
 * Requires: AVX/AVX2, FMA, F16C
 */
[[gnu::target("avx,avx2,fma,f16c")]] void
uniform_cpu_impl_fp16_avx_avx2_fma_f16c(Tensor *ten, float low, float high,
                                        Generator_ *generator,
                                        novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_fp16_value,
                            uniform_fp16_pattern, UNIFORM_GRID_MASK_16,
                            UNIFORM_FP16_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the float16 "
                                    "format. Use bounds the format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_fp16_avx_avx2_fma_f16c,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(float16),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.half, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform bfloat16 samples using AVX-512.
 *
 * Requires: AVX512F, AVX512BF16
 */
[[gnu::target("avx512f,avx512bf16")]] void
uniform_cpu_impl_bf16_avx512bf16(Tensor *ten, float low, float high,
                                 Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_bf16_value,
                            uniform_bf16_pattern, UNIFORM_GRID_MASK_16,
                            UNIFORM_BF16_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the bfloat16 "
                                    "format. Use bounds the format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_bf16_avx512bf16,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(bfloat16),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.bf16, n, &plan, base, span, generator, st);
}

/**
 * @brief Fill @p ten with uniform bfloat16 samples using AVX/AVX2.
 *
 * Requires: AVX/AVX2, FMA
 */
[[gnu::target("avx,avx2,fma")]] void
uniform_cpu_impl_bf16_avx_avx2_fma(Tensor *ten, float low, float high,
                                   Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0U) {
    *st = uniform_ok();
    return;
  }
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_range_narrow(low, high, uniform_bf16_value,
                            uniform_bf16_pattern, UNIFORM_GRID_MASK_16,
                            UNIFORM_BF16_MAX_FINITE, &base, &span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit the bfloat16 "
                                    "format. Use bounds the format can hold.\n"};
    return;
  }
  static const UniformPlanU32 plan = {
      .map = uniform_map_bf16_avx_avx2_fma,
      .raw_per_unit = 1U,
      .out_bytes = sizeof(bfloat16),
      .parallel_floor = UNIFORM_PARALLEL_FLOOR_CHEAP_MAP,
  };
  uniform_run_u32(ten->data.bf16, n, &plan, base, span, generator, st);
}

#endif

/* =========================================================================
 * Dispatch
 * ========================================================================= */

#ifdef _GNUC_CLANG_

static const UniformVariant uniform_variants_f32[] = {
    {.fn = uniform_cpu_impl_f32_avx512, .isa = UniformISAAVX512F},
    {.fn = uniform_cpu_impl_f32_avx_avx2_fma, .isa = UniformISAAVX2FMA},
    {.fn = uniform_cpu_impl_f32_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_f64[] = {
    {.fn = uniform_cpu_impl_f64_avx512dq, .isa = UniformISAAVX512DQ},
    {.fn = uniform_cpu_impl_f64_avx_avx2_fma, .isa = UniformISAAVX2FMA},
    {.fn = uniform_cpu_impl_f64_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_fp16[] = {
    {.fn = uniform_cpu_impl_fp16_avx512, .isa = UniformISAAVX512F},
    {.fn = uniform_cpu_impl_fp16_avx_avx2_fma_f16c,
     .isa = UniformISAAVX2FMAF16C},
    {.fn = uniform_cpu_impl_fp16_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_bf16[] = {
    {.fn = uniform_cpu_impl_bf16_avx512bf16, .isa = UniformISAAVX512BF16},
    {.fn = uniform_cpu_impl_bf16_avx_avx2_fma, .isa = UniformISAAVX2FMA},
    {.fn = uniform_cpu_impl_bf16_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

#else

static const UniformVariant uniform_variants_f32[] = {
    {.fn = uniform_cpu_impl_f32_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_f64[] = {
    {.fn = uniform_cpu_impl_f64_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_fp16[] = {
    {.fn = uniform_cpu_impl_fp16_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_bf16[] = {
    {.fn = uniform_cpu_impl_bf16_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

#endif

static const UniformVariant uniform_variants_fp8e4m3[] = {
    {.fn = uniform_cpu_impl_fp8e4m3_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_fp8e5m2[] = {
    {.fn = uniform_cpu_impl_fp8e5m2_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

static const UniformVariant uniform_variants_fp4e2m1[] = {
    {.fn = uniform_cpu_impl_fp4e2m1_scalar, .isa = UniformISANone},
    {.fn = nullptr, .isa = UniformISANone},
};

/**
 * @brief Variant tables indexed by the tensor data type.
 *
 * @details
 * The table spans every data type, so a type without a variant reads
 * as a null entry instead of stepping past the end of the table.
 */
static const UniformVariant *const uniform_variant_tables[NUM_DTYPES] = {
    [Float32] = uniform_variants_f32,
    [Float64] = uniform_variants_f64,
    [Float16] = uniform_variants_fp16,
    [BFloat16] = uniform_variants_bf16,
    [Float8E4M3fn] = uniform_variants_fp8e4m3,
    [Float8E5M2] = uniform_variants_fp8e5m2,
    [Float4E2M1fn] = uniform_variants_fp4e2m1,
};

/**
 * @brief Select the fastest variant the host can run for @p dtype.
 *
 * @param[in] dtype  Data type of the destination tensor.
 *
 * @return The chosen variant, or null when the data type has none.
 */
static const UniformVariant *uniform_pick_variant(DType_ dtype) {
  if (dtype >= NUM_DTYPES) {
    return nullptr;
  }
  const UniformVariant *table = uniform_variant_tables[dtype];
  if (table == nullptr) {
    return nullptr;
  }
  for (const UniformVariant *variant = table; variant->fn != nullptr;
       ++variant) {
    if (uniform_isa_supported(variant->isa)) {
      return variant;
    }
  }
  return nullptr;
}

/**
 * @brief Fill @p ten with uniform samples drawn from [low, high).
 *
 * @details
 * The tensor is filled in place: the storage is neither replaced nor
 * resized. The version counter moves once per successful fill, so
 * aliasing code can see the mutation.
 *
 * The caller provides a live CPU tensor of a floating point type, with
 * its elements contiguous, and a generator that lives on the same
 * device. A data type without a variant is reported as
 * @c novaDtypeNotSupported.
 *
 * @param[in,out] ten       Destination tensor, filled in place.
 * @param[in]     low       Lower bound of the sampling interval.
 * @param[in]     high      Exclusive upper bound of the sampling interval.
 * @param[in]     generator Private generator, or null for the process
 *                          default.
 *
 * @return Status with @c err set to @c novaSuccess on success.
 */
novaStatus_t uniform_cpu_impl(Tensor *ten, float low, float high,
                              Generator_ *generator) {
  if (ten->dtype >= NUM_DTYPES) {
    return (novaStatus_t){.err = novaInvalidDtype,
                          .message = "Unknown data type for a uniform fill.\n"};
  }
  if (!is_floating(ten)) {
    return (novaStatus_t){.err = novaInvalidDtype,
                          .message = "Uniform fills need a floating point "
                                     "tensor.\n"};
  }
  if (generator != nullptr && generator->device != ten->device) {
    return (novaStatus_t){.err = novaInvalidDevice,
                          .message = "Device mismatch, cannot use a generator "
                                     "where the Generator.device != "
                                     "Tensor.device.\n"};
  }
  if (!(low < high)) {
    return (novaStatus_t){.err = novaInvalidValue,
                          .message = "Uniform fills need an ordered range. "
                                     "Use low < high.\n"};
  }

  const UniformVariant *variant = uniform_pick_variant(ten->dtype);
  if (variant == nullptr) {
    return (novaStatus_t){.err = novaDtypeNotSupported,
                          .message =
                              "No uniform operation for this data type.\n"};
  }

  novaStatus_t result = {0};
  variant->fn(ten, low, high, generator, &result);
  if (result.err != novaSuccess) {
    return result;
  }

  ten->version_ += 1;
  return result;
}
