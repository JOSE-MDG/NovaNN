/**
 * @file Uniform.c
 * @brief CPU uniform sampling operation.
 *
 * @details
 * Fills a tensor that already exists with samples drawn uniformly from
 * the half-open interval [low, high). The destination storage is
 * neither replaced nor resized.
 *
 * Every variant runs in two phases: the draw phase asks the generator
 * for raw words, the map phase turns those words into samples. The raw
 * words live in their own buffer, because the destination keeps the
 * sample type and sharing storage would let the compiler assume the
 * two do not alias.
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
 * format cannot escape the requested range.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
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
 * @struct UniformScratch
 * @brief Draw buffer backing a uniform fill.
 *
 * @details
 * The draws live in their own buffer: the destination tensor keeps the
 * sample type, so the two never share storage and the compiler cannot
 * assume they do not alias.
 */
typedef struct {
  RustHandle handle; ///< Rust-owned reservation behind the buffer.
  data_ptr data;     ///< Draw words, valid while the handle lives.
} UniformScratch;

/**
 * @brief Reserve the draw buffer of @p bytes.
 *
 * @param[in]  bytes    Bytes to reserve.
 * @param[out] scratch  Destination buffer description.
 * @param[out] st       Caller-owned status; set to success on completion.
 *
 * @return @c true when the buffer is ready to use.
 */
static bool uniform_scratch_acquire(size_t bytes, UniformScratch *scratch,
                                    novaStatus_t *st) {
  const novaStatus_t status = safe_allocator(bytes, DEVICE_CPU, false,
                                             &scratch->handle, nullptr, false);
  if (status.err != novaSuccess) {
    *st = status;
    return false;
  }
  scratch->data.v = get_data_from(&scratch->handle);
  return true;
}

/**
 * @brief Release the draw buffer.
 *
 * @param[in,out] scratch  Buffer to release.
 */
static novaStatus_t uniform_scratch_release(UniformScratch *scratch) {
  novaStatus_t status = {0};
  if (!release(&scratch->handle, &status)) {
    return (novaStatus_t){.err = novaReleaseError,
                          .message = "Draws need somewhere to land. The "
                                     "scratch reservation could not be "
                                     "released.\n"};
  }
  return uniform_ok();
}

/**
 * @brief Draw the raw words of a float32 fill and resolve its interval.
 *
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] scratch    Draw buffer, ready to use on success.
 * @param[in]  n          Number of samples.
 * @param[in]  low        Lower bound of the sampling interval.
 * @param[in]  high       Exclusive upper bound of the sampling interval.
 * @param[out] base       Lower end of the sampling interval.
 * @param[out] span       Width of the sampling interval.
 * @param[out] st         Caller-owned status; set to success on completion.
 *
 * @return @c true when the fill is ready to be mapped.
 */
static bool uniform_prepare_f32(Generator_ *generator, UniformScratch *scratch,
                                size_t n, float low, float high, float *base,
                                float *span, novaStatus_t *st) {
  if (!uniform_range_f32(low, high, base, span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float32. Use "
                                    "bounds float32 can hold.\n"};
    return false;
  }
  if (!uniform_scratch_acquire(n * sizeof(uint32_t), scratch, st)) {
    return false;
  }
  uint32_t *raw = scratch->data.u32;
  const novaStatus_t fill = uniform_fill_u32(DEVICE_CPU, generator, raw, n);
  if (fill.err != novaSuccess) {
    *st = fill;
    return false;
  }
  return true;
}

/**
 * @brief Draw the raw words of a float64 fill and resolve its interval.
 *
 * @copydetails uniform_prepare_f32()
 *
 * @param[in]  generator  Private generator, or null for the default.
 * @param[out] scratch    Draw buffer, ready to use on success.
 * @param[in]  n          Number of samples.
 * @param[in]  low        Lower bound of the sampling interval.
 * @param[in]  high       Exclusive upper bound of the sampling interval.
 * @param[out] base       Lower end of the sampling interval.
 * @param[out] span       Width of the sampling interval.
 * @param[out] st         Caller-owned status; set to success on completion.
 *
 * @return @c true when the fill is ready to be mapped.
 */
static bool uniform_prepare_f64(Generator_ *generator, UniformScratch *scratch,
                                size_t n, float low, float high, double *base,
                                double *span, novaStatus_t *st) {
  if (!uniform_range_f64(low, high, base, span)) {
    *st = (novaStatus_t){.err = novaInvalidValue,
                         .message = "The interval does not fit float64. Use "
                                    "bounds float64 can hold.\n"};
    return false;
  }
  if (!uniform_scratch_acquire(n * sizeof(uint64_t), scratch, st)) {
    return false;
  }
  uint64_t *raw = scratch->data.u64;
  const novaStatus_t fill = uniform_fill_u64(DEVICE_CPU, generator, raw, n);
  if (fill.err != novaSuccess) {
    *st = fill;
    return false;
  }
  return true;
}

/**
 * @brief Draw the raw words of a reduced-precision fill and resolve its
 *        interval.
 *
 * @copydetails uniform_prepare_f32()
 *
 * @param[in]  generator   Private generator, or null for the default.
 * @param[out] scratch     Draw buffer, ready to use on success.
 * @param[in]  n           Number of samples.
 * @param[in]  low         Lower bound of the sampling interval.
 * @param[in]  high        Exclusive upper bound of the sampling interval.
 * @param[in]  to_float    Decoder of the format.
 * @param[in]  from_float  Encoder of the format.
 * @param[in]  mask        Magnitude mask of the format.
 * @param[in]  max_finite  Largest finite magnitude pattern.
 * @param[out] base        Lower end of the sampling interval.
 * @param[out] span        Width of the sampling interval.
 * @param[in]  range_error Message reported when the format cannot hold the
 *                         interval.
 * @param[out] st          Caller-owned status; set to success on completion.
 *
 * @return @c true when the fill is ready to be mapped.
 */
static bool uniform_prepare_narrow(Generator_ *generator,
                                   UniformScratch *scratch, size_t n, float low,
                                   float high, UniformToFloat to_float,
                                   UniformFromFloat from_float, uint16_t mask,
                                   uint16_t max_finite, float *base,
                                   float *span, const char *range_error,
                                   novaStatus_t *st) {
  if (!uniform_range_narrow(low, high, to_float, from_float, mask, max_finite,
                            base, span)) {
    *st = (novaStatus_t){.err = novaInvalidValue, .message = range_error};
    return false;
  }
  if (!uniform_scratch_acquire(n * sizeof(uint32_t), scratch, st)) {
    return false;
  }
  uint32_t *raw = scratch->data.u32;
  const novaStatus_t fill = uniform_fill_u32(DEVICE_CPU, generator, raw, n);
  if (fill.err != novaSuccess) {
    *st = fill;
    return false;
  }
  return true;
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
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_f32(generator, &scratch, n, low, high, &base, &span,
                           st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const uint32_t *raw = scratch.data.u32;
  float *out = ten->data.f32;
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (size_t i = 0; i < n; ++i) {
    out[i] = fmaf(uniform_unit_f32(raw[i]), span, base);
  }

  *st = uniform_scratch_release(&scratch);
}

/**
 * @brief Fill @p ten with uniform float64 samples using portable code.
 */
void uniform_cpu_impl_f64_scalar(Tensor *ten, float low, float high,
                                 Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  double base = 0.0;
  double span = 0.0;
  if (!uniform_prepare_f64(generator, &scratch, n, low, high, &base, &span,
                           st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const uint64_t *raw = scratch.data.u64;
  double *out = ten->data.f64;
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (size_t i = 0; i < n; ++i) {
    out[i] = fma(uniform_unit_f64(raw[i]), span, base);
  }

  *st = uniform_scratch_release(&scratch);
}

/**
 * @brief Fill @p ten with uniform float16 samples using portable code.
 */
void uniform_cpu_impl_fp16_scalar(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_narrow(generator, &scratch, n, low, high,
                              uniform_fp16_value, uniform_fp16_pattern,
                              UNIFORM_GRID_MASK_16, UNIFORM_FP16_MAX_FINITE,
                              &base, &span,
                              "The interval does not fit the float16 format. "
                              "Use bounds the format can hold.\n",
                              st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const uint32_t *raw = scratch.data.u32;
  float16 *out = ten->data.half;
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (size_t i = 0; i < n; ++i) {
    out[i] = fp16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }

  *st = uniform_scratch_release(&scratch);
}

/**
 * @brief Fill @p ten with uniform bfloat16 samples using portable code.
 */
void uniform_cpu_impl_bf16_scalar(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_narrow(generator, &scratch, n, low, high,
                              uniform_bf16_value, uniform_bf16_pattern,
                              UNIFORM_GRID_MASK_16, UNIFORM_BF16_MAX_FINITE,
                              &base, &span,
                              "The interval does not fit the bfloat16 format. "
                              "Use bounds the format can hold.\n",
                              st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const uint32_t *raw = scratch.data.u32;
  bfloat16 *out = ten->data.bf16;
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (size_t i = 0; i < n; ++i) {
    out[i] = bf16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }

  *st = uniform_scratch_release(&scratch);
}

/**
 * @brief Fill @p ten with uniform float8_e4m3fn samples using portable code.
 */
void uniform_cpu_impl_fp8e4m3_scalar(Tensor *ten, float low, float high,
                                     Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_narrow(generator, &scratch, n, low, high,
                              uniform_fp8e4m3_value, uniform_fp8e4m3_pattern,
                              UNIFORM_GRID_MASK_8, UNIFORM_FP8E4M3_MAX_FINITE,
                              &base, &span,
                              "The interval does not fit the float8_e4m3fn "
                              "format. Use bounds the format can hold.\n",
                              st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const uint32_t *raw = scratch.data.u32;
  float8_e4m3fn *out = ten->data.fp8e4m3fn;
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (size_t i = 0; i < n; ++i) {
    out[i] = fp8e4m3fn_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }

  *st = uniform_scratch_release(&scratch);
}

/**
 * @brief Fill @p ten with uniform float8_e5m2 samples using portable code.
 */
void uniform_cpu_impl_fp8e5m2_scalar(Tensor *ten, float low, float high,
                                     Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_narrow(generator, &scratch, n, low, high,
                              uniform_fp8e5m2_value, uniform_fp8e5m2_pattern,
                              UNIFORM_GRID_MASK_8, UNIFORM_FP8E5M2_MAX_FINITE,
                              &base, &span,
                              "The interval does not fit the float8_e5m2 "
                              "format. Use bounds the format can hold.\n",
                              st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const uint32_t *raw = scratch.data.u32;
  float8_e5m2 *out = ten->data.fp8e5m2;
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (size_t i = 0; i < n; ++i) {
    out[i] = fp8e5m2_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }

  *st = uniform_scratch_release(&scratch);
}

/* =========================================================================
 * SIMD variants
 * ========================================================================= */

#ifdef _GNUC_CLANG_

/**
 * @brief Fill @p ten with uniform float32 samples using AVX-512.
 *
 * Requires: AVX512F
 */
[[gnu::target("avx512f")]] void
uniform_cpu_impl_f32_avx512(Tensor *ten, float low, float high,
                            Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_f32(generator, &scratch, n, low, high, &base, &span,
                           st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = n - (n % step);
  float *out = ten->data.f32;
  const uint32_t *raw = scratch.data.u32;
  const __m512i v_one = _mm512_set1_epi32(1);
  const __m512 v_scale = _mm512_set1_ps(0x1p-24f);
  const __m512 v_base = _mm512_set1_ps(base);
  const __m512 v_span = _mm512_set1_ps(span);
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (i = 0; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi32(v_raw, 8), v_one);
    const __m512 v_unit = _mm512_mul_ps(_mm512_cvtepi32_ps(v_mant), v_scale);
    _mm512_storeu_ps(&out[i], _mm512_fmadd_ps(v_unit, v_span, v_base));
  }
#ifdef NOVA_OPENMP
  i = rem;
#endif
  for (; i < n; ++i) {
    out[i] = fmaf(uniform_unit_f32(raw[i]), span, base);
  }

  *st = uniform_scratch_release(&scratch);
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
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_f32(generator, &scratch, n, low, high, &base, &span,
                           st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const size_t step = NOVA_SIMD_U32_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = n - (n % step);
  float *out = ten->data.f32;
  const uint32_t *raw = scratch.data.u32;
  const __m256i v_one = _mm256_set1_epi32(1);
  const __m256 v_scale = _mm256_set1_ps(0x1p-24f);
  const __m256 v_base = _mm256_set1_ps(base);
  const __m256 v_span = _mm256_set1_ps(span);
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (i = 0; i < rem; i += step) {
    const __m256i v_raw = _mm256_loadu_si256((const __m256i_u *)&raw[i]);
    const __m256i v_mant = _mm256_or_si256(_mm256_srli_epi32(v_raw, 8), v_one);
    const __m256 v_unit = _mm256_mul_ps(_mm256_cvtepi32_ps(v_mant), v_scale);
    _mm256_storeu_ps(&out[i], _mm256_fmadd_ps(v_unit, v_span, v_base));
  }
#ifdef NOVA_OPENMP
  i = rem;
#endif
  for (; i < n; ++i) {
    out[i] = fmaf(uniform_unit_f32(raw[i]), span, base);
  }

  *st = uniform_scratch_release(&scratch);
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
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  double base = 0.0;
  double span = 0.0;
  if (!uniform_prepare_f64(generator, &scratch, n, low, high, &base, &span,
                           st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const size_t step = NOVA_SIMD_U64_WITH_AVX512F;
  size_t i = 0;
  size_t rem = n - (n % step);
  double *out = ten->data.f64;
  const uint64_t *raw = scratch.data.u64;
  const __m512i v_one = _mm512_set1_epi64(1);
  const __m512d v_scale = _mm512_set1_pd(0x1p-53);
  const __m512d v_base = _mm512_set1_pd(base);
  const __m512d v_span = _mm512_set1_pd(span);
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (i = 0; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi64(v_raw, 11), v_one);
    const __m512d v_unit = _mm512_mul_pd(_mm512_cvtepu64_pd(v_mant), v_scale);
    _mm512_storeu_pd(&out[i], _mm512_fmadd_pd(v_unit, v_span, v_base));
  }
#ifdef NOVA_OPENMP
  i = rem;
#endif
  for (; i < n; ++i) {
    out[i] = fma(uniform_unit_f64(raw[i]), span, base);
  }

  *st = uniform_scratch_release(&scratch);
}

/**
 * @brief Fill @p ten with uniform float64 samples using AVX/AVX2.
 *
 * Requires: AVX/AVX2, FMA
 *
 * @details
 * AVX2 has no unsigned quadword to double conversion, so the 53-bit
 * mantissa is rebuilt from its two halves. The split is exact: the
 * high part holds 21 bits, the low part 32 bits, and every integer
 * below 2^53 is representable as a double.
 */
[[gnu::target("avx,avx2,fma")]] void
uniform_cpu_impl_f64_avx_avx2_fma(Tensor *ten, float low, float high,
                                  Generator_ *generator, novaStatus_t *st) {
  const size_t n = ten->size;
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  double base = 0.0;
  double span = 0.0;
  if (!uniform_prepare_f64(generator, &scratch, n, low, high, &base, &span,
                           st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const size_t step = NOVA_SIMD_U64_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = n - (n % step);
  double *out = ten->data.f64;
  const uint64_t *raw = scratch.data.u64;
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
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (i = 0; i < rem; i += step) {
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
    _mm256_storeu_pd(&out[i], _mm256_fmadd_pd(v_unit, v_span, v_base));
  }
#ifdef NOVA_OPENMP
  i = rem;
#endif
  for (; i < n; ++i) {
    out[i] = fma(uniform_unit_f64(raw[i]), span, base);
  }

  *st = uniform_scratch_release(&scratch);
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
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_narrow(generator, &scratch, n, low, high,
                              uniform_fp16_value, uniform_fp16_pattern,
                              UNIFORM_GRID_MASK_16, UNIFORM_FP16_MAX_FINITE,
                              &base, &span,
                              "The interval does not fit the float16 format. "
                              "Use bounds the format can hold.\n",
                              st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = n - (n % step);
  float16 *out = ten->data.half;
  const uint32_t *raw = scratch.data.u32;
  const __m512i v_one = _mm512_set1_epi32(1);
  const __m512 v_scale = _mm512_set1_ps(0x1p-24f);
  const __m512 v_base = _mm512_set1_ps(base);
  const __m512 v_span = _mm512_set1_ps(span);
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (i = 0; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi32(v_raw, 8), v_one);
    const __m512 v_unit = _mm512_mul_ps(_mm512_cvtepi32_ps(v_mant), v_scale);
    const __m512 v_out = _mm512_fmadd_ps(v_unit, v_span, v_base);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
    const __m256i v_half =
        _mm512_cvtps_ph(v_out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#pragma GCC diagnostic pop
    _mm256_storeu_si256((__m256i_u *)&out[i], v_half);
  }
#ifdef NOVA_OPENMP
  i = rem;
#endif
  for (; i < n; ++i) {
    out[i] = fp16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }

  *st = uniform_scratch_release(&scratch);
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
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_narrow(generator, &scratch, n, low, high,
                              uniform_fp16_value, uniform_fp16_pattern,
                              UNIFORM_GRID_MASK_16, UNIFORM_FP16_MAX_FINITE,
                              &base, &span,
                              "The interval does not fit the float16 format. "
                              "Use bounds the format can hold.\n",
                              st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const size_t step = NOVA_SIMD_U32_WITH_AVX_AVX2;
  size_t i = 0;
  size_t rem = n - (n % step);
  float16 *out = ten->data.half;
  const uint32_t *raw = scratch.data.u32;
  const __m256i v_one = _mm256_set1_epi32(1);
  const __m256 v_scale = _mm256_set1_ps(0x1p-24f);
  const __m256 v_base = _mm256_set1_ps(base);
  const __m256 v_span = _mm256_set1_ps(span);
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (i = 0; i < rem; i += step) {
    const __m256i v_raw = _mm256_loadu_si256((const __m256i_u *)&raw[i]);
    const __m256i v_mant = _mm256_or_si256(_mm256_srli_epi32(v_raw, 8), v_one);
    const __m256 v_unit = _mm256_mul_ps(_mm256_cvtepi32_ps(v_mant), v_scale);
    const __m256 v_out = _mm256_fmadd_ps(v_unit, v_span, v_base);
    const __m128i v_half =
        _mm256_cvtps_ph(v_out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i_u *)&out[i], v_half);
  }
#ifdef NOVA_OPENMP
  i = rem;
#endif
  for (; i < n; ++i) {
    out[i] = fp16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }

  *st = uniform_scratch_release(&scratch);
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
  if (n == 0) {
    *st = uniform_ok();
    return;
  }

  UniformScratch scratch = {0};
  float base = 0.0f;
  float span = 0.0f;
  if (!uniform_prepare_narrow(generator, &scratch, n, low, high,
                              uniform_bf16_value, uniform_bf16_pattern,
                              UNIFORM_GRID_MASK_16, UNIFORM_BF16_MAX_FINITE,
                              &base, &span,
                              "The interval does not fit the bfloat16 format. "
                              "Use bounds the format can hold.\n",
                              st)) {
    uniform_scratch_release(&scratch);
    return;
  }

  const size_t step = NOVA_SIMD_U32_WITH_AVX512F;
  size_t i = 0;
  size_t rem = n - (n % step);
  bfloat16 *out = ten->data.bf16;
  const uint32_t *raw = scratch.data.u32;
  const __m512i v_one = _mm512_set1_epi32(1);
  const __m512 v_scale = _mm512_set1_ps(0x1p-24f);
  const __m512 v_base = _mm512_set1_ps(base);
  const __m512 v_span = _mm512_set1_ps(span);
#ifdef NOVA_OPENMP
  const uint32 threads = get_num_threads_from(ParallelComputeGroup, st);
  if (st->err != novaSuccess) {
    uniform_scratch_release(&scratch);
    return;
  }
  const ElementwiseWork work = make_elementwise_work(ten);
  const ParallelDecision decision = decide_elementwise(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
  for (i = 0; i < rem; i += step) {
    const __m512i v_raw = _mm512_loadu_si512((const void *)&raw[i]);
    const __m512i v_mant = _mm512_or_si512(_mm512_srli_epi32(v_raw, 8), v_one);
    const __m512 v_unit = _mm512_mul_ps(_mm512_cvtepi32_ps(v_mant), v_scale);
    const __m512 v_out = _mm512_fmadd_ps(v_unit, v_span, v_base);
    const __m256bh v_brain = _mm512_cvtneps_pbh(v_out);
    memcpy(&out[i], &v_brain, sizeof v_brain);
  }
#ifdef NOVA_OPENMP
  i = rem;
#endif
  for (; i < n; ++i) {
    out[i] = bf16_from_float(fmaf(uniform_unit_f32(raw[i]), span, base));
  }

  *st = uniform_scratch_release(&scratch);
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
