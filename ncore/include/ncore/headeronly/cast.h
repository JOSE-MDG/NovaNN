/**
 * @file cast.h
 * @brief SIMD-dispatched type-cast kernels for all DType_ pairs.
 *
 * @details
 * Header-only file that provides 90 inline cast functions, each
 * selecting the best available SIMD implementation at runtime
 * based on the CPU capabilities reported by @ref get_cpu_capabilities().
 *
 * ## Architecture
 *
 * Every cast function follows the same dispatch pattern:
 *
 * ```
 * tfP_to_Q_(src, dst):
 *     if (AVX-512 available)  → lookup_tP_to_Q_[0](src, dst)
 *     if (AVX2/F16C available)→ lookup_tP_to_Q_[1](src, dst)
 *     ...
 *     fallback               → lookup_tP_to_Q_[N](src, dst)
 * ```
 *
 * The actual kernel implementations live in @ref cast_tables.h
 * (generated lookup tables) and are resolved via function pointers.
 * This file only contains the dispatch logic.
 *
 * ## X-Macro categories
 *
 * The cast functions are grouped into seven X-macro families that
 * enumerate every supported (src, dst) pair:
 */
// clang-format off
/**
 * | Macro              | Pairs | Description                          |
 * |--------------------|------:|--------------------------------------|
 * | @ref CAST_FP_TYPES |    12 | Float ↔ Float (all combinations)     |
 * | @ref CAST_FP_TO_INT|    24 | Float → Signed/Unsigned integer       |
 * | @ref CAST_INT_TO_FP|    24 | Signed/Unsigned integer → Float       |
 * | @ref CAST_SINT_TO_SINT| 6 | Signed integer width changes          |
 * | @ref CAST_UINT_TO_UINT| 6 | Unsigned integer width changes        |
 * | @ref CAST_SINT_TO_UINT| 9 | Signed → Unsigned (sign + width)      |
 * | @ref CAST_UINT_TO_SINT| 9 | Unsigned → Signed (sign + width)      |
 */
// clang-format on
/**
 * Each macro accepts a callback `F(from, to)` and expands to a
 * series of `F(...)` invocations.  The @ref DECL_CAST macro uses
 * these to generate forward declarations for all 90 functions.
 *
 * ## Quantised types
 *
 * `QSigned8` and `QUnSigned8` share their native storage type
 * (`int8_t` / `uint8_t`) with `Signed8` / `UnSigned8`, so no
 * separate quantised cast kernels exist.  The dispatch table
 * (`cast_dispatch`) maps quantised types to the same underlying
 * cast functions.
 *
 * ## Naming convention
 *
 * All cast functions follow the pattern `t{src}_to_{dst}_()`:
 * - `tfp16_to_f32_()` — Float16 → Float32
 * - `ts8_to_u32_()`   — Signed8 → UnSigned32
 * - `tu64_to_s8_()`   — UnSigned64 → Signed8
 *
 * The trailing underscore indicates an internal (non-public)
 * function.
 *
 * @see cast_dispatch        12×12 dispatch table (src → dst).
 * @see cast_dispatch_tables.c  Constructor that populates the table.
 * @see cast_tables.h        Per-kernel lookup tables.
 * @see simd.h               CPU capability detection.
 * @see dtype.h              @ref DType_ enum and @ref NUM_DTYPES.
 */

#pragma once

#include <immintrin.h>
#include <ncore/macros.h>
#include <ncore/simd.h>
#include <ncore/tables/cast_tables.h>
#include <ncore/tensor.h>

/**
 * @defgroup CAST_FP_TYPES Float ↔ Float casts
 * @{
 * @brief X-macro enumerating all float-to-float conversion pairs.
 *
 * @details
 * Generates 12 entries covering every combination of
 * `{Float16, Float32, Float64, BFloat16}` as both source and
 * target (excluding identity).
 */
#define CAST_FP_TYPES(F)                                                       \
  F(fp16, f32)                                                                 \
  F(fp16, f64)                                                                 \
  F(fp16, bf16)                                                                \
  F(f32, fp16)                                                                 \
  F(f32, f64)                                                                  \
  F(f32, bf16)                                                                 \
  F(bf16, fp16) F(bf16, f32) F(bf16, f64) F(f64, fp16) F(f64, f32) F(f64, bf16)
/** @} */

/**
 * @defgroup CAST_FP_TO_INT Float → Integer casts
 * @{
 * @brief X-macro enumerating all float-to-integer conversion pairs.
 *
 * @details
 * Generates 24 entries: each of the 4 float types to each of the
 * 6 integer types (`Signed8`, `Signed32`, `Signed64`, `UnSigned8`,
 * `UnSigned32`, `UnSigned64`).
 */
#define CAST_FP_TO_INT(F)                                                      \
  F(fp16, s8)                                                                  \
  F(fp16, s32)                                                                 \
  F(fp16, s64)                                                                 \
  F(fp16, u8)                                                                  \
  F(fp16, u32)                                                                 \
  F(fp16, u64)                                                                 \
  F(bf16, s8)                                                                  \
  F(bf16, s32)                                                                 \
  F(bf16, s64)                                                                 \
  F(bf16, u8)                                                                  \
  F(bf16, u32)                                                                 \
  F(bf16, u64)                                                                 \
  F(f32, s8)                                                                   \
  F(f32, s32)                                                                  \
  F(f32, s64)                                                                  \
  F(f32, u8)                                                                   \
  F(f32, u32)                                                                  \
  F(f32, u64)                                                                  \
  F(f64, s8) F(f64, s32) F(f64, s64) F(f64, u8) F(f64, u32) F(f64, u64)
/** @} */

/**
 * @defgroup CAST_INT_TO_FP Integer → Float casts
 * @{
 * @brief X-macro enumerating all integer-to-float conversion pairs.
 *
 * @details
 * Generates 24 entries: each of the 6 integer types to each of
 * the 4 float types.
 */
#define CAST_INT_TO_FP(F)                                                      \
  F(s8, fp16)                                                                  \
  F(s32, fp16)                                                                 \
  F(s64, fp16)                                                                 \
  F(u8, fp16)                                                                  \
  F(u32, fp16)                                                                 \
  F(u64, fp16)                                                                 \
  F(s8, bf16)                                                                  \
  F(s32, bf16)                                                                 \
  F(s64, bf16)                                                                 \
  F(u8, bf16)                                                                  \
  F(u32, bf16)                                                                 \
  F(u64, bf16)                                                                 \
  F(s8, f32)                                                                   \
  F(s32, f32)                                                                  \
  F(s64, f32)                                                                  \
  F(u8, f32)                                                                   \
  F(u32, f32)                                                                  \
  F(u64, f32)                                                                  \
  F(s8, f64) F(s32, f64) F(s64, f64) F(u8, f64) F(u32, f64) F(u64, f64)
/** @} */

/**
 * @defgroup CAST_SINT_TO_SINT Signed ↔ Signed integer casts
 * @{
 * @brief X-macro for signed integer width-changing conversions.
 */
#define CAST_SINT_TO_SINT(F)                                                   \
  F(s8, s32) F(s8, s64) F(s32, s8) F(s32, s64) F(s64, s8) F(s64, s32)
/** @} */

/**
 * @defgroup CAST_UINT_TO_UINT Unsigned ↔ Unsigned integer casts
 * @{
 * @brief X-macro for unsigned integer width-changing conversions.
 */
#define CAST_UINT_TO_UINT(F)                                                   \
  F(u8, u32) F(u8, u64) F(u32, u8) F(u32, u64) F(u64, u8) F(u64, u32)
/** @} */

/**
 * @defgroup CAST_SINT_TO_UINT Signed → Unsigned integer casts
 * @{
 * @brief X-macro for signed-to-unsigned conversions (sign +
 *        width change).
 */
#define CAST_SINT_TO_UINT(F)                                                   \
  F(s8, u8)                                                                    \
  F(s8, u32)                                                                   \
  F(s8, u64)                                                                   \
  F(s32, u8) F(s32, u32) F(s32, u64) F(s64, u8) F(s64, u32) F(s64, u64)
/** @} */

/**
 * @defgroup CAST_UINT_TO_SINT Unsigned → Signed integer casts
 * @{
 * @brief X-macro for unsigned-to-signed conversions (sign +
 *        width change).
 */
#define CAST_UINT_TO_SINT(F)                                                   \
  F(u8, s8)                                                                    \
  F(u8, s32)                                                                   \
  F(u8, s64)                                                                   \
  F(u32, s8) F(u32, s32) F(u32, s64) F(u64, s8) F(u64, s32) F(u64, s64)
/** @} */

/**
 * @brief Generate a forward declaration for a single cast function.
 *
 * @details
 * Expands to:
 * ```c
 * static inline void t{from}_to_{to}_(const Tensor *restrict src,
 *                                     Tensor *restrict dst);
 * ```
 *
 * Called via the X-macro families to declare all 90 cast
 * functions before their definitions.
 *
 * @param from  Source dtype short name (e.g., `fp16`, `s8`).
 * @param to    Target dtype short name (e.g., `f32`, `u64`).
 */
#define DECL_CAST(from, to)                                                    \
  static inline void t##from##_to_##to##_(const Tensor *restrict src,          \
                                          Tensor *restrict dst);

CAST_FP_TYPES(DECL_CAST)
CAST_FP_TO_INT(DECL_CAST)
CAST_INT_TO_FP(DECL_CAST)
CAST_SINT_TO_SINT(DECL_CAST)
CAST_UINT_TO_UINT(DECL_CAST)
CAST_SINT_TO_UINT(DECL_CAST)
CAST_UINT_TO_SINT(DECL_CAST)

/**
 * @def Caps_
 * @brief Shorthand for `get_cpu_capabilities()`.
 *
 * @details
 * Resolves to a `const Capabilities_ *` at each call site.
 * The pointer is cached per-thread by the underlying
 * `call_once` initialisation, so repeated access is cheap.
 *
 * @see get_cpu_capabilities()  Returns the singleton capabilities.
 * @see Capabilities_           Struct with SIMD feature flags.
 */
#define Caps_ get_cpu_capabilities()

/**
 * @brief Casts fp16 tensor to f32.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, F16C, Scalar.
 */
static inline void tfp16_to_f32_(const Tensor *restrict src,
                                 Tensor *restrict dst) {

  if (Caps_->avx512f_) {
    lookup_tfp16_to_f32_[0](src, dst);
    return;
  }

  if ((Caps_->avx_ || Caps_->avx2_) && Caps_->f16c_) {
    lookup_tfp16_to_f32_[1](src, dst);
    return;
  }

  lookup_tfp16_to_f32_[2](src, dst);
}

/**
 * @brief Casts fp16 tensor to f64.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F+AVX512FP16, F16C, Scalar.
 */
static inline void tfp16_to_f64_(const Tensor *restrict src,
                                 Tensor *restrict dst) {

  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tfp16_to_f64_[0](src, dst);
    return;
  }

  if ((Caps_->avx_ || Caps_->avx2_) && Caps_->f16c_) {
    lookup_tfp16_to_f64_[1](src, dst);
    return;
  }

  lookup_tfp16_to_f64_[2](src, dst);
}

/**
 * @brief Casts fp16 tensor to bf16.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tfp16_to_bf16_(const Tensor *restrict src,
                                  Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_tfp16_to_bf16_[0](src, dst);
    return;
  }

  lookup_tfp16_to_bf16_[1](src, dst);
}

/**
 * @brief Casts f32 tensor to fp16.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, F16C, Scalar.
 */
static inline void tf32_to_fp16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {

  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tf32_to_fp16_[0](src, dst);
    return;
  }

  if ((Caps_->avx_ || Caps_->avx2_) && Caps_->f16c_) {
    lookup_tf32_to_fp16_[1](src, dst);
    return;
  }

  lookup_tf32_to_fp16_[2](src, dst);
}

/**
 * @brief Casts f32 tensor to f64.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tf32_to_f64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tf32_to_f64_[0](src, dst);
    return;
  }

  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_tf32_to_f64_[1](src, dst);
    return;
  }

  if (Caps_->sse4_2_) {
    lookup_tf32_to_f64_[2](src, dst);
    return;
  }

  lookup_tf32_to_f64_[3](src, dst);
}

/**
 * @brief Casts f32 tensor to bf16.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F+AVX512BF16+AVX512BW+AVX512VL, Scalar.
 */
static inline void tf32_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_bw_ &&
      Caps_->avx512_vl_) {
    lookup_tf32_to_bf16_[0](src, dst);
    return;
  }
  lookup_tf32_to_bf16_[1](src, dst);
}

/**
 * @brief Casts bf16 tensor to fp16.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void tbf16_to_fp16_(const Tensor *restrict src,
                                  Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_fp16_) {
    lookup_tbf16_to_fp16_[0](src, dst);
    return;
  }
  lookup_tbf16_to_fp16_[1](src, dst);
}

/**
 * @brief Casts bf16 tensor to f32.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_f32_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_tbf16_to_f32_[0](src, dst);
    return;
  }
  lookup_tbf16_to_f32_[1](src, dst);
}

/**
 * @brief Casts bf16 tensor to f64.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_f64_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_tbf16_to_f64_[0](src, dst);
    return;
  }
  lookup_tbf16_to_f64_[1](src, dst);
}

/**
 * @brief Casts f64 tensor to fp16.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void tf64_to_fp16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tf64_to_fp16_[0](src, dst);
    return;
  }
  lookup_tf64_to_fp16_[1](src, dst);
}

/**
 * @brief Casts f64 tensor to f32.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tf64_to_f32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tf64_to_f32_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_tf64_to_f32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tf64_to_f32_[2](src, dst);
    return;
  }
  lookup_tf64_to_f32_[3](src, dst);
}

/**
 * @brief Casts f64 tensor to bf16.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F+AVX512BF16+AVX512VL, Scalar.
 */
static inline void tf64_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_vl_) {
    lookup_tf64_to_bf16_[0](src, dst);
    return;
  }
  lookup_tf64_to_bf16_[1](src, dst);
}

// Cast fp to int

/**
 * @brief Casts fp16 tensor to s8.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void tfp16_to_s8_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_ && Caps_->avx512_bw_) {
    lookup_tfp16_to_s8_[0](src, dst);
    return;
  }
  lookup_tfp16_to_s8_[1](src, dst);
}
/**
 * @brief Casts fp16 tensor to s32.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tfp16_to_s32_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tfp16_to_s32_[0](src, dst);
    return;
  }
  lookup_tfp16_to_s32_[1](src, dst);
}
/**
 * @brief Casts fp16 tensor to s64.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tfp16_to_s64_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tfp16_to_s64_[0](src, dst);
    return;
  }
  lookup_tfp16_to_s64_[1](src, dst);
}
/**
 * @brief Casts fp16 tensor to u8.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void tfp16_to_u8_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_ && Caps_->avx512_bw_) {
    lookup_tfp16_to_u8_[0](src, dst);
    return;
  }
  lookup_tfp16_to_u8_[1](src, dst);
}
/**
 * @brief Casts fp16 tensor to u32.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tfp16_to_u32_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tfp16_to_u32_[0](src, dst);
    return;
  }
  lookup_tfp16_to_u32_[1](src, dst);
}
/**
 * @brief Casts fp16 tensor to u64.
 * @param src Source tensor (fp16).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tfp16_to_u64_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tfp16_to_u64_[0](src, dst);
    return;
  }
  lookup_tfp16_to_u64_[1](src, dst);
}
/**
 * @brief Casts bf16 tensor to s8.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_s8_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512_bf16_ && Caps_->avx512_bw_ && Caps_->avx512f_) {
    lookup_tbf16_to_s8_[0](src, dst);
    return;
  }
  lookup_tbf16_to_s8_[1](src, dst);
}
/**
 * @brief Casts bf16 tensor to s32.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_s32_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_tbf16_to_s32_[0](src, dst);
    return;
  }
  lookup_tbf16_to_s32_[1](src, dst);
}
/**
 * @brief Casts bf16 tensor to s64.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_s64_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_dq_ &&
      Caps_->avx512_vl_) {
    lookup_tbf16_to_s64_[0](src, dst);
    return;
  }
  lookup_tbf16_to_s64_[1](src, dst);
}
/**
 * @brief Casts bf16 tensor to u8.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_u8_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_bw_) {
    lookup_tbf16_to_u8_[0](src, dst);
    return;
  }
  lookup_tbf16_to_u8_[1](src, dst);
}
/**
 * @brief Casts bf16 tensor to u32.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_u32_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_tbf16_to_u32_[0](src, dst);
    return;
  }
  lookup_tbf16_to_u32_[1](src, dst);
}
/**
 * @brief Casts bf16 tensor to u64.
 * @param src Source tensor (bf16).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tbf16_to_u64_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_dq_ &&
      Caps_->avx512_vl_) {
    lookup_tbf16_to_u64_[0](src, dst);
    return;
  }
  lookup_tbf16_to_u64_[1](src, dst);
}
/**
 * @brief Casts f32 tensor to s8.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F, AVX2, Scalar.
 */
static inline void tf32_to_s8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bw_) {
    lookup_tf32_to_s8_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tf32_to_s8_[1](src, dst);
    return;
  }
  lookup_tf32_to_s8_[2](src, dst);
}
/**
 * @brief Casts f32 tensor to s32.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tf32_to_s32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tf32_to_s32_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_tf32_to_s32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tf32_to_s32_[2](src, dst);
    return;
  }
  lookup_tf32_to_s32_[3](src, dst);
}
/**
 * @brief Casts f32 tensor to s64.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf32_to_s64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_tf32_to_s64_[0](src, dst);
    return;
  }
  lookup_tf32_to_s64_[1](src, dst);
}
/**
 * @brief Casts f32 tensor to u8.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, AVX2, Scalar.
 */
static inline void tf32_to_u8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bw_) {
    lookup_tf32_to_u8_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tf32_to_u8_[1](src, dst);
    return;
  }
  lookup_tf32_to_u8_[2](src, dst);
}
/**
 * @brief Casts f32 tensor to u32.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf32_to_u32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tf32_to_u32_[0](src, dst);
    return;
  }
  lookup_tf32_to_u32_[1](src, dst);
}
/**
 * @brief Casts f32 tensor to u64.
 * @param src Source tensor (f32).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf32_to_u64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_tf32_to_u64_[0](src, dst);
    return;
  }
  lookup_tf32_to_u64_[1](src, dst);
}
/**
 * @brief Casts f64 tensor to s8.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F+AVX2, Scalar.
 */
static inline void tf64_to_s8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx2_) {
    lookup_tf64_to_s8_[0](src, dst);
    return;
  }
  lookup_tf64_to_s8_[1](src, dst);
}
/**
 * @brief Casts f64 tensor to s32.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tf64_to_s32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tf64_to_s32_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_tf64_to_s32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tf64_to_s32_[2](src, dst);
    return;
  }
  lookup_tf64_to_s32_[3](src, dst);
}
/**
 * @brief Casts f64 tensor to s64.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F+AVX512DQ, Scalar.
 */
static inline void tf64_to_s64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_tf64_to_s64_[0](src, dst);
    return;
  }
  lookup_tf64_to_s64_[1](src, dst);
}
/**
 * @brief Casts f64 tensor to u8.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F+AVX2, Scalar.
 */
static inline void tf64_to_u8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx2_) {
    lookup_tf64_to_u8_[0](src, dst);
    return;
  }
  lookup_tf64_to_u8_[1](src, dst);
}
/**
 * @brief Casts f64 tensor to u32.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf64_to_u32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tf64_to_u32_[0](src, dst);
    return;
  }
  lookup_tf64_to_u32_[1](src, dst);
}
/**
 * @brief Casts f64 tensor to u64.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf64_to_u64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_tf64_to_u64_[0](src, dst);
    return;
  }
  lookup_tf64_to_u64_[1](src, dst);
}

// Cast int to fp
/**
 * @brief Casts s8 tensor to fp16.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void ts8_to_fp16_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_ts8_to_fp16_[0](src, dst);
    return;
  }
  lookup_ts8_to_fp16_[1](src, dst);
}
/**
 * @brief Casts s32 tensor to fp16.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void ts32_to_fp16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_ts32_to_fp16_[0](src, dst);
    return;
  }
  lookup_ts32_to_fp16_[1](src, dst);
}
/**
 * @brief Casts s64 tensor to fp16.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void ts64_to_fp16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_ts64_to_fp16_[0](src, dst);
    return;
  }
  lookup_ts64_to_fp16_[1](src, dst);
}
/**
 * @brief Casts u8 tensor to fp16.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void tu8_to_fp16_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tu8_to_fp16_[0](src, dst);
    return;
  }
  lookup_tu8_to_fp16_[1](src, dst);
}
/**
 * @brief Casts u32 tensor to fp16.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void tu32_to_fp16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tu32_to_fp16_[0](src, dst);
    return;
  }
  lookup_tu32_to_fp16_[1](src, dst);
}
/**
 * @brief Casts u64 tensor to fp16.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (fp16).
 * @note Variants: AVX512FP16, Scalar.
 */
static inline void tu64_to_fp16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_fp16_) {
    lookup_tu64_to_fp16_[0](src, dst);
    return;
  }
  lookup_tu64_to_fp16_[1](src, dst);
}
/**
 * @brief Casts s8 tensor to bf16.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts8_to_bf16_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_ts8_to_bf16_[0](src, dst);
    return;
  }
  lookup_ts8_to_bf16_[1](src, dst);
}
/**
 * @brief Casts s32 tensor to bf16.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts32_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_ts32_to_bf16_[0](src, dst);
    return;
  }
  lookup_ts32_to_bf16_[1](src, dst);
}
/**
 * @brief Casts s64 tensor to bf16.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts64_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_dq_ &&
      Caps_->avx512_vl_) {
    lookup_ts64_to_bf16_[0](src, dst);
    return;
  }
  lookup_ts64_to_bf16_[1](src, dst);
}
/**
 * @brief Casts u8 tensor to bf16.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu8_to_bf16_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_tu8_to_bf16_[0](src, dst);
    return;
  }
  lookup_tu8_to_bf16_[1](src, dst);
}
/**
 * @brief Casts u32 tensor to bf16.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu32_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
    lookup_tu32_to_bf16_[0](src, dst);
    return;
  }
  lookup_tu32_to_bf16_[1](src, dst);
}
/**
 * @brief Casts u64 tensor to bf16.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (bf16).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu64_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_ && Caps_->avx512_dq_ &&
      Caps_->avx512_vl_) {
    lookup_tu64_to_bf16_[0](src, dst);
    return;
  }
  lookup_tu64_to_bf16_[1](src, dst);
}
/**
 * @brief Casts s8 tensor to f32.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, AVX2, Scalar.
 */
static inline void ts8_to_f32_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts8_to_f32_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_ts8_to_f32_[1](src, dst);
    return;
  }
  lookup_ts8_to_f32_[2](src, dst);
}
/**
 * @brief Casts s32 tensor to f32.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts32_to_f32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts32_to_f32_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_ts32_to_f32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts32_to_f32_[2](src, dst);
    return;
  }
  lookup_ts32_to_f32_[3](src, dst);
}
/**
 * @brief Casts s64 tensor to f32.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts64_to_f32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_ts64_to_f32_[0](src, dst);
    return;
  }
  lookup_ts64_to_f32_[1](src, dst);
}
/**
 * @brief Casts u8 tensor to f32.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, AVX2, Scalar.
 */
static inline void tu8_to_f32_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu8_to_f32_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tu8_to_f32_[1](src, dst);
    return;
  }
  lookup_tu8_to_f32_[2](src, dst);
}
/**
 * @brief Casts u32 tensor to f32.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu32_to_f32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu32_to_f32_[0](src, dst);
    return;
  }
  lookup_tu32_to_f32_[1](src, dst);
}
/**
 * @brief Casts u64 tensor to f32.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (f32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu64_to_f32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_tu64_to_f32_[0](src, dst);
    return;
  }
  lookup_tu64_to_f32_[1](src, dst);
}
/**
 * @brief Casts s8 tensor to f64.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F+AVX2, Scalar.
 */
static inline void ts8_to_f64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx2_) {
    lookup_ts8_to_f64_[0](src, dst);
    return;
  }
  lookup_ts8_to_f64_[1](src, dst);
}
/**
 * @brief Casts s32 tensor to f64.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts32_to_f64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts32_to_f64_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_ts32_to_f64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts32_to_f64_[2](src, dst);
    return;
  }
  lookup_ts32_to_f64_[3](src, dst);
}
/**
 * @brief Casts s64 tensor to f64.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F+AVX512DQ, Scalar.
 */
static inline void ts64_to_f64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_ts64_to_f64_[0](src, dst);
    return;
  }
  lookup_ts64_to_f64_[1](src, dst);
}
/**
 * @brief Casts u8 tensor to f64.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F+AVX2, Scalar.
 */
static inline void tu8_to_f64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx2_) {
    lookup_tu8_to_f64_[0](src, dst);
    return;
  }
  lookup_tu8_to_f64_[1](src, dst);
}
/**
 * @brief Casts u32 tensor to f64.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu32_to_f64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu32_to_f64_[0](src, dst);
    return;
  }
  lookup_tu32_to_f64_[1](src, dst);
}
/**
 * @brief Casts u64 tensor to f64.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F+AVX512DQ, Scalar.
 */
static inline void tu64_to_f64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_dq_) {
    lookup_tu64_to_f64_[0](src, dst);
    return;
  }
  lookup_tu64_to_f64_[1](src, dst);
}

// Cast sint to sint

/**
 * @brief Casts s8 tensor to s32.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts8_to_s32_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts8_to_s32_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_ts8_to_s32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts8_to_s32_[2](src, dst);
    return;
  }
  lookup_ts8_to_s32_[3](src, dst);
}
/**
 * @brief Casts s8 tensor to s64.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts8_to_s64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts8_to_s64_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_ts8_to_s64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts8_to_s64_[2](src, dst);
    return;
  }
  lookup_ts8_to_s64_[3](src, dst);
}
/**
 * @brief Casts s32 tensor to s8.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts32_to_s8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts32_to_s8_[0](src, dst);
    return;
  }
  lookup_ts32_to_s8_[1](src, dst);
}
/**
 * @brief Casts s32 tensor to s64.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts32_to_s64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts32_to_s64_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_ts32_to_s64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts32_to_s64_[2](src, dst);
    return;
  }
  lookup_ts32_to_s64_[3](src, dst);
}
/**
 * @brief Casts s64 tensor to s8.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts64_to_s8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bw_) {
    lookup_ts64_to_s8_[0](src, dst);
    return;
  }
  lookup_ts64_to_s8_[1](src, dst);
}
/**
 * @brief Casts s64 tensor to s32.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts64_to_s32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts64_to_s32_[0](src, dst);
    return;
  }
  lookup_ts64_to_s32_[1](src, dst);
}

// Cast uint to uint

/**
 * @brief Casts u8 tensor to u32.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, AVX2, Scalar.
 */
static inline void tu8_to_u32_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu8_to_u32_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tu8_to_u32_[1](src, dst);
    return;
  }
  lookup_tu8_to_u32_[2](src, dst);
}
/**
 * @brief Casts u8 tensor to u64.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu8_to_u64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu8_to_u64_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tu8_to_u64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu8_to_u64_[2](src, dst);
    return;
  }
  lookup_tu8_to_u64_[3](src, dst);
}
/**
 * @brief Casts u32 tensor to u8.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu32_to_u8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu32_to_u8_[0](src, dst);
    return;
  }
  lookup_tu32_to_u8_[1](src, dst);
}
/**
 * @brief Casts u32 tensor to u64.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu32_to_u64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu32_to_u64_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tu32_to_u64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu32_to_u64_[2](src, dst);
    return;
  }
  lookup_tu32_to_u64_[3](src, dst);
}
/**
 * @brief Casts u64 tensor to u8.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu64_to_u8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bw_) {
    lookup_tu64_to_u8_[0](src, dst);
    return;
  }
  lookup_tu64_to_u8_[1](src, dst);
}
/**
 * @brief Casts u64 tensor to u32.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu64_to_u32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu64_to_u32_[0](src, dst);
    return;
  }
  lookup_tu64_to_u32_[1](src, dst);
}

// Cast sint to uint

/**
 * @brief Casts s8 tensor to u8.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts8_to_u8_(const Tensor *restrict src,
                              Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts8_to_u8_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_ts8_to_u8_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts8_to_u8_[2](src, dst);
    return;
  }
  lookup_ts8_to_u8_[3](src, dst);
}
/**
 * @brief Casts s8 tensor to u32.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts8_to_u32_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts8_to_u32_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_ts8_to_u32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts8_to_u32_[2](src, dst);
    return;
  }
  lookup_ts8_to_u32_[3](src, dst);
}
/**
 * @brief Casts s8 tensor to u64.
 * @param src Source tensor (s8).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts8_to_u64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts8_to_u64_[0](src, dst);
    return;
  }
  lookup_ts8_to_u64_[1](src, dst);
}
/**
 * @brief Casts s32 tensor to u8.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts32_to_u8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bw_) {
    lookup_ts32_to_u8_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_ts32_to_u8_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts32_to_u8_[2](src, dst);
    return;
  }
  lookup_ts32_to_u8_[3](src, dst);
}
/**
 * @brief Casts s32 tensor to u32.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts32_to_u32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts32_to_u32_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_ts32_to_u32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts32_to_u32_[2](src, dst);
    return;
  }
  lookup_ts32_to_u32_[3](src, dst);
}
/**
 * @brief Casts s32 tensor to u64.
 * @param src Source tensor (s32).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts32_to_u64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts32_to_u64_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_ts32_to_u64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_ts32_to_u64_[2](src, dst);
    return;
  }
  lookup_ts32_to_u64_[3](src, dst);
}
/**
 * @brief Casts s64 tensor to u8.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts64_to_u8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts64_to_u8_[0](src, dst);
    return;
  }
  lookup_ts64_to_u8_[1](src, dst);
}
/**
 * @brief Casts s64 tensor to u32.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (u32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts64_to_u32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts64_to_u32_[0](src, dst);
    return;
  }
  lookup_ts64_to_u32_[1](src, dst);
}
/**
 * @brief Casts s64 tensor to u64.
 * @param src Source tensor (s64).
 * @param dst Destination tensor (u64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void ts64_to_u64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts64_to_u64_[0](src, dst);
    return;
  }

  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_ts64_to_u64_[1](src, dst);
    return;
  }

  if (Caps_->sse4_2_) {
    lookup_ts64_to_u64_[2](src, dst);
    return;
  }

  lookup_ts64_to_u64_[3](src, dst);
}

// Cast uint to sint

/**
 * @brief Casts u8 tensor to s8.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu8_to_s8_(const Tensor *restrict src,
                              Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu8_to_s8_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_tu8_to_s8_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu8_to_s8_[2](src, dst);
    return;
  }
  lookup_tu8_to_s8_[3](src, dst);
}
/**
 * @brief Casts u8 tensor to s32.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu8_to_s32_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu8_to_s32_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tu8_to_s32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu8_to_s32_[2](src, dst);
    return;
  }
  lookup_tu8_to_s32_[3](src, dst);
}
/**
 * @brief Casts u8 tensor to s64.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu8_to_s64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu8_to_s64_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tu8_to_s64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu8_to_s64_[2](src, dst);
    return;
  }
  lookup_tu8_to_s64_[3](src, dst);
}
/**
 * @brief Casts u32 tensor to s8.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu32_to_s8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bw_) {
    lookup_tu32_to_s8_[0](src, dst);
    return;
  }
  lookup_tu32_to_s8_[1](src, dst);
}
/**
 * @brief Casts u32 tensor to s32.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu32_to_s32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu32_to_s32_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_tu32_to_s32_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu32_to_s32_[2](src, dst);
    return;
  }
  lookup_tu32_to_s32_[3](src, dst);
}
/**
 * @brief Casts u32 tensor to s64.
 * @param src Source tensor (u32).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu32_to_s64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu32_to_s64_[0](src, dst);
    return;
  }
  if (Caps_->avx2_) {
    lookup_tu32_to_s64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu32_to_s64_[2](src, dst);
    return;
  }
  lookup_tu32_to_s64_[3](src, dst);
}
/**
 * @brief Casts u64 tensor to s8.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (s8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu64_to_s8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu64_to_s8_[0](src, dst);
    return;
  }
  lookup_tu64_to_s8_[1](src, dst);
}
/**
 * @brief Casts u64 tensor to s32.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (s32).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu64_to_s32_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu64_to_s32_[0](src, dst);
    return;
  }
  lookup_tu64_to_s32_[1](src, dst);
}
/**
 * @brief Casts u64 tensor to s64.
 * @param src Source tensor (u64).
 * @param dst Destination tensor (s64).
 * @note Variants: AVX512F, AVX2, SSE4.2, Scalar.
 */
static inline void tu64_to_s64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tu64_to_s64_[0](src, dst);
    return;
  }
  if (Caps_->avx_ || Caps_->avx2_) {
    lookup_tu64_to_s64_[1](src, dst);
    return;
  }
  if (Caps_->sse4_2_) {
    lookup_tu64_to_s64_[2](src, dst);
    return;
  }
  lookup_tu64_to_s64_[3](src, dst);
}
