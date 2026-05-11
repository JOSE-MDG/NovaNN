#pragma once

#include <immintrin.h>
#include <ncore/macros.h>
#include <ncore/simd.h>
#include <ncore/tables/cast_tables.h>
#include <ncore/tensor.h>

#define CAST_FP_TYPES(F)                                                       \
  F(fp16, f32)                                                                 \
  F(fp16, f64)                                                                 \
  F(fp16, bf16)                                                                \
  F(f32, fp16)                                                                 \
  F(f32, f64)                                                                  \
  F(f32, bf16)                                                                 \
  F(bf16, fp16) F(bf16, f32) F(bf16, f64) F(f64, fp16) F(f64, f32) F(f64, bf16)

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

#define CAST_SINT_TO_SINT(F)                                                   \
  F(s8, s32) F(s8, s64) F(s32, s8) F(s32, s64) F(s64, s8) F(s64, s32)

#define CAST_UINT_TO_UINT(F)                                                   \
  F(u8, u32) F(u8, u64) F(u32, u8) F(u32, u64) F(u64, u8) F(u64, u32)

#define CAST_SINT_TO_UINT(F)                                                   \
  F(s8, u8)                                                                    \
  F(s8, u32)                                                                   \
  F(s8, u64)                                                                   \
  F(s32, u8) F(s32, u32) F(s32, u64) F(s64, u8) F(s64, u32) F(s64, u64)

#define CAST_UINT_TO_SINT(F)                                                   \
  F(u8, s8)                                                                    \
  F(u8, s32)                                                                   \
  F(u8, s64)                                                                   \
  F(u32, s8) F(u32, s32) F(u32, s64) F(u64, s8) F(u64, s32) F(u64, s64)

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
 * @note Variants: AVX512F, F16C, Scalar.
 */
static inline void tfp16_to_f64_(const Tensor *restrict src,
                                 Tensor *restrict dst) {

  if (Caps_->avx512f_) {
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
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf32_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
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
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf64_to_bf16_(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  if (Caps_->avx512f_ && Caps_->avx512_bf16_) {
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
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf64_to_s8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
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
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf64_to_s64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_tf64_to_s64_[0](src, dst);
    return;
  }
  lookup_tf64_to_s64_[1](src, dst);
}
/**
 * @brief Casts f64 tensor to u8.
 * @param src Source tensor (f64).
 * @param dst Destination tensor (u8).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tf64_to_u8_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
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
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts8_to_f64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
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
 * @note Variants: AVX512F, Scalar.
 */
static inline void ts64_to_f64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
    lookup_ts64_to_f64_[0](src, dst);
    return;
  }
  lookup_ts64_to_f64_[1](src, dst);
}
/**
 * @brief Casts u8 tensor to f64.
 * @param src Source tensor (u8).
 * @param dst Destination tensor (f64).
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu8_to_f64_(const Tensor *restrict src,
                               Tensor *restrict dst) {
  if (Caps_->avx512f_) {
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
 * @note Variants: AVX512F, Scalar.
 */
static inline void tu64_to_f64_(const Tensor *restrict src,
                                Tensor *restrict dst) {
  if (Caps_->avx512f_) {
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
