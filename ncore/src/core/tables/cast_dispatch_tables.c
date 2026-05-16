/**
 * @file cast_dispatch_tables.c
 * @brief Cast function dispatch table initialization.
 *
 * Initializes the global 2D dispatch table that maps source and target
 * data types to their corresponding cast function implementations.
 */

#include <ncore/headeronly/cast.h>
#include <ncore/macros.h>

castFn cast_dispatch[NUM_DTYPES][NUM_DTYPES] = {NULL};

__attribute__((constructor)) static inline void init_cast_dispatch() {
  /* Floating-point to floating-point */
  cast_dispatch[Float16][Float32] = tfp16_to_f32_;
  cast_dispatch[Float16][Float64] = tfp16_to_f64_;
  cast_dispatch[Float16][BFloat16] = tfp16_to_bf16_;

  cast_dispatch[Float32][Float16] = tf32_to_fp16_;
  cast_dispatch[Float32][Float64] = tf32_to_f64_;
  cast_dispatch[Float32][BFloat16] = tf32_to_bf16_;

  cast_dispatch[BFloat16][Float16] = tbf16_to_fp16_;
  cast_dispatch[BFloat16][Float32] = tbf16_to_f32_;
  cast_dispatch[BFloat16][Float64] = tbf16_to_f64_;

  cast_dispatch[Float64][Float16] = tf64_to_fp16_;
  cast_dispatch[Float64][Float32] = tf64_to_f32_;
  cast_dispatch[Float64][BFloat16] = tf64_to_bf16_;

  /* Floating-point to integer */
  cast_dispatch[Float16][Signed8] = tfp16_to_s8_;
  cast_dispatch[Float16][Signed32] = tfp16_to_s32_;
  cast_dispatch[Float16][Signed64] = tfp16_to_s64_;
  cast_dispatch[Float16][UnSigned8] = tfp16_to_u8_;
  cast_dispatch[Float16][UnSigned32] = tfp16_to_u32_;
  cast_dispatch[Float16][UnSigned64] = tfp16_to_u64_;

  cast_dispatch[BFloat16][Signed8] = tbf16_to_s8_;
  cast_dispatch[BFloat16][Signed32] = tbf16_to_s32_;
  cast_dispatch[BFloat16][Signed64] = tbf16_to_s64_;
  cast_dispatch[BFloat16][UnSigned8] = tbf16_to_u8_;
  cast_dispatch[BFloat16][UnSigned32] = tbf16_to_u32_;
  cast_dispatch[BFloat16][UnSigned64] = tbf16_to_u64_;

  cast_dispatch[Float32][Signed8] = tf32_to_s8_;
  cast_dispatch[Float32][Signed32] = tf32_to_s32_;
  cast_dispatch[Float32][Signed64] = tf32_to_s64_;
  cast_dispatch[Float32][UnSigned8] = tf32_to_u8_;
  cast_dispatch[Float32][UnSigned32] = tf32_to_u32_;
  cast_dispatch[Float32][UnSigned64] = tf32_to_u64_;

  cast_dispatch[Float64][Signed8] = tf64_to_s8_;
  cast_dispatch[Float64][Signed32] = tf64_to_s32_;
  cast_dispatch[Float64][Signed64] = tf64_to_s64_;
  cast_dispatch[Float64][UnSigned8] = tf64_to_u8_;
  cast_dispatch[Float64][UnSigned32] = tf64_to_u32_;
  cast_dispatch[Float64][UnSigned64] = tf64_to_u64_;

  /* Integer to floating-point */
  cast_dispatch[Signed8][Float16] = ts8_to_fp16_;
  cast_dispatch[Signed8][BFloat16] = ts8_to_bf16_;
  cast_dispatch[Signed8][Float32] = ts8_to_f32_;
  cast_dispatch[Signed8][Float64] = ts8_to_f64_;

  cast_dispatch[Signed32][Float16] = ts32_to_fp16_;
  cast_dispatch[Signed32][BFloat16] = ts32_to_bf16_;
  cast_dispatch[Signed32][Float32] = ts32_to_f32_;
  cast_dispatch[Signed32][Float64] = ts32_to_f64_;

  cast_dispatch[Signed64][Float16] = ts64_to_fp16_;
  cast_dispatch[Signed64][BFloat16] = ts64_to_bf16_;
  cast_dispatch[Signed64][Float32] = ts64_to_f32_;
  cast_dispatch[Signed64][Float64] = ts64_to_f64_;

  cast_dispatch[UnSigned8][Float16] = tu8_to_fp16_;
  cast_dispatch[UnSigned8][BFloat16] = tu8_to_bf16_;
  cast_dispatch[UnSigned8][Float32] = tu8_to_f32_;
  cast_dispatch[UnSigned8][Float64] = tu8_to_f64_;

  cast_dispatch[UnSigned32][Float16] = tu32_to_fp16_;
  cast_dispatch[UnSigned32][BFloat16] = tu32_to_bf16_;
  cast_dispatch[UnSigned32][Float32] = tu32_to_f32_;
  cast_dispatch[UnSigned32][Float64] = tu32_to_f64_;

  cast_dispatch[UnSigned64][Float16] = tu64_to_fp16_;
  cast_dispatch[UnSigned64][BFloat16] = tu64_to_bf16_;
  cast_dispatch[UnSigned64][Float32] = tu64_to_f32_;
  cast_dispatch[UnSigned64][Float64] = tu64_to_f64_;

  /* Signed integer to signed integer */
  cast_dispatch[Signed8][Signed32] = ts8_to_s32_;
  cast_dispatch[Signed8][Signed64] = ts8_to_s64_;
  cast_dispatch[Signed32][Signed8] = ts32_to_s8_;
  cast_dispatch[Signed32][Signed64] = ts32_to_s64_;
  cast_dispatch[Signed64][Signed8] = ts64_to_s8_;
  cast_dispatch[Signed64][Signed32] = ts64_to_s32_;

  /* Unsigned integer to unsigned integer */
  cast_dispatch[UnSigned8][UnSigned32] = tu8_to_u32_;
  cast_dispatch[UnSigned8][UnSigned64] = tu8_to_u64_;
  cast_dispatch[UnSigned32][UnSigned8] = tu32_to_u8_;
  cast_dispatch[UnSigned32][UnSigned64] = tu32_to_u64_;
  cast_dispatch[UnSigned64][UnSigned8] = tu64_to_u8_;
  cast_dispatch[UnSigned64][UnSigned32] = tu64_to_u32_;

  /* Signed integer to unsigned integer */
  cast_dispatch[Signed8][UnSigned8] = ts8_to_u8_;
  cast_dispatch[Signed8][UnSigned32] = ts8_to_u32_;
  cast_dispatch[Signed8][UnSigned64] = ts8_to_u64_;

  cast_dispatch[Signed32][UnSigned8] = ts32_to_u8_;
  cast_dispatch[Signed32][UnSigned32] = ts32_to_u32_;
  cast_dispatch[Signed32][UnSigned64] = ts32_to_u64_;

  cast_dispatch[Signed64][UnSigned8] = ts64_to_u8_;
  cast_dispatch[Signed64][UnSigned32] = ts64_to_u32_;
  cast_dispatch[Signed64][UnSigned64] = ts64_to_u64_;

  /* Unsigned integer to signed integer */
  cast_dispatch[UnSigned8][Signed8] = tu8_to_s8_;
  cast_dispatch[UnSigned8][Signed32] = tu8_to_s32_;
  cast_dispatch[UnSigned8][Signed64] = tu8_to_s64_;

  cast_dispatch[UnSigned32][Signed8] = tu32_to_s8_;
  cast_dispatch[UnSigned32][Signed32] = tu32_to_s32_;
  cast_dispatch[UnSigned32][Signed64] = tu32_to_s64_;

  cast_dispatch[UnSigned64][Signed8] = tu64_to_s8_;
  cast_dispatch[UnSigned64][Signed32] = tu64_to_s32_;
  cast_dispatch[UnSigned64][Signed64] = tu64_to_s64_;
}
