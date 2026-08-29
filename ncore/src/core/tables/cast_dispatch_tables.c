/**
 * @file cast_dispatch_tables.c
 * @brief Initialisation of the global cast-function dispatch table.
 *
 * @details
 * Populates the NUM_DTYPES×NUM_DTYPES @c cast_dispatch matrix that maps every
 * pair of @ref DType_ values (source → target) to a type-specific cast kernel.
 * The table is populated once at load time via an
 * @c INITIALIZE(init_cast_dispatch) function.
 *
 * The row index (@c src) is the source @ref DType_ and the column
 * index (@c dst) is the target @ref DType_.  A @c nullptr entry
 * (including the diagonal) means the cast is either unnecessary
 * (same type) or not supported.
 *
 * @section constructor-ordering Constructor ordering
 *
 * @c INITIALIZE(init_cast_dispatch) runs at program load time.  Because
 * this file only writes to the @c cast_dispatch array (no other
 * globals depend on it), there are no inter-file ordering
 * constraints.
 *
 * @see cast_dispatch   The NUM_DTYPES×NUM_DTYPES dispatch matrix.
 * @see castFn          Function pointer type for cast kernels.
 * @see cast.h          X-macro definitions that generate the
 *                      individual cast kernel declarations.
 * @see dtype.h         @ref DType_ enum and @ref NUM_DTYPES.
 */

#include <ncore/core/dtype.h>
#include <ncore/headeronly/cast.h>
#include <ncore/headeronly/macros.h>

/**
 * @var cast_dispatch
 * @brief NUM_DTYPES×NUM_DTYPES dispatch table mapping (src, dst) @ref DType_
 * pairs to cast function pointers.
 *
 * @details
 * Indexed as @c cast_dispatch[src][dst] where @c src is the source
 * data type and @c dst is the target data type.  The diagonal and
 * unsupported combinations are @c nullptr.
 *
 * The table is zero-initialised (@c = {}) and fully populated
 * by @ref init_cast_dispatch() at program load time.
 *
 * @see init_cast_dispatch()  Populates the table.
 * @see castFn                Function pointer type.
 */
CastFn cast_dispatch[NUM_DTYPES][NUM_DTYPES] = {};

/**
 * @brief Populate every non-trivial entry in @c cast_dispatch.
 *
 * @details
 * Called automatically at library initialisation time via
 * @c INITIALIZE(init_cast_dispatch).  Assigns a @c castFn pointer to
 * each supported (src, dst) pair.  Entries left as @c nullptr are:
 * @li Diagonal (@c src == dst): identity cast, no function needed.
 * @li Quantised types: no explicit entries.
 *
 * @subsection organisation Organisation
 *
 * The assignments are grouped by conversion category for
 * readability:
 *
 * @li Float ↔ Float
 * @li Float → Integer
 * @li Integer → Float
 * @li Signed ↔ Signed
 * @li Unsigned ↔ Unsigned
 * @li Signed → Unsigned
 * @li Unsigned → Signed
 *
 * @pre  The @c cast.h header must have been included so that all
 *       @c tf32_to_*, @c ts8_to_*, etc. symbols are declared.
 * @post All 210 non-trivial entries of @c cast_dispatch are set to
 *       valid function pointers.
 *
 * @see cast_dispatch   The table being populated.
 * @see castFn          Function pointer type.
 */
INITIALIZE(init_cast_dispatch) {
  /* Floating-point to floating-point */
  cast_dispatch[Float4E2M1fn][Float8E4M3fn] = tfp4e2m1_to_fp8e4m3;
  cast_dispatch[Float4E2M1fn][Float8E5M2] = tfp4e2m1_to_fp8e5m2;
  cast_dispatch[Float4E2M1fn][Float16] = tfp4e2m1_to_fp16;
  cast_dispatch[Float4E2M1fn][BFloat16] = tfp4e2m1_to_bf16;
  cast_dispatch[Float4E2M1fn][Float32] = tfp4e2m1_to_f32;
  cast_dispatch[Float4E2M1fn][Float64] = tfp4e2m1_to_f64;

  cast_dispatch[Float8E4M3fn][Float4E2M1fn] = tfp8e4m3_to_fp4e2m1;
  cast_dispatch[Float8E4M3fn][Float8E5M2] = tfp8e4m3_to_fp8e5m2;
  cast_dispatch[Float8E4M3fn][Float16] = tfp8e4m3_to_fp16;
  cast_dispatch[Float8E4M3fn][BFloat16] = tfp8e4m3_to_bf16;
  cast_dispatch[Float8E4M3fn][Float32] = tfp8e4m3_to_f32;
  cast_dispatch[Float8E4M3fn][Float64] = tfp8e4m3_to_f64;

  cast_dispatch[Float8E5M2][Float4E2M1fn] = tfp8e5m2_to_fp4e2m1;
  cast_dispatch[Float8E5M2][Float8E4M3fn] = tfp8e5m2_to_fp8e4m3;
  cast_dispatch[Float8E5M2][Float16] = tfp8e5m2_to_fp16;
  cast_dispatch[Float8E5M2][BFloat16] = tfp8e5m2_to_bf16;
  cast_dispatch[Float8E5M2][Float32] = tfp8e5m2_to_f32;
  cast_dispatch[Float8E5M2][Float64] = tfp8e5m2_to_f64;

  cast_dispatch[Float16][Float4E2M1fn] = tfp16_to_fp4e2m1;
  cast_dispatch[Float16][Float8E4M3fn] = tfp16_to_fp8e4m3;
  cast_dispatch[Float16][Float8E5M2] = tfp16_to_fp8e5m2;
  cast_dispatch[Float16][BFloat16] = tfp16_to_bf16;
  cast_dispatch[Float16][Float32] = tfp16_to_f32;
  cast_dispatch[Float16][Float64] = tfp16_to_f64;

  cast_dispatch[Float32][Float4E2M1fn] = tf32_to_fp4e2m1;
  cast_dispatch[Float32][Float8E4M3fn] = tf32_to_fp8e4m3;
  cast_dispatch[Float32][Float8E5M2] = tf32_to_fp8e5m2;
  cast_dispatch[Float32][Float16] = tf32_to_fp16;
  cast_dispatch[Float32][BFloat16] = tf32_to_bf16;
  cast_dispatch[Float32][Float64] = tf32_to_f64;

  cast_dispatch[BFloat16][Float4E2M1fn] = tbf16_to_fp4e2m1;
  cast_dispatch[BFloat16][Float8E4M3fn] = tbf16_to_fp8e4m3;
  cast_dispatch[BFloat16][Float8E5M2] = tbf16_to_fp8e5m2;
  cast_dispatch[BFloat16][Float16] = tbf16_to_fp16;
  cast_dispatch[BFloat16][Float32] = tbf16_to_f32;
  cast_dispatch[BFloat16][Float64] = tbf16_to_f64;

  cast_dispatch[Float64][Float4E2M1fn] = tf64_to_fp4e2m1;
  cast_dispatch[Float64][Float8E4M3fn] = tf64_to_fp8e4m3;
  cast_dispatch[Float64][Float8E5M2] = tf64_to_fp8e5m2;
  cast_dispatch[Float64][Float16] = tf64_to_fp16;
  cast_dispatch[Float64][BFloat16] = tf64_to_bf16;
  cast_dispatch[Float64][Float32] = tf64_to_f32;

  /* Floating-point to integer */

  cast_dispatch[Float4E2M1fn][Signed8] = tfp4e2m1_to_s8;
  cast_dispatch[Float4E2M1fn][Signed16] = tfp4e2m1_to_s16;
  cast_dispatch[Float4E2M1fn][Signed32] = tfp4e2m1_to_s32;
  cast_dispatch[Float4E2M1fn][Signed64] = tfp4e2m1_to_s64;
  cast_dispatch[Float4E2M1fn][UnSigned8] = tfp4e2m1_to_u8;
  cast_dispatch[Float4E2M1fn][UnSigned16] = tfp4e2m1_to_u16;
  cast_dispatch[Float4E2M1fn][UnSigned32] = tfp4e2m1_to_u32;
  cast_dispatch[Float4E2M1fn][UnSigned64] = tfp4e2m1_to_u64;

  cast_dispatch[Float8E4M3fn][Signed8] = tfp8e4m3_to_s8;
  cast_dispatch[Float8E4M3fn][Signed16] = tfp8e4m3_to_s16;
  cast_dispatch[Float8E4M3fn][Signed32] = tfp8e4m3_to_s32;
  cast_dispatch[Float8E4M3fn][Signed64] = tfp8e4m3_to_s64;
  cast_dispatch[Float8E4M3fn][UnSigned8] = tfp8e4m3_to_u8;
  cast_dispatch[Float8E4M3fn][UnSigned16] = tfp8e4m3_to_u16;
  cast_dispatch[Float8E4M3fn][UnSigned32] = tfp8e4m3_to_u32;
  cast_dispatch[Float8E4M3fn][UnSigned64] = tfp8e4m3_to_u64;

  cast_dispatch[Float8E5M2][Signed8] = tfp8e5m2_to_s8;
  cast_dispatch[Float8E5M2][Signed16] = tfp8e5m2_to_s16;
  cast_dispatch[Float8E5M2][Signed32] = tfp8e5m2_to_s32;
  cast_dispatch[Float8E5M2][Signed64] = tfp8e5m2_to_s64;
  cast_dispatch[Float8E5M2][UnSigned8] = tfp8e5m2_to_u8;
  cast_dispatch[Float8E5M2][UnSigned16] = tfp8e5m2_to_u16;
  cast_dispatch[Float8E5M2][UnSigned32] = tfp8e5m2_to_u32;
  cast_dispatch[Float8E5M2][UnSigned64] = tfp8e5m2_to_u64;

  cast_dispatch[Float16][Signed8] = tfp16_to_s8;
  cast_dispatch[Float16][Signed16] = tfp16_to_s16;
  cast_dispatch[Float16][Signed32] = tfp16_to_s32;
  cast_dispatch[Float16][Signed64] = tfp16_to_s64;
  cast_dispatch[Float16][UnSigned8] = tfp16_to_u8;
  cast_dispatch[Float16][UnSigned16] = tfp16_to_u16;
  cast_dispatch[Float16][UnSigned32] = tfp16_to_u32;
  cast_dispatch[Float16][UnSigned64] = tfp16_to_u64;

  cast_dispatch[BFloat16][Signed8] = tbf16_to_s8;
  cast_dispatch[BFloat16][Signed16] = tbf16_to_s16;
  cast_dispatch[BFloat16][Signed32] = tbf16_to_s32;
  cast_dispatch[BFloat16][Signed64] = tbf16_to_s64;
  cast_dispatch[BFloat16][UnSigned8] = tbf16_to_u8;
  cast_dispatch[BFloat16][UnSigned16] = tbf16_to_u16;
  cast_dispatch[BFloat16][UnSigned32] = tbf16_to_u32;
  cast_dispatch[BFloat16][UnSigned64] = tbf16_to_u64;

  cast_dispatch[Float32][Signed8] = tf32_to_s8;
  cast_dispatch[Float32][Signed16] = tf32_to_s16;
  cast_dispatch[Float32][Signed32] = tf32_to_s32;
  cast_dispatch[Float32][Signed64] = tf32_to_s64;
  cast_dispatch[Float32][UnSigned8] = tf32_to_u8;
  cast_dispatch[Float32][UnSigned16] = tf32_to_u16;
  cast_dispatch[Float32][UnSigned32] = tf32_to_u32;
  cast_dispatch[Float32][UnSigned64] = tf32_to_u64;

  cast_dispatch[Float64][Signed8] = tf64_to_s8;
  cast_dispatch[Float64][Signed16] = tf64_to_s16;
  cast_dispatch[Float64][Signed32] = tf64_to_s32;
  cast_dispatch[Float64][Signed64] = tf64_to_s64;
  cast_dispatch[Float64][UnSigned8] = tf64_to_u8;
  cast_dispatch[Float64][UnSigned16] = tf64_to_u16;
  cast_dispatch[Float64][UnSigned32] = tf64_to_u32;
  cast_dispatch[Float64][UnSigned64] = tf64_to_u64;

  /* Integer to floating-point */
  cast_dispatch[Signed8][Float4E2M1fn] = ts8_to_fp4e2m1;
  cast_dispatch[Signed8][Float8E4M3fn] = ts8_to_fp8e4m3;
  cast_dispatch[Signed8][Float8E5M2] = ts8_to_fp8e5m2;
  cast_dispatch[Signed8][Float16] = ts8_to_fp16;
  cast_dispatch[Signed8][BFloat16] = ts8_to_bf16;
  cast_dispatch[Signed8][Float32] = ts8_to_f32;
  cast_dispatch[Signed8][Float64] = ts8_to_f64;

  cast_dispatch[Signed16][Float4E2M1fn] = ts16_to_fp4e2m1;
  cast_dispatch[Signed16][Float8E4M3fn] = ts16_to_fp8e4m3;
  cast_dispatch[Signed16][Float8E5M2] = ts16_to_fp8e5m2;
  cast_dispatch[Signed16][Float16] = ts16_to_fp16;
  cast_dispatch[Signed16][BFloat16] = ts16_to_bf16;
  cast_dispatch[Signed16][Float32] = ts16_to_f32;
  cast_dispatch[Signed16][Float64] = ts16_to_f64;

  cast_dispatch[Signed32][Float4E2M1fn] = ts32_to_fp4e2m1;
  cast_dispatch[Signed32][Float8E4M3fn] = ts32_to_fp8e4m3;
  cast_dispatch[Signed32][Float8E5M2] = ts32_to_fp8e5m2;
  cast_dispatch[Signed32][Float16] = ts32_to_fp16;
  cast_dispatch[Signed32][BFloat16] = ts32_to_bf16;
  cast_dispatch[Signed32][Float32] = ts32_to_f32;
  cast_dispatch[Signed32][Float64] = ts32_to_f64;

  cast_dispatch[Signed64][Float4E2M1fn] = ts64_to_fp4e2m1;
  cast_dispatch[Signed64][Float8E4M3fn] = ts64_to_fp8e4m3;
  cast_dispatch[Signed64][Float8E5M2] = ts64_to_fp8e5m2;
  cast_dispatch[Signed64][Float16] = ts64_to_fp16;
  cast_dispatch[Signed64][BFloat16] = ts64_to_bf16;
  cast_dispatch[Signed64][Float32] = ts64_to_f32;
  cast_dispatch[Signed64][Float64] = ts64_to_f64;

  cast_dispatch[UnSigned8][Float4E2M1fn] = tu8_to_fp4e2m1;
  cast_dispatch[UnSigned8][Float8E4M3fn] = tu8_to_fp8e4m3;
  cast_dispatch[UnSigned8][Float8E5M2] = tu8_to_fp8e5m2;
  cast_dispatch[UnSigned8][Float16] = tu8_to_fp16;
  cast_dispatch[UnSigned8][BFloat16] = tu8_to_bf16;
  cast_dispatch[UnSigned8][Float32] = tu8_to_f32;
  cast_dispatch[UnSigned8][Float64] = tu8_to_f64;

  cast_dispatch[UnSigned16][Float4E2M1fn] = tu16_to_fp4e2m1;
  cast_dispatch[UnSigned16][Float8E4M3fn] = tu16_to_fp8e4m3;
  cast_dispatch[UnSigned16][Float8E5M2] = tu16_to_fp8e5m2;
  cast_dispatch[UnSigned16][Float16] = tu16_to_fp16;
  cast_dispatch[UnSigned16][BFloat16] = tu16_to_bf16;
  cast_dispatch[UnSigned16][Float32] = tu16_to_f32;
  cast_dispatch[UnSigned16][Float64] = tu16_to_f64;

  cast_dispatch[UnSigned32][Float4E2M1fn] = tu32_to_fp4e2m1;
  cast_dispatch[UnSigned32][Float8E4M3fn] = tu32_to_fp8e4m3;
  cast_dispatch[UnSigned32][Float8E5M2] = tu32_to_fp8e5m2;
  cast_dispatch[UnSigned32][Float16] = tu32_to_fp16;
  cast_dispatch[UnSigned32][BFloat16] = tu32_to_bf16;
  cast_dispatch[UnSigned32][Float32] = tu32_to_f32;
  cast_dispatch[UnSigned32][Float64] = tu32_to_f64;

  cast_dispatch[UnSigned64][Float4E2M1fn] = tu64_to_fp4e2m1;
  cast_dispatch[UnSigned64][Float8E4M3fn] = tu64_to_fp8e4m3;
  cast_dispatch[UnSigned64][Float8E5M2] = tu64_to_fp8e5m2;
  cast_dispatch[UnSigned64][Float16] = tu64_to_fp16;
  cast_dispatch[UnSigned64][BFloat16] = tu64_to_bf16;
  cast_dispatch[UnSigned64][Float32] = tu64_to_f32;
  cast_dispatch[UnSigned64][Float64] = tu64_to_f64;

  /* Signed integer to signed integer */
  cast_dispatch[Signed8][Signed16] = ts8_to_s16;
  cast_dispatch[Signed8][Signed32] = ts8_to_s32;
  cast_dispatch[Signed8][Signed64] = ts8_to_s64;
  cast_dispatch[Signed16][Signed8] = ts16_to_s8;
  cast_dispatch[Signed16][Signed32] = ts16_to_s32;
  cast_dispatch[Signed16][Signed64] = ts16_to_s64;
  cast_dispatch[Signed32][Signed8] = ts32_to_s8;
  cast_dispatch[Signed32][Signed16] = ts32_to_s16;
  cast_dispatch[Signed32][Signed64] = ts32_to_s64;
  cast_dispatch[Signed64][Signed8] = ts64_to_s8;
  cast_dispatch[Signed64][Signed16] = ts64_to_s16;
  cast_dispatch[Signed64][Signed32] = ts64_to_s32;

  /* Unsigned integer to unsigned integer */
  cast_dispatch[UnSigned8][UnSigned16] = tu8_to_u16;
  cast_dispatch[UnSigned8][UnSigned32] = tu8_to_u32;
  cast_dispatch[UnSigned8][UnSigned64] = tu8_to_u64;
  cast_dispatch[UnSigned16][UnSigned8] = tu16_to_u8;
  cast_dispatch[UnSigned16][UnSigned32] = tu16_to_u32;
  cast_dispatch[UnSigned16][UnSigned64] = tu16_to_u64;
  cast_dispatch[UnSigned32][UnSigned8] = tu32_to_u8;
  cast_dispatch[UnSigned32][UnSigned16] = tu32_to_u16;
  cast_dispatch[UnSigned32][UnSigned64] = tu32_to_u64;
  cast_dispatch[UnSigned64][UnSigned8] = tu64_to_u8;
  cast_dispatch[UnSigned64][UnSigned16] = tu64_to_u16;
  cast_dispatch[UnSigned64][UnSigned32] = tu64_to_u32;

  /* Signed integer to unsigned integer */
  cast_dispatch[Signed8][UnSigned8] = ts8_to_u8;
  cast_dispatch[Signed8][UnSigned16] = ts8_to_u16;
  cast_dispatch[Signed8][UnSigned32] = ts8_to_u32;
  cast_dispatch[Signed8][UnSigned64] = ts8_to_u64;

  cast_dispatch[Signed16][UnSigned8] = ts16_to_u8;
  cast_dispatch[Signed16][UnSigned16] = ts16_to_u16;
  cast_dispatch[Signed16][UnSigned32] = ts16_to_u32;
  cast_dispatch[Signed16][UnSigned64] = ts16_to_u64;

  cast_dispatch[Signed32][UnSigned8] = ts32_to_u8;
  cast_dispatch[Signed32][UnSigned16] = ts32_to_u16;
  cast_dispatch[Signed32][UnSigned32] = ts32_to_u32;
  cast_dispatch[Signed32][UnSigned64] = ts32_to_u64;

  cast_dispatch[Signed64][UnSigned8] = ts64_to_u8;
  cast_dispatch[Signed64][UnSigned16] = ts64_to_u16;
  cast_dispatch[Signed64][UnSigned32] = ts64_to_u32;
  cast_dispatch[Signed64][UnSigned64] = ts64_to_u64;

  /* Unsigned integer to signed integer */
  cast_dispatch[UnSigned8][Signed8] = tu8_to_s8;
  cast_dispatch[UnSigned8][Signed16] = tu8_to_s16;
  cast_dispatch[UnSigned8][Signed32] = tu8_to_s32;
  cast_dispatch[UnSigned8][Signed64] = tu8_to_s64;

  cast_dispatch[UnSigned16][Signed8] = tu16_to_s8;
  cast_dispatch[UnSigned16][Signed16] = tu16_to_s16;
  cast_dispatch[UnSigned16][Signed32] = tu16_to_s32;
  cast_dispatch[UnSigned16][Signed64] = tu16_to_s64;

  cast_dispatch[UnSigned32][Signed8] = tu32_to_s8;
  cast_dispatch[UnSigned32][Signed16] = tu32_to_s16;
  cast_dispatch[UnSigned32][Signed32] = tu32_to_s32;
  cast_dispatch[UnSigned32][Signed64] = tu32_to_s64;

  cast_dispatch[UnSigned64][Signed8] = tu64_to_s8;
  cast_dispatch[UnSigned64][Signed16] = tu64_to_s16;
  cast_dispatch[UnSigned64][Signed32] = tu64_to_s32;
  cast_dispatch[UnSigned64][Signed64] = tu64_to_s64;
}
