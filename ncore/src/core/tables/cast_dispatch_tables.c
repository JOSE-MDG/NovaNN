/**
 * @file cast_dispatch_tables.c
 * @brief Initialisation of the global cast-function dispatch table.
 *
 * @details
 * Populates the 12×12 `cast_dispatch` matrix that maps every pair
 * of @ref DType_ values (source → target) to a type-specific cast
 * kernel.  The table is populated once at load time via a
 * `__attribute__((constructor))` function.
 *
 * ## Table layout
 *
 * ```
 * cast_dispatch[src][dst] → castFn
 * ```
 *
 * The row index (`src`) is the source @ref DType_ and the column
 * index (`dst`) is the target @ref DType_.  A `NULL` entry
 * (including the diagonal) means the cast is either unnecessary
 * (same type) or not supported.
 *
 * ## Conversion categories
 *
 * The 132 non-null entries are organised into six groups:
 */
// clang-format off
/**
 * | Category                | Count | Description                           |
 * |-------------------------|------:|---------------------------------------|
 * | Float ↔ Float           |    12 | Lossless/lossy floating-point casts   |
 * | Float → Integer         |    24 | Truncation / rounding to integer      |
 * | Integer → Float         |    24 | Widening / narrowing to float         |
 * | Signed ↔ Signed         |     6 | Width-changing signed integer casts   |
 * | Unsigned ↔ Unsigned     |     6 | Width-changing unsigned integer casts |
 * | Signed ↔ Unsigned       |    36 | Sign-change + width-change combos     |
 */
// clang-format on
/**
 * Quantised types (`QSigned8`, `QUnSigned8`) share their native
 * storage type (`int8_t` / `uint8_t`) with `Signed8` / `UnSigned8`
 * and are therefore covered by the same cast kernels — no separate
 * quantised entries exist in this table.
 *
 * ## Constructor ordering
 *
 * `__attribute__((constructor))` runs before `main()`.  Because
 * this file only writes to the `cast_dispatch` array (no other
 * globals depend on it), there are no inter-file ordering
 * constraints.
 *
 * @see cast_dispatch   The 12×12 dispatch matrix.
 * @see castFn          Function pointer type for cast kernels.
 * @see cast.h          X-macro definitions that generate the
 *                      individual cast kernel declarations.
 * @see dtype.h         @ref DType_ enum and @ref NUM_DTYPES.
 */

#include <ncore/headeronly/cast.h>
#include <ncore/macros.h>

/**
 * @var cast_dispatch
 * @brief 12×12 dispatch table mapping (src, dst) @ref DType_ pairs
 *        to cast function pointers.
 *
 * @details
 * Indexed as `cast_dispatch[src][dst]` where `src` is the source
 * data type and `dst` is the target data type.  The diagonal and
 * unsupported combinations are `NULL`.
 *
 * The table is zero-initialised (`= {NULL}`) and fully populated
 * by @ref init_cast_dispatch() at program load time.
 *
 * @see init_cast_dispatch()  Populates the table.
 * @see castFn                Function pointer type.
 */
castFn cast_dispatch[NUM_DTYPES][NUM_DTYPES] = {{NULL}};

/**
 * @brief Populate every non-trivial entry in `cast_dispatch`.
 *
 * @details
 * Called automatically before `main()` via
 * `__attribute__((constructor))`.  Assigns a `castFn` pointer to
 * each supported (src, dst) pair.  Entries left as `NULL` are:
 * - **Diagonal** (`src == dst`): identity cast, no function needed.
 * - **Unsupported**: e.g.,    (handled by the
 *   same native-type kernels).
 *
 * ### Organisation
 *
 * The assignments are grouped by conversion category for
 * readability:
 *
 * 1. **Float ↔ Float** (lines 16–30)
 * 2. **Float → Integer** (lines 33–59)
 * 3. **Integer → Float** (lines 62–90)
 * 4. **Signed ↔ Signed** (lines 93–98)
 * 5. **Unsigned ↔ Unsigned** (lines 101–106)
 * 6. **Signed ↔ Unsigned** (lines 109–132)
 *
 * @pre  The `cast.h` header must have been included so that all
 *       `tf32_to_*`, `ts8_to_*`, etc. symbols are declared.
 * @post All 132 non-trivial entries of `cast_dispatch` are set to
 *       valid function pointers.
 *
 * @see cast_dispatch   The table being populated.
 * @see castFn          Function pointer type.
 */
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
