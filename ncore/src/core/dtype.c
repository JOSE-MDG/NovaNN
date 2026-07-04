/**
 * @file dtype.c
 * @brief Implementation of dtype classification and casting functions.
 *
 * @details
 * This file provides functions to check the properties of a tensor's
 * data type using lookup tables defined in @ref dtype_tables.h, as
 * well as a generic cast function that dispatches to the appropriate
 * cast implementation based on source and target dtypes.
 *
 * ## Classification
 *
 * The six classification functions (`is_floating`, `is_integer`,
 * etc.) each index into a precomputed `uint8_t` lookup table
 * (e.g., `floating[NUM_DTYPES][1]`) and cast the result to `bool`.
 * The tables are populated at compile time from the @ref DType_
 * enumeration.  This avoids switch statements and gives O(1)
 * classification for any dtype.
 *
 * ## Cast Dispatch
 *
 * The `cast()` function resolves the correct element-wise
 * conversion via the `cast_dispatch` 2D array (indexed by
 * `[source_dtype][target_dtype]`) and calls the matching
 * `castFn` implementation.  The actual cast kernels are defined
 * in the `cast_dispatch_tables` and `cast_tables` translation
 * units.
 *
 * ## Thread Safety
 *
 * All functions are pure lookups or dispatch calls and are
 * thread-safe.  The classification tables and the cast dispatch
 * table are read-only after process startup.
 *
 * @see dtype.h          Public API declarations and DType_ enum.
 * @see dtype_tables.h   Precomputed classification and size tables.
 * @see cast.h           Per-dtype cast function pointers (castFn).
 * @see tensor.h         Tensor struct embedding a DType_ field.
 */

#include <ncore/core/dtype.h>
#include <ncore/headeronly/cast.h>
#include <ncore/tables/dtype_tables.h>
#include <ncore/tensor.h>

/**
 * @var cast_dispatch
 * @brief Dispatch table for cast functions.
 *
 * @details
 * A `NUM_DTYPES × NUM_DTYPES` 2D array indexed by
 * `[source_dtype][target_dtype]`, containing function pointers to
 * the appropriate cast implementation.  Populated at compile time
 * from @ref cast_dispatch_tables.
 *
 * For example, `cast_dispatch[Float32][Signed32]` points to the
 * function that converts float32 elements to int32 elements.
 *
 * @see cast()
 * @see castFn
 */
extern CastFn cast_dispatch[NUM_DTYPES][NUM_DTYPES];

/**
 * @brief Check whether a tensor's dtype is a floating-point type.
 *
 * @details
 * Indexes into the `floating` lookup table with `input->dtype`
 * as the row index.  The table entry is `1` for float types
 * (`Float32`, `Float64`, `Float16`, `BFloat16`) and `0` for all
 * others.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `nullptr` and must have a valid `dtype` field.
 *
 * @return `true` if `input->dtype` is a floating-point type,
 *         `false` otherwise.
 *
 * @see is_integer()
 * @see is_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_floating(const Tensor *restrict input) {
  return (bool)floating[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is an integer type
 *        (signed or unsigned, including quantized).
 *
 * @details
 * Indexes into the `integer` lookup table with `input->dtype` as
 * the row index.  Returns `true` for all integer and quantized
 * integer types (`Signed8`, `UnSigned8`, `QSigned8`,
 * `QUnSigned8`, `Signed32`, `UnSigned32`, `Signed64`,
 * `UnSigned64`).
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `nullptr` and must have a valid `dtype` field.
 *
 * @return `true` if `input->dtype` is any integer type,
 *         `false` otherwise.
 *
 * @see is_floating()
 * @see is_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_integer(const Tensor *restrict input) {
  return (bool)integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is a signed integer type.
 *
 * @details
 * Indexes into the `signed_integer` lookup table with
 * `input->dtype` as the row index.  Returns `true` for
 * `Signed8`, `QSigned8`, `Signed32`, and `Signed64`.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `nullptr` and must have a valid `dtype` field.
 *
 * @return `true` if `input->dtype` is a signed integer type
 *         (including quantized), `false` otherwise.
 *
 * @see is_unsigned_integer()
 * @see is_quantized_signed_integer()
 * @see is_integer()
 */
bool is_signed_integer(const Tensor *restrict input) {
  return (bool)signed_integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is an unsigned integer type.
 *
 * @details
 * Indexes into the `unsigned_integer` lookup table with
 * `input->dtype` as the row index.  Returns `true` for
 * `UnSigned8`, `QUnSigned8`, `UnSigned32`, and `UnSigned64`.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `nullptr` and must have a valid `dtype` field.
 *
 * @return `true` if `input->dtype` is an unsigned integer type
 *         (including quantized), `false` otherwise.
 *
 * @see is_signed_integer()
 * @see is_quantized_unsigned_integer()
 * @see is_integer()
 */
bool is_unsigned_integer(const Tensor *restrict input) {
  return (bool)unsigned_integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is a quantized signed
 *        integer type.
 *
 * @details
 * Indexes into the `quantized_signed_integer` lookup table with
 * `input->dtype` as the row index.  Returns `true` only for
 * `QSigned8`.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `nullptr` and must have a valid `dtype` field.
 *
 * @return `true` if `input->dtype` is `QSigned8`, `false`
 *         otherwise.
 *
 * @see is_quantized_unsigned_integer()
 * @see is_signed_integer()
 * @see is_integer()
 */
bool is_quantized_signed_integer(const Tensor *restrict input) {
  return (bool)quantized_signed_integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is a quantized unsigned
 *        integer type.
 *
 * @details
 * Indexes into the `quantized_unsigned_integer` lookup table with
 * `input->dtype` as the row index.  Returns `true` only for
 * `QUnSigned8`.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   `nullptr` and must have a valid `dtype` field.
 *
 * @return `true` if `input->dtype` is `QUnSigned8`, `false`
 *         otherwise.
 *
 * @see is_quantized_signed_integer()
 * @see is_unsigned_integer()
 * @see is_integer()
 */
bool is_quantized_unsigned_integer(const Tensor *restrict input) {
  return (bool)quantized_unsigned_integer[input->dtype][0];
}

/**
 * @brief Cast a tensor's data to a different dtype.
 *
 * @details
 * Dispatches through the @ref cast_dispatch table to select the
 * correct element-wise conversion kernel.  The source tensor's
 * `dtype` field determines the source type, and @p target_dtype
 * determines the destination type.
 *
 * The destination tensor must be pre-allocated with the target
 * dtype and a shape compatible with the source.  This function
 * does not allocate memory — it only fills the data buffer of
 * @p dst.
 *
 * @param[in]  src           Source tensor.  Must not be `nullptr`,
 *                           must have `is_allocated_ == true`,
 *                           and a valid `storage` pointer.
 * @param[in]  target_dtype  Desired output @ref DType_.
 * @param[out] dst           Destination tensor (must be
 *                           pre-allocated with the target dtype).
 *                           Must not be `nullptr`.
 *
 * @pre  @p dst must have been created via
 *       `create_unallocated_tensor()` with the correct shape and
 *       the target dtype.
 * @pre  @p src must have a valid, allocated storage buffer.
 * @post On success, @p dst contains the type-converted copy of
 *       @p src's data.
 *
 * @see cast_dispatch  Dispatch table mapping (src, dst) dtype
 *                     pairs to cast kernels.
 * @see DType_         Runtime data-type identifier.
 */
void cast(const Tensor *restrict src, DType_ target_dtype,
          Tensor *restrict dst) {
  CastFn func = cast_dispatch[src->dtype][target_dtype];
  func(src, dst);
}

/**
 * @brief Return the size in bytes of a given @ref DType_.
 *
 * @details
 * Looks up the byte-width from the precomputed
 * `lookup_dtype_sizes` table, which is indexed by @ref DType_
 * values.  For example, `dtype_size(Float32)` returns `4` and
 * `dtype_size(Signed64)` returns `8`.
 *
 * @param[in] dtype  The data type to query.  Should be a valid
 *                   @ref DType_ value.
 *
 * @return Size of one element of @p dtype in bytes.
 *
 * @see DType_         Data-type enumeration.
 * @see lookup_dtype_sizes  Underlying lookup table.
 */
size_t dtype_size(DType_ dtype) { return lookup_dtype_sizes[dtype]; }
