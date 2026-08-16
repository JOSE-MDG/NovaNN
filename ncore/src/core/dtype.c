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
 * @section classification Classification
 *
 * The six classification functions (@c is_floating, @c is_integer,
 * etc.) each index into a precomputed @c uint8_t lookup table
 * (e.g., @c floating[NUM_DTYPES][1]) and cast the result to @c bool.
 * The tables are populated at compile time from the @ref DType_
 * enumeration.  This avoids switch statements and gives O(1)
 * classification for any dtype.
 *
 * @section cast-dispatch Cast Dispatch
 *
 * The @c cast() function resolves the correct element-wise
 * conversion via the @c cast_dispatch 2D array (indexed by
 * @c [source_dtype][target_dtype]) and calls the matching
 * @c castFn implementation.  The actual cast kernels are defined
 * in the @c cast_dispatch_tables and @c cast_tables translation
 * units.
 *
 * @section thread-safety Thread Safety
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
 * A @c NUM_DTYPES × NUM_DTYPES 2D array indexed by
 * @c [source_dtype][target_dtype], containing function pointers to
 * the appropriate cast implementation.  Populated at compile time
 * from @ref cast_dispatch_tables.
 *
 * For example, @c cast_dispatch[Float32][Signed32] points to the
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
 * Indexes into the @c floating lookup table with @c input->dtype
 * as the row index.  The table entry is @c 1 for float types
 * (@c Float32, @c Float64, @c Float16, @c BFloat16,
 * @c Float8E4M3fn, @c Float8E5M2, @c Float4E2M1fn) and @c 0 for all
 * others.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is a floating-point type,
 *         @c false otherwise.
 *
 * @see is_integer()
 * @see is_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_floating(const Tensor *restrict input) {
  return floating[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is an integer type
 *        (signed or unsigned, including quantized).
 *
 * @details
 * Indexes into the @c integer lookup table with @c input->dtype as
 * the row index.  Returns @c true for all integer and quantized
 * integer types (@c Signed8, @c Signed16, @c Signed32, @c Signed64,
 * @c UnSigned8, @c UnSigned16, @c UnSigned32, @c UnSigned64,
 * @c QSigned8, @c QSigned16, @c QSigned32, @c QUnSigned8,
 * @c QUnSigned16, @c QUnSigned32).
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is any integer type,
 *         @c false otherwise.
 *
 * @see is_floating()
 * @see is_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_integer(const Tensor *restrict input) {
  return integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is a signed integer type.
 *
 * @details
 * Indexes into the @c signed_integer lookup table with
 * @c input->dtype as the row index.  Returns @c true for
 * @c Signed8, @c QSigned8, @c Signed32, and @c Signed64.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is a signed integer type
 *         (including quantized), @c false otherwise.
 *
 * @see is_unsigned_integer()
 * @see is_quantized_signed_integer()
 * @see is_integer()
 */
bool is_signed_integer(const Tensor *restrict input) {
  return signed_integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is an unsigned integer type.
 *
 * @details
 * Indexes into the @c unsigned_integer lookup table with
 * @c input->dtype as the row index.  Returns @c true for
 * @c UnSigned8, @c QUnSigned8, @c UnSigned32, and @c UnSigned64.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is an unsigned integer type
 *         (including quantized), @c false otherwise.
 *
 * @see is_signed_integer()
 * @see is_quantized_unsigned_integer()
 * @see is_integer()
 */
bool is_unsigned_integer(const Tensor *restrict input) {
  return unsigned_integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is a quantized signed
 *        integer type.
 *
 * @details
 * Indexes into the @c quantized_signed_integer lookup table with
 * @c input->dtype as the row index.  Returns @c true only for
 * @c QSigned8.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is @c QSigned8, @c false
 *         otherwise.
 *
 * @see is_quantized_unsigned_integer()
 * @see is_signed_integer()
 * @see is_integer()
 */
bool is_quantized_signed_integer(const Tensor *restrict input) {
  return quantized_signed_integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is a quantized unsigned
 *        integer type.
 *
 * @details
 * Indexes into the @c quantized_unsigned_integer lookup table with
 * @c input->dtype as the row index.  Returns @c true only for
 * @c QUnSigned8.
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is @c QUnSigned8, @c false
 *         otherwise.
 *
 * @see is_quantized_signed_integer()
 * @see is_unsigned_integer()
 * @see is_integer()
 */
bool is_quantized_unsigned_integer(const Tensor *restrict input) {
  return quantized_unsigned_integer[input->dtype][0];
}

/**
 * @brief Check whether a given @ref DType_ can be quantized.
 *
 * @details
 * Indexes into the @c quantizable_dtype lookup table with @p dtype as
 * the index.  Returns @c true for @c Float4E2M1fn, @c QSigned8,
 * @c QUnSigned8, @c QSigned16, @c QUnSigned16, @c QSigned32, and
 * @c QUnSigned32.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return @c true if @p dtype is a quantizable type, @c false otherwise.
 *
 * @see is_floating()
 * @see is_integer()
 * @see quantizable_dtype  Underlying lookup table.
 */
bool is_quantizable_dtype(DType_ dtype) { return quantizable_dtype[dtype][0]; }

/**
 * @brief Cast a tensor's data to a different dtype.
 *
 * @details
 * Dispatches through the @ref cast_dispatch table to select the
 * correct element-wise conversion kernel.  The source tensor's
 * @c dtype field determines the source type, and @p target_dtype
 * determines the destination type.
 *
 * The destination tensor must be pre-allocated with the target
 * dtype and a shape compatible with the source.  This function
 * does not allocate memory — it only fills the data buffer of
 * @p dst.
 *
 * @param[in]  src           Source tensor.  Must not be @c nullptr,
 *                           must have @c is_allocated_ == true,
 *                           and a valid @c storage pointer.
 * @param[out] dst           Destination tensor (must be
 *                           pre-allocated with the target dtype).
 *                           Must not be @c nullptr.
 * @param[in]  target_dtype  Desired output @ref DType_.
 *
 * @pre  @p dst must have been created via
 *       @c create_unallocated_tensor() with the correct shape and
 *       the target dtype.
 * @pre  @p src must have a valid, allocated storage buffer.
 * @post On success, @p dst contains the type-converted copy of
 *       @p src's data.
 *
 * @see cast_dispatch  Dispatch table mapping (src, dst) dtype
 *                     pairs to cast kernels.
 * @see DType_         Runtime data-type identifier.
 */
void cast(const Tensor *restrict src, Tensor *restrict dst,
          DType_ target_dtype) {
  CastFn func = cast_dispatch[src->dtype][target_dtype];
  func(src, dst);
}

/**
 * @brief Return the @ref CastFn function pointer registered for a
 *        (source, target) dtype pair.
 *
 * @details
 * Indexes into the @ref cast_dispatch table with @p src_dtype as
 * the row index and @p target_dtype as the column index, and
 * returns the element-wise conversion kernel stored at that
 * position, without executing it.
 *
 * Unlike @ref cast(), which calls the resolved kernel directly
 * with no null check, this function lets the caller validate a
 * pair up front.  The table is populated at load time and
 * read-only afterwards, so the lookup is O(1).
 *
 * @param[in] src_dtype    Source data type.
 * @param[in] target_dtype Destination data type.
 *
 * @return The @ref CastFn function pointer stored at
 *         @c cast_dispatch[src_dtype][target_dtype], or
 *         @c nullptr if the cast is the identity or not
 *         supported.
 *
 * @see cast()         Executes the resolved kernel.
 * @see cast_dispatch  Underlying dispatch table.
 * @see CastFn         Function pointer type.
 */
CastFn get_dispatched_cast_func(DType_ src_dtype, DType_ target_dtype) {
  CastFn fn = cast_dispatch[src_dtype][target_dtype];
  return fn;
}

/**
 * @brief Return the size in bytes of a given @ref DType_.
 *
 * @details
 * Looks up the byte-width from the precomputed
 * @c lookup_dtype_sizes table, which is indexed by @ref DType_
 * values.  For example, @c dtype_size(Float32) returns @c 4 and
 * @c dtype_size(Signed64) returns @c 8.
 *
 * @param[in] dtype  The data type to query.  Should be a valid
 *                   @ref DType_ value.
 *
 * @return Size of one element of @p dtype in bytes.
 *
 * @see DType_              Data-type enumeration.
 * @see lookup_dtype_sizes  Underlying lookup table.
 */
size_t dtype_size(DType_ dtype) { return lookup_dtype_sizes[dtype]; }

/**
 * @brief Return the packing factor of a given @ref DType_.
 *
 * @details
 * The packing factor indicates how many logical elements are stored
 * in a single storage unit.  For most types this is 1, meaning one
 * element per storage byte (or word).  For packed types such as
 * @ref Float4E2M1fn the factor is 2, because each byte holds two
 * 4-bit elements.
 *
 * @param[in] dtype  The data type to query.
 *
 * @return Number of logical elements packed into one storage unit.
 *         Typically 1, except for packed types where it is 2.
 *
 * @see dtype_size()
 * @see DType_
 */
size_t dtype_packing_factor(DType_ dtype) {
  if (dtype == Float4E2M1fn) {
    return 2;
  }
  return 1;
}
