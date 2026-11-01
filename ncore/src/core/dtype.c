/**
 * @file dtype.c
 * @brief Implementation of dtype classification and casting functions.
 *
 * @details
 * Classification queries index precomputed boolean lookup tables
 * from @ref dtype_tables.h (O(1), no switch); casting resolves the
 * element-wise kernel through the @c cast_dispatch 2D array,
 * defined in @c cast_dispatch_tables.c.  GPU-resident tensors go to
 * the GPU casting kernels.  All state is read-only after process
 * startup, so every function is thread-safe.
 *
 * @see dtype.h          Public API declarations and DType_ enum.
 * @see dtype_tables.h   Precomputed classification and size tables.
 * @see cast.h           Per-dtype cast function pointers (castFn).
 * @see tensor.h         Tensor struct embedding a DType_ field.
 */

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/cast.h>
#include <ncore/native/kernels/casting.h>
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
 * @see cast()
 * @see castFn
 */
extern CastFn cast_dispatch[NUM_DTYPES][NUM_DTYPES];

/**
 * @brief Check whether a tensor's dtype is a floating-point type.
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
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is any non-quantized integer
 *         type, @c false otherwise.
 *
 * @see is_floating()
 * @see is_signed_integer()
 * @see is_unsigned_integer()
 */
bool is_integer(const Tensor *restrict input) {
  return integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is a signed integer type
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is a non-quantized signed
 *         integer type, @c false otherwise.
 *
 * @see is_unsigned_integer()
 * @see is_quantized_signed_integer()
 * @see is_integer()
 */
bool is_signed_integer(const Tensor *restrict input) {
  return signed_integer[input->dtype][0];
}

/**
 * @brief Check whether a tensor's dtype is an unsigned integer type
 *
 * @param[in] input  Pointer to the tensor to query.  Must not be
 *                   @c nullptr and must have a valid @c dtype field.
 *
 * @return @c true if @c input->dtype is a non-quantized unsigned
 *         integer type, @c false otherwise.
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
 * Tensors on the compute device dispatch to the GPU casting kernels
 * and report the launch status.  All other tensors dispatch through
 * the @ref cast_dispatch table to select the correct element-wise
 * conversion kernel.  The source tensor's @c dtype field determines
 * the source type, and @p target_dtype determines the destination
 * type.
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
 * @return The selected kernel's status with @p dst holding the
 *         type-converted copy of @p src's data on success, or an error
 *         status describing the failure.
 *
 * @pre  @p dst must have been created via
 *       @c create_unallocated_tensor() with the correct shape and
 *       the target dtype.
 * @pre  @p src must have a valid, allocated storage buffer.
 *
 * @see cast_dispatch  Dispatch table mapping (src, dst) dtype
 *                     pairs to cast kernels.
 * @see DType_         Runtime data-type identifier.
 */
novaStatus_t cast(const Tensor *restrict src, Tensor *restrict dst,
                  DType_ target_dtype) {
  if (in_device(src)) {
    return launchDtypeCastingKernel(src, dst);
  }
  const CastFn func = cast_dispatch[src->dtype][target_dtype];
  if (func == nullptr) {
    return (novaStatus_t){
        .err = novaCastNotSupported,
        .message = nova_get_error_msg(novaCastNotSupported, nullptr)};
  }
  novaStatus_t st;
  func(src, dst, &st);
  return st;
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
  return cast_dispatch[src_dtype][target_dtype];
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
