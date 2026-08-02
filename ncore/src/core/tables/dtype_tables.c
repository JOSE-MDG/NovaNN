/**
 * @file dtype_tables.c
 * @brief Definitions of dtype classification and size lookup tables.
 *
 * @details
 * This file contains the definitions of the global @c const lookup
 * tables declared in @ref dtype_tables.h.  Each classification
 * table is a @c NUM_DTYPES × 1 array of @c bool, populated using
 * C99 designated initialisers so that only the @c true entries
 * need to be listed explicitly (all others default to @c false).
 *
 * The tables are:
 * @li @ref floating — floating-point types.
 * @li @ref integer — all integer types (signed, unsigned, quantized).
 * @li @ref signed_integer — signed integer types (including quantized).
 * @li @ref unsigned_integer — unsigned integer types (including quantized).
 * @li @ref quantized_signed_integer — quantized signed only.
 * @li @ref quantized_unsigned_integer — quantized unsigned only.
 * @li @ref lookup_dtype_sizes — byte-width per dtype.
 *
 * @section thread-safety Thread Safety
 *
 * All tables are @c const and read-only after process startup.
 * They are safe to access from any thread.
 *
 * @see dtype_tables.h  Public declarations.
 * @see dtype.c         Classification functions using these tables.
 * @see dtype.h         DType_ enumeration.
 */

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tables/dtype_tables.h>

/**
 * @var floating
 * @brief Boolean mask for floating-point dtypes.
 *
 * @details
 * @c floating[dtype][0] is @c true when @c dtype is @c Float32,
 * @c Float64, @c Float16, @c BFloat16, @c Float8E4M3fn, @c Float8E5M2,
 * or @c Float4E2M1fn.  All other entries are @c false.  Used by
 * @ref is_floating() in @ref dtype.c.
 */
const bool floating[NUM_DTYPES][1] = {
    [Float32] = {true},      [Float64] = {true},      [Float16] = {true},
    [BFloat16] = {true},     [Float8E4M3fn] = {true}, [Float8E5M2] = {true},
    [Float4E2M1fn] = {true}, [Signed8] = {false},     [UnSigned8] = {false},
    [QSigned8] = {false},    [QUnSigned8] = {false},  [Signed16] = {false},
    [UnSigned16] = {false},  [QSigned16] = {false},   [QUnSigned16] = {false},
    [Signed32] = {false},    [UnSigned32] = {false},  [QSigned32] = {false},
    [QUnSigned32] = {false}, [Signed64] = {false},    [UnSigned64] = {false},
};

/**
 * @var integer
 * @brief Boolean mask for all integer dtypes (signed, unsigned,
 *        and quantized).
 *
 * @details
 * @c integer[dtype][0] is @c true for @c Signed8, @c UnSigned8,
 * @c QSigned8, @c QUnSigned8, @c Signed16, @c UnSigned16, @c QSigned16,
 * @c QUnSigned16, @c Signed32, @c UnSigned32, @c QSigned32,
 * @c QUnSigned32, @c Signed64, @c UnSigned64.  All float types are
 * @c false.  Used by @ref is_integer() in @ref dtype.c.
 */
const bool integer[NUM_DTYPES][1] = {
    [Float32] = {false},      [Float64] = {false},      [Float16] = {false},
    [BFloat16] = {false},     [Float8E4M3fn] = {false}, [Float8E5M2] = {false},
    [Float4E2M1fn] = {false}, [Signed8] = {true},       [UnSigned8] = {true},
    [QSigned8] = {true},      [QUnSigned8] = {true},    [Signed16] = {true},
    [UnSigned16] = {true},    [QSigned16] = {true},     [QUnSigned16] = {true},
    [Signed32] = {true},      [UnSigned32] = {true},    [QSigned32] = {true},
    [QUnSigned32] = {true},   [Signed64] = {true},      [UnSigned64] = {true},
};

/**
 * @var signed_integer
 * @brief Boolean mask for signed integer dtypes (including
 *        quantized).
 *
 * @details
 * @c signed_integer[dtype][0] is @c true for @c Signed8, @c QSigned8,
 * @c Signed16, @c QSigned16, @c Signed32, @c QSigned32, and @c Signed64.
 * Note that the quantized types @c QSigned8, @c QSigned16, and
 * @c QSigned32 are included because they are backed by signed
 * storage types.  Used by @ref is_signed_integer() in @ref dtype.c.
 */
const bool signed_integer[NUM_DTYPES][1] = {
    [Float32] = {false},      [Float64] = {false},      [Float16] = {false},
    [BFloat16] = {false},     [Float8E4M3fn] = {false}, [Float8E5M2] = {false},
    [Float4E2M1fn] = {false}, [Signed8] = {true},       [UnSigned8] = {false},
    [QSigned8] = {true},      [QUnSigned8] = {false},   [Signed16] = {true},
    [UnSigned16] = {false},   [QSigned16] = {true},     [QUnSigned16] = {false},
    [Signed32] = {true},      [UnSigned32] = {false},   [QSigned32] = {true},
    [QUnSigned32] = {false},  [Signed64] = {true},      [UnSigned64] = {false},
};

/**
 * @var unsigned_integer
 * @brief Boolean mask for unsigned integer dtypes (including
 *        quantized).
 *
 * @details
 * @c unsigned_integer[dtype][0] is @c true for @c UnSigned8,
 * @c QUnSigned8, @c UnSigned16, @c QUnSigned16, @c UnSigned32,
 * @c QUnSigned32, and @c UnSigned64.  Note that the quantized types
 * @c QUnSigned8, @c QUnSigned16, and @c QUnSigned32 are included
 * because they are backed by unsigned storage types.  Used by
 * @ref is_unsigned_integer() in @ref dtype.c.
 */
const bool unsigned_integer[NUM_DTYPES][1] = {
    [Float32] = {false},      [Float64] = {false},      [Float16] = {false},
    [BFloat16] = {false},     [Float8E4M3fn] = {false}, [Float8E5M2] = {false},
    [Float4E2M1fn] = {false}, [Signed8] = {false},      [UnSigned8] = {true},
    [QSigned8] = {false},     [QUnSigned8] = {true},    [Signed16] = {false},
    [UnSigned16] = {true},    [QSigned16] = {false},    [QUnSigned16] = {true},
    [Signed32] = {false},     [UnSigned32] = {true},    [QSigned32] = {false},
    [QUnSigned32] = {true},   [Signed64] = {false},     [UnSigned64] = {true},
};

/**
 * @var quantized_signed_integer
 * @brief Boolean mask for quantized signed integer dtypes.
 *
 * @details
 * @c quantized_signed_integer[dtype][0] is @c true for @c QSigned8,
 * @c QSigned16, and @c QSigned32.  Used by
 * @ref is_quantized_signed_integer() in @ref dtype.c.
 */
const bool quantized_signed_integer[NUM_DTYPES][1] = {
    [Float32] = {false},      [Float64] = {false},      [Float16] = {false},
    [BFloat16] = {false},     [Float8E4M3fn] = {false}, [Float8E5M2] = {false},
    [Float4E2M1fn] = {false}, [Signed8] = {false},      [UnSigned8] = {false},
    [QSigned8] = {true},      [QUnSigned8] = {false},   [Signed16] = {false},
    [UnSigned16] = {false},   [QSigned16] = {true},     [QUnSigned16] = {false},
    [Signed32] = {false},     [UnSigned32] = {false},   [QSigned32] = {true},
    [QUnSigned32] = {false},  [Signed64] = {false},     [UnSigned64] = {false},
};

/**
 * @var quantized_unsigned_integer
 * @brief Boolean mask for quantized unsigned integer dtypes.
 *
 * @details
 * @c quantized_unsigned_integer[dtype][0] is @c true for
 * @c QUnSigned8, @c QUnSigned16, and @c QUnSigned32.  Used by
 * @ref is_quantized_unsigned_integer() in @ref dtype.c.
 */
const bool quantized_unsigned_integer[NUM_DTYPES][1] = {
    [Float32] = {false},      [Float64] = {false},      [Float16] = {false},
    [BFloat16] = {false},     [Float8E4M3fn] = {false}, [Float8E5M2] = {false},
    [Float4E2M1fn] = {false}, [Signed8] = {false},      [UnSigned8] = {false},
    [QSigned8] = {false},     [QUnSigned8] = {true},    [Signed16] = {false},
    [UnSigned16] = {false},   [QSigned16] = {false},    [QUnSigned16] = {true},
    [Signed32] = {false},     [UnSigned32] = {false},   [QSigned32] = {false},
    [QUnSigned32] = {true},   [Signed64] = {false},     [UnSigned64] = {false},
};

/**
 * @var quantizable_dtype
 * @brief Boolean mask for dtypes that can be quantized.
 *
 * @details
 * @c quantizable_dtype[dtype][0] is @c true for @c Float4E2M1fn,
 * @c QSigned8, @c QUnSigned8, @c QSigned16, @c QUnSigned16,
 * @c QSigned32, and @c QUnSigned32.  Used by
 * @ref is_quantizable_dtype() in @ref dtype.c.
 */
const bool quantizable_dtype[NUM_DTYPES][1] = {
    [Float32] = {false},     [Float64] = {false},      [Float16] = {false},
    [BFloat16] = {false},    [Float8E4M3fn] = {false}, [Float8E5M2] = {false},
    [Float4E2M1fn] = {true}, [Signed8] = {false},      [UnSigned8] = {false},
    [QSigned8] = {true},     [QUnSigned8] = {true},    [Signed16] = {false},
    [UnSigned16] = {false},  [QSigned16] = {true},     [QUnSigned16] = {true},
    [Signed32] = {false},    [UnSigned32] = {false},   [QSigned32] = {true},
    [QUnSigned32] = {true},  [Signed64] = {false},     [UnSigned64] = {false},
};

/**
 * @var lookup_dtype_sizes
 * @brief Byte-width of each @ref DType_, indexed by the enum value.
 *
 * @details
 * @c lookup_dtype_sizes[dtype] returns @c sizeof the corresponding
 * C type for that dtype.  For example:
 * @li @c lookup_dtype_sizes[Float32] = @c sizeof(float32) = 4
 * @li @c lookup_dtype_sizes[Float64] = @c sizeof(float64) = 8
 * @li @c lookup_dtype_sizes[Float8E4M3fn] = @c sizeof(float8_e4m3fn) = 1
 * @li @c lookup_dtype_sizes[Signed64] = @c sizeof(int64) = 8
 * @li @c lookup_dtype_sizes[QSigned8] = @c sizeof(qint8) = 1
 *
 * Used by @ref dtype_size() in @ref dtype.c.
 */
const size_t lookup_dtype_sizes[NUM_DTYPES] = {
    [Float32] = sizeof(float32),
    [Float64] = sizeof(float64),
    [Float16] = sizeof(float16),
    [BFloat16] = sizeof(bfloat16),
    [Float8E4M3fn] = sizeof(float8_e4m3fn),
    [Float8E5M2] = sizeof(float8_e5m2),
    [Float4E2M1fn] = sizeof(float4_e2m1fn_x2),
    [Signed8] = sizeof(int8),
    [UnSigned8] = sizeof(uint8),
    [QSigned8] = sizeof(qint8),
    [QUnSigned8] = sizeof(quint8),
    [Signed16] = sizeof(int16),
    [UnSigned16] = sizeof(uint16),
    [QSigned16] = sizeof(qint16),
    [QUnSigned16] = sizeof(quint16),
    [Signed32] = sizeof(int32),
    [UnSigned32] = sizeof(uint32),
    [QSigned32] = sizeof(qint32),
    [QUnSigned32] = sizeof(quint32),
    [Signed64] = sizeof(int64),
    [UnSigned64] = sizeof(uint64),
};
