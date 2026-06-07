/**
 * @file dtype_tables.c
 * @brief Definitions of dtype classification and size lookup tables.
 *
 * @details
 * This file contains the definitions of the global `const` lookup
 * tables declared in @ref dtype_tables.h.  Each classification
 * table is a `NUM_DTYPES × 1` array of `bool`, populated using
 * C99 designated initialisers so that only the `true` entries
 * need to be listed explicitly (all others default to `false`).
 *
 * The tables are:
 * - @ref floating — floating-point types.
 * - @ref integer — all integer types (signed, unsigned, quantized).
 * - @ref signed_integer — signed integer types (including quantized).
 * - @ref unsigned_integer — unsigned integer types (including quantized).
 * - @ref quantized_signed_integer — quantized signed only.
 * - @ref quantized_unsigned_integer — quantized unsigned only.
 * - @ref lookup_dtype_sizes — byte-width per dtype.
 *
 * ## Thread Safety
 *
 * All tables are `const` and read-only after process startup.
 * They are safe to access from any thread.
 *
 * @see dtype_tables.h  Public declarations.
 * @see dtype.c         Classification functions using these tables.
 * @see dtype.h         DType_ enumeration.
 */

#include <ncore/dtype.h>
#include <ncore/tables/dtype_tables.h>

/**
 * @var floating
 * @brief Boolean mask for floating-point dtypes.
 *
 * @details
 * `floating[dtype][0]` is `true` when `dtype` is `Float32`,
 * `Float64`, `Float16`, or `BFloat16`.  All other entries are
 * `false`.  Used by @ref is_floating() in @ref dtype.c.
 */
const bool floating[NUM_DTYPES][1] = {
    [Float32] = {true},     [Float64] = {true},     [Float16] = {true},
    [BFloat16] = {true},    [Signed8] = {false},    [UnSigned8] = {false},
    [QSigned8] = {false},   [QUnSigned8] = {false}, [Signed32] = {false},
    [UnSigned32] = {false}, [Signed64] = {false},   [UnSigned64] = {false},
};

/**
 * @var integer
 * @brief Boolean mask for all integer dtypes (signed, unsigned,
 *        and quantized).
 *
 * @details
 * `integer[dtype][0]` is `true` for `Signed8`, `UnSigned8`,
 * `QSigned8`, `QUnSigned8`, `Signed32`, `UnSigned32`, `Signed64`,
 * and `UnSigned64`.  All float types are `false`.  Used by
 * @ref is_integer() in @ref dtype.c.
 */
const bool integer[NUM_DTYPES][1] = {
    [Float32] = {false},   [Float64] = {false},   [Float16] = {false},
    [BFloat16] = {false},  [Signed8] = {true},    [UnSigned8] = {true},
    [QSigned8] = {true},   [QUnSigned8] = {true}, [Signed32] = {true},
    [UnSigned32] = {true}, [Signed64] = {true},   [UnSigned64] = {true},
};

/**
 * @var signed_integer
 * @brief Boolean mask for signed integer dtypes (including
 *        quantized).
 *
 * @details
 * `signed_integer[dtype][0]` is `true` for `Signed8`, `QSigned8`,
 * `Signed32`, and `Signed64`.  Note that the quantized type
 * `QSigned8` is included because it is backed by a signed
 * `int8_t` storage type.  Used by @ref is_signed_integer() in
 * @ref dtype.c.
 */
const bool signed_integer[NUM_DTYPES][1] = {
    [Float32] = {false},    [Float64] = {false},    [Float16] = {false},
    [BFloat16] = {false},   [Signed8] = {true},     [UnSigned8] = {false},
    [QSigned8] = {true},    [QUnSigned8] = {false}, [Signed32] = {true},
    [UnSigned32] = {false}, [Signed64] = {true},    [UnSigned64] = {false},
};

/**
 * @var unsigned_integer
 * @brief Boolean mask for unsigned integer dtypes (including
 *        quantized).
 *
 * @details
 * `unsigned_integer[dtype][0]` is `true` for `UnSigned8`,
 * `QUnSigned8`, `UnSigned32`, and `UnSigned64`.  Note that the
 * quantized type `QUnSigned8` is included because it is backed
 * by an unsigned `uint8_t` storage type.  Used by
 * @ref is_unsigned_integer() in @ref dtype.c.
 */
const bool unsigned_integer[NUM_DTYPES][1] = {
    [Float32] = {false},   [Float64] = {false},   [Float16] = {false},
    [BFloat16] = {false},  [Signed8] = {false},   [UnSigned8] = {true},
    [QSigned8] = {false},  [QUnSigned8] = {true}, [Signed32] = {false},
    [UnSigned32] = {true}, [Signed64] = {false},  [UnSigned64] = {true},
};

/**
 * @var quantized_signed_integer
 * @brief Boolean mask for quantized signed integer dtypes.
 *
 * @details
 * `quantized_signed_integer[dtype][0]` is `true` only for
 * `QSigned8`.  Used by @ref is_quantized_signed_integer() in
 * @ref dtype.c.
 */
const bool quantized_signed_integer[NUM_DTYPES][1] = {
    [Float32] = {false},    [Float64] = {false},    [Float16] = {false},
    [BFloat16] = {false},   [Signed8] = {false},    [UnSigned8] = {false},
    [QSigned8] = {true},    [QUnSigned8] = {false}, [Signed32] = {false},
    [UnSigned32] = {false}, [Signed64] = {false},   [UnSigned64] = {false},
};

/**
 * @var quantized_unsigned_integer
 * @brief Boolean mask for quantized unsigned integer dtypes.
 *
 * @details
 * `quantized_unsigned_integer[dtype][0]` is `true` only for
 * `QUnSigned8`.  Used by @ref is_quantized_unsigned_integer() in
 * @ref dtype.c.
 */
const bool quantized_unsigned_integer[NUM_DTYPES][1] = {
    [Float32] = {false},    [Float64] = {false},   [Float16] = {false},
    [BFloat16] = {false},   [Signed8] = {false},   [UnSigned8] = {false},
    [QSigned8] = {false},   [QUnSigned8] = {true}, [Signed32] = {false},
    [UnSigned32] = {false}, [Signed64] = {false},  [UnSigned64] = {false},
};

/**
 * @var lookup_dtype_sizes
 * @brief Byte-width of each @ref DType_, indexed by the enum value.
 *
 * @details
 * `lookup_dtype_sizes[dtype]` returns `sizeof` the corresponding
 * C type for that dtype.  For example:
 * - `lookup_dtype_sizes[Float32]` = `sizeof(float32)` = 4
 * - `lookup_dtype_sizes[Float64]` = `sizeof(float64)` = 8
 * - `lookup_dtype_sizes[Signed64]` = `sizeof(int64)` = 8
 * - `lookup_dtype_sizes[QSigned8]` = `sizeof(qint8)` = 1
 *
 * Used by @ref dtype_size() in @ref dtype.c.
 */
const size_t lookup_dtype_sizes[NUM_DTYPES] = {
    [Float32] = sizeof(float32), [Float64] = sizeof(float64),
    [Float16] = sizeof(float16), [BFloat16] = sizeof(bfloat16),
    [Signed8] = sizeof(int8),    [UnSigned8] = sizeof(uint8),
    [QSigned8] = sizeof(qint8),  [QUnSigned8] = sizeof(quint8),
    [Signed32] = sizeof(int32),  [UnSigned32] = sizeof(uint32),
    [Signed64] = sizeof(int64),  [UnSigned64] = sizeof(uint64),
};
