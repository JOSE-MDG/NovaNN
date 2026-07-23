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

#include <ncore/core/dtype.h>
#include <ncore/headeronly/macros.h>
#include <ncore/tables/dtype_tables.h>

/**
 * @var floating
 * @brief Boolean mask for floating-point dtypes.
 *
 * @details
 * `floating[dtype][0]` is `true` when `dtype` is `Float32`,
 * `Float64`, `Float16`, `BFloat16`, `Float8E4M3fn`, `Float8E5M2`,
 * or `Float4E2M1fn`.  All other entries are `false`.  Used by
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
 * `integer[dtype][0]` is `true` for `Signed8`, `UnSigned8`,
 * `QSigned8`, `QUnSigned8`, `Signed16`, `UnSigned16`, `QSigned16`,
 * `QUnSigned16`, `Signed32`, `UnSigned32`, `QSigned32`,
 * `QUnSigned32`, `Signed64`, `UnSigned64`.  All float types are
 * `false`.  Used by @ref is_integer() in @ref dtype.c.
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
 * `signed_integer[dtype][0]` is `true` for `Signed8`, `QSigned8`,
 * `Signed16`, `QSigned16`, `Signed32`, `QSigned32`, and `Signed64`.
 * Note that the quantized types `QSigned8`, `QSigned16`, and
 * `QSigned32` are included because they are backed by signed
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
 * `unsigned_integer[dtype][0]` is `true` for `UnSigned8`,
 * `QUnSigned8`, `UnSigned16`, `QUnSigned16`, `UnSigned32`,
 * `QUnSigned32`, and `UnSigned64`.  Note that the quantized types
 * `QUnSigned8`, `QUnSigned16`, and `QUnSigned32` are included
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
 * `quantized_signed_integer[dtype][0]` is `true` for `QSigned8`,
 * `QSigned16`, and `QSigned32`.  Used by
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
 * `quantized_unsigned_integer[dtype][0]` is `true` for
 * `QUnSigned8`, `QUnSigned16`, and `QUnSigned32`.  Used by
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
 * `quantizable_dtype[dtype][0]` is `true` for `Float4E2M1fn`,
 * `QSigned8`, `QUnSigned8`, `QSigned16`, `QUnSigned16`,
 * `QSigned32`, and `QUnSigned32`.  Used by
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
 * `lookup_dtype_sizes[dtype]` returns `sizeof` the corresponding
 * C type for that dtype.  For example:
 * - `lookup_dtype_sizes[Float32]` = `sizeof(float32)` = 4
 * - `lookup_dtype_sizes[Float64]` = `sizeof(float64)` = 8
 * - `lookup_dtype_sizes[Float8E4M3fn]` = `sizeof(float8_e4m3fn)` = 1
 * - `lookup_dtype_sizes[Signed64]` = `sizeof(int64)` = 8
 * - `lookup_dtype_sizes[QSigned8]` = `sizeof(qint8)` = 1
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
