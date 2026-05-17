/**
 * @file dtype_tables.c
 * @brief Definitions of dtype classification lookup tables.
 *
 * This file contains the initialization of the global lookup tables
 * that categorize data types. Each table provides a boolean mask
 * for checking whether a given dtype belongs to a specific category.
 *
 * @note These tables use designated initializers for clarity and
 *       maintainability. Only the true entries are explicitly listed.
 */

#include <ncore/dtype.h>
#include <ncore/tables/dtype_tables.h>

/**
 * @brief Lookup table for floating-point dtypes.
 *
 * Contains true for Float32, Float64, Float16, and BFloat16.
 */
const bool floating[NUM_DTYPES][1] = {
    [Float32] = {true},     [Float64] = {true},     [Float16] = {true},
    [BFloat16] = {true},    [Signed8] = {false},    [UnSigned8] = {false},
    [QSigned8] = {false},   [QUnSigned8] = {false}, [Signed32] = {false},
    [UnSigned32] = {false}, [Signed64] = {false},   [UnSigned64] = {false},
};

/**
 * @brief Lookup table for integer dtypes.
 *
 * Contains true for all signed and unsigned integer types,
 * as well as quantized integer types.
 */
const bool integer[NUM_DTYPES][1] = {
    [Float32] = {false},   [Float64] = {false},   [Float16] = {false},
    [BFloat16] = {false},  [Signed8] = {true},    [UnSigned8] = {true},
    [QSigned8] = {true},   [QUnSigned8] = {true}, [Signed32] = {true},
    [UnSigned32] = {true}, [Signed64] = {true},   [UnSigned64] = {true},
};

/**
 * @brief Lookup table for signed integer dtypes.
 *
 * Contains true for Signed8 (note: likely typo for Signed8), Signed32,
 * Signed64, and quantized signed types.
 */
const bool signed_integer[NUM_DTYPES][1] = {
    [Float32] = {false},    [Float64] = {false},    [Float16] = {false},
    [BFloat16] = {false},   [Signed8] = {true},     [UnSigned8] = {false},
    [QSigned8] = {true},    [QUnSigned8] = {false}, [Signed32] = {true},
    [UnSigned32] = {false}, [Signed64] = {true},    [UnSigned64] = {false},
};

/**
 * @brief Lookup table for unsigned integer dtypes.
 *
 * Contains true for UnSigned8, UnSigned32, UnSigned64, and
 * quantized unsigned types.
 */
const bool unsigned_integer[NUM_DTYPES][1] = {
    [Float32] = {false},   [Float64] = {false},   [Float16] = {false},
    [BFloat16] = {false},  [Signed8] = {false},   [UnSigned8] = {true},
    [QSigned8] = {false},  [QUnSigned8] = {true}, [Signed32] = {false},
    [UnSigned32] = {true}, [Signed64] = {false},  [UnSigned64] = {true},
};

/**
 * @brief Lookup table for quantized signed integer dtypes.
 *
 * Contains true only for QSigned8.
 */
const bool quantized_signed_integer[NUM_DTYPES][1] = {
    [Float32] = {false},    [Float64] = {false},    [Float16] = {false},
    [BFloat16] = {false},   [Signed8] = {false},    [UnSigned8] = {false},
    [QSigned8] = {true},    [QUnSigned8] = {false}, [Signed32] = {false},
    [UnSigned32] = {false}, [Signed64] = {false},   [UnSigned64] = {false},
};

/**
 * @brief Lookup table for quantized unsigned integer dtypes.
 *
 * Contains true only for QUnSigned8.
 */
const bool quantized_unsigned_integer[NUM_DTYPES][1] = {
    [Float32] = {false},    [Float64] = {false},   [Float16] = {false},
    [BFloat16] = {false},   [Signed8] = {false},   [UnSigned8] = {false},
    [QSigned8] = {false},   [QUnSigned8] = {true}, [Signed32] = {false},
    [UnSigned32] = {false}, [Signed64] = {false},  [UnSigned64] = {false},
};

/**
 * @brief Lookup table for the size of the types of each data type
 *
 * Contains the size for each data type
 */
const size_t lookup_dtype_sizes[NUM_DTYPES] = {[Float16] = sizeof(float16),
                                               [BFloat16] = sizeof(bfloat16),
                                               [Float32] = sizeof(float32),
                                               [Float64] = sizeof(float64),
                                               [Signed8] = sizeof(int8),
                                               [UnSigned8] = sizeof(uint8),
                                               [Signed32] = sizeof(int32),
                                               [UnSigned32] = sizeof(uint32),
                                               [Signed64] = sizeof(int64),
                                               [UnSigned64] = sizeof(uint64),
                                               [QSigned8] = sizeof(qint8),
                                               [QUnSigned8] = sizeof(quint8)

};
