/**
 * @file macros.h
 * @brief Core macros, constants, and SIMD lane-count definitions.
 *
 * @details
 * Provides foundational preprocessor definitions used throughout NovaNN:
 * alignment attributes, dimension limits, type-count constants, an
 * internal assertion helper, and per-datatype SIMD lane counts for
 * SSE, AVX/AVX2, and AVX-512F instruction sets.
 *
 * ## Assertion
 * NOVA_INTERNAL_ASSERT is a fatal assertion macro that prints a
 * formatted message to stderr and calls exit() on failure.  It
 * supports variadic format arguments.
 *
 * ## SIMD Lane Counts
 * Each NOVA_SIMD_* constant gives the number of elements of a given
 * datatype that fit in one vector register for the indicated ISA.
 * These are used for loop unrolling and vectorisation decisions.
 *
 * @see dtype.h Data type identifiers referenced by these constants.
 * @see simd.h Runtime SIMD capability detection.
 */

#pragma once

#include <stdalign.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// C99 `restrict` is not a C++ keyword. Some public headers use `restrict`
// in their API, so provide a portable definition when compiling as C++.
#if defined(__cplusplus) && !defined(restrict)
#define restrict __restrict__
#endif

/**
 * @brief Align a struct or variable to N bytes.
 *
 * Expands to __attribute__((aligned(N))).  Used to enforce
 * cache-line alignment (64 bytes) for tensor hot fields.
 */
#define ALIGN(N) __attribute__((aligned(N)))

/**
 * @brief Maximum number of tensor dimensions supported.
 */
#define NOVA_MAX_DIMS 64

/**
 * @brief Total number of supported data types.
 */
#define NUM_DTYPES 12

/**
 * @brief Number of floating-point data types (f32, f64, f16, bf16).
 */
#define NUM_FLOATS 4

/**
 * @brief Total number of integer data types (signed + unsigned).
 */
#define NUM_INTEGERS 8

/**
 * @brief Number of signed integer data types.
 */
#define NUM_SIGNED_INTEGERS 4

/**
 * @brief Number of unsigned integer data types.
 */
#define NUM_UNSIGNED_INTEGERS 4

/**
 * @brief Total number of quantised integer data types.
 */
#define NUM_QUANTIZED_INTEGERS 2

/**
 * @brief Number of signed quantised integer data types.
 */
#define NUM_QUANTIZED_SIGNED_INTEGERS 1

/**
 * @brief Number of unsigned quantised integer data types.
 */
#define NUM_QUANTIZED_UNSIGNED_INTEGERS 1

/**
 * @brief Number of supported backend implementations.
 */
#define NUM_BACKENDS 5

/**
 * @brief Apply a GCC/Clang type attribute.
 *
 * Expands to __attribute__((mode)).  Used for mode-based type
 * specifications (e.g. pointer-sized integers).
 */
#define ATTR(mode) __attribute__((mode))

/**
 * @brief Assertion macro for internal invariants.
 *
 * If the assertion evaluates to false, a formatted message is
 * written to stderr and the process exits with EXIT_FAILURE.
 * Supports printf-style variadic format arguments.
 *
 * @param assertion Boolean expression to test.
 * @param msg       Format string for the error message.
 * @param ...       Optional format arguments.
 */
#define NOVA_INTERNAL_ASSERT(assertion, msg, ...)                              \
  do {                                                                         \
    if (!(assertion)) {                                                        \
      fprintf(stderr, msg __VA_OPT__(, ) __VA_ARGS__);                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/** @name SIMD lane counts — SSE
 *  Number of elements per SSE vector register for each datatype.
 */
///@{
#define NOVA_SIMD_F32_WITH_SSE 4
#define NOVA_SIMD_F64_WITH_SSE 2
#define NOVA_SIMD_FP16_WITH_SSE 8
#define NOVA_SIMD_BF16_WITH_SSE 8
#define NOVA_SIMD_S8_WITH_SSE 16
#define NOVA_SIMD_U8_WITH_SSE 16
#define NOVA_SIMD_S32_WITH_SSE 4
#define NOVA_SIMD_U32_WITH_SSE 4
#define NOVA_SIMD_S64_WITH_SSE 2
#define NOVA_SIMD_U64_WITH_SSE 2
#define NOVA_SIMD_QS8_WITH_SSE 16
#define NOVA_SIMD_QU8_WITH_SSE 16
///@}

/** @name SIMD lane counts — AVX / AVX2
 *  Number of elements per AVX/AVX2 vector register for each datatype.
 */
///@{
#define NOVA_SIMD_F32_WITH_AVX_AVX2 8
#define NOVA_SIMD_F64_WITH_AVX_AVX2 4
#define NOVA_SIMD_FP16_WITH_AVX_AVX2 16
#define NOVA_SIMD_BF16_WITH_AVX_AVX2 16
#define NOVA_SIMD_S8_WITH_AVX_AVX2 32
#define NOVA_SIMD_U8_WITH_AVX_AVX2 32
#define NOVA_SIMD_S32_WITH_AVX_AVX2 8
#define NOVA_SIMD_U32_WITH_AVX_AVX2 8
#define NOVA_SIMD_S64_WITH_AVX_AVX2 4
#define NOVA_SIMD_U64_WITH_AVX_AVX2 4
#define NOVA_SIMD_QS8_WITH_AVX_AVX2 32
#define NOVA_SIMD_QU8_WITH_AVX_AVX2 32
///@}

/** @name SIMD lane counts — AVX-512F
 *  Number of elements per AVX-512F vector register for each datatype.
 */
///@{
#define NOVA_SIMD_F32_WITH_AVX512F 16
#define NOVA_SIMD_F64_WITH_AVX512F 8
#define NOVA_SIMD_FP16_WITH_AVX512F 32
#define NOVA_SIMD_BF16_WITH_AVX512F 32
#define NOVA_SIMD_S8_WITH_AVX512F 64
#define NOVA_SIMD_U8_WITH_AVX512F 64
#define NOVA_SIMD_S32_WITH_AVX512F 16
#define NOVA_SIMD_U32_WITH_AVX512F 16
#define NOVA_SIMD_S64_WITH_AVX512F 8
#define NOVA_SIMD_U64_WITH_AVX512F 8
#define NOVA_SIMD_QS8_WITH_AVX512F 64
#define NOVA_SIMD_QU8_WITH_AVX512F 64
///@}
