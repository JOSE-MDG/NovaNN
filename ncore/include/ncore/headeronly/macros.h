/**
 * @file macros.h
 * @brief Core macros, constants, and SIMD lane-count definitions.
 *
 * @details
 * Foundational preprocessor definitions used throughout the entire
 * NovaNN codebase.  Every public and internal header includes this
 * file.
 *
 * ## Contents
 *
 */
// clang-format off
/**
 * | Category            | Symbols                                            |
 * |---------------------|----------------------------------------------------|
 * | Portability         | `restrict`, `ALIGN`, `ATTR`                        |
 * | Limits              | `NOVA_MAX_DIMS`, `NUM_DTYPES`, `NUM_BACKENDS`, …   |
 * | Assertion           | `NOVA_INTERNAL_ASSERT`                             |
 * | SIMD lane counts    | `NOVA_SIMD_*_WITH_SSE`, `…_WITH_AVX_AVX2`, `…_WITH_AVX512F` |
 * | Terminal colours    | `NCORE_LOG_PREFIX`, `NCORE_LOG_BOLD`, …            |
 */
// clang-format on
/**
 * ## Design rules
 *
 * - **No function definitions** — This file is purely preprocessor
 *   constants and macros.
 * - **No dependencies** — Only standard C headers (`<stdio.h>`,
 *   `<stdlib.h>`, `<stdbool.h>`, `<stdalign.h>`).
 * - **C and C++ compatible** — The `restrict` keyword and fprintf() function
 *    were mapped to `__restrict__` and std::cerr << ... when compiling as C++.
 *
 * @see dtype.h   DType_ enum referenced by `NUM_DTYPES` etc.
 * @see simd.h    Runtime SIMD capability detection.
 * @see status.h  novaStatus_t enum referenced by `NUM_ERRORS`.
 */

#pragma once

#include <stdalign.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <iostream>
#endif

#if defined(__cplusplus) && !defined(restrict)
#define restrict __restrict__
#endif

/**
 * @def restrict
 * @brief Portable `restrict` qualifier for C++ compatibility.
 *
 * @details
 * C99 `restrict` is not a keyword in C++.  When compiling as C++
 * (detected via `__cplusplus`), this maps `restrict` to the
 * compiler-specific `__restrict__` extension.  In C mode the
 * standard `restrict` is used unchanged.
 *
 * @see ALIGN(N)   Another portability macro.
 */

/**
 * @def ALIGN(N)
 * @brief Align a type or variable to @p N bytes.
 *
 * @details
 * Expands to `__attribute__((aligned(N)))` on GCC/Clang.  Used
 * extensively to enforce cache-line alignment (64 bytes) for
 * tensor hot fields and SIMD-friendly data structures.
 *
 * @param N  Alignment boundary in bytes.  Must be a power of two.
 *
 * @code{.c}
 * struct ALIGN(64) Tensor { ... };   // cache-line aligned
 * @endcode
 *
 * @see ATTR(mode)  Mode-based type attribute.
 */
#define ALIGN(N) __attribute__((aligned(N)))

/**
 * @def ATTR(mode)
 * @brief Apply a GCC/Clang type attribute.
 *
 * @details
 * Expands to `__attribute__((mode))`.  Used for mode-based type
 * specifications such as pointer-sized integers.
 *
 * @param mode  Attribute mode name (e.g., `SI` for 32-bit int).
 */
#define ATTR(mode) __attribute__((mode))

/**
 * @def NOVA_MAX_DIMS
 * @brief Maximum number of tensor dimensions supported.
 *
 * @details
 * Fixed at 64 — well beyond the typical 4–8 dimensions used in
 * deep learning.  Used to size the `shape_t` and `strides_t`
 * fixed arrays in @ref Tensor.
 *
 * @see shape_t   Fixed-size shape array.
 * @see strides_t Fixed-size strides array.
 */
#define NOVA_MAX_DIMS 64

/**
 * @def NUM_DTYPES
 * @brief Total number of supported data types.
 *
 * @details
 * Equals 12, covering Float32, Float64, Float16, BFloat16,
 * Signed8, UnSigned8, QSigned8, QUnSigned8, Signed32,
 * UnSigned32, Signed64, and UnSigned64.
 *
 * @see DType_ enum in dtype.h.
 */
#define NUM_DTYPES 12

/**
 * @def NUM_ERRORS
 * @brief Total number of type errors
 *
 */
#define NUM_ERRORS 31

/**
 * @def NUM_FLOATS
 * @brief Number of floating-point data types (f32, f64, f16, bf16).
 */
#define NUM_FLOATS 4

/**
 * @def NUM_INTEGERS
 * @brief Total number of integer data types (signed + unsigned,
 *        including quantised).
 */
#define NUM_INTEGERS 8

/**
 * @def NUM_SIGNED_INTEGERS
 * @brief Number of signed integer data types (s8, s32, s64,
 *        qs8).
 */
#define NUM_SIGNED_INTEGERS 4

/**
 * @def NUM_UNSIGNED_INTEGERS
 * @brief Number of unsigned integer data types (u8, u32, u64,
 *        qu8).
 */
#define NUM_UNSIGNED_INTEGERS 4

/**
 * @def NUM_QUANTIZED_INTEGERS
 * @brief Total number of quantised integer data types.
 */
#define NUM_QUANTIZED_INTEGERS 2

/**
 * @def NUM_QUANTIZED_SIGNED_INTEGERS
 * @brief Number of signed quantised integer data types (qs8).
 */
#define NUM_QUANTIZED_SIGNED_INTEGERS 1

/**
 * @def NUM_QUANTIZED_UNSIGNED_INTEGERS
 * @brief Number of unsigned quantised integer data types (qu8).
 */
#define NUM_QUANTIZED_UNSIGNED_INTEGERS 1

/**
 * @def NUM_BACKENDS
 * @brief Number of supported compute backends.
 *
 * @details
 * Equals 5: CUDA, HIP, CPU, Meta, Miopen, OneDNN and Generic.
 *
 * @see Backend enum in backend.h.
 */
#define NUM_BACKENDS 7

/**
 * @def NOVA_INTERNAL_ASSERT(assertion, msg, ...)
 * @brief Fatal assertion for internal invariants.
 *
 * @details
 * If @p assertion evaluates to false, a formatted message is
 * written to `stderr` and the process exits with
 * `EXIT_FAILURE`.  Uses `__VA_OPT__` for clean expansion when
 * no variadic arguments are provided.
 *
 * This macro is for debugging aid — it guards
 * invariants that, if violated, indicate a bug in NovaNN
 * itself (e.g., null storage after a successful allocation).
 *
 * @param assertion  Boolean expression to test.
 * @param msg        `printf`-style format string for the error
 *                   message.
 * @param ...        Optional format arguments.
 *
 * @code{.c}
 * NOVA_INTERNAL_ASSERT(ptr != NULL,
 *                      "[ALLOC] malloc returned NULL\n");
 * @endcode
 *
 * @note The message should include a module tag in brackets
 *       (e.g., `[STORAGE]`, `[CUDA]`) for easy identification.
 */
#ifdef __cplusplus
#define NOVA_INTERNAL_ASSERT(assertion, msg, ...)                              \
  do {                                                                         \
    if (!(assertion)) {                                                        \
      std::cerr << msg __VA_OPT__(<< __VA_ARGS__);                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#else
#define NOVA_INTERNAL_ASSERT(assertion, msg, ...)                              \
  do {                                                                         \
    if (!(assertion)) {                                                        \
      fprintf(stderr, msg __VA_OPT__(, ) __VA_ARGS__);                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

/**
 * @defgroup SIMD_LANES SIMD lane counts
 * @{
 * @brief Number of elements of each datatype that fit in one SIMD
 *        vector register for a given ISA.
 *
 * @details
 * These constants are used for:
 * - Loop unrolling and vectorisation factor selection.
 * - Buffer sizing for SIMD-aligned temporary storage.
 * - Compile-time assertions that tensor sizes are multiples of
 *   the vector width.
 *
 * The naming convention is `NOVA_SIMD_{TYPE}_WITH_{ISA}`.
 */

/** @name SIMD lane counts — SSE
 *  128-bit register width.
 */
///@{
#define NOVA_SIMD_F32_WITH_SSE 4  ///< float32:   128 / 32 = 4
#define NOVA_SIMD_F64_WITH_SSE 2  ///< float64:   128 / 64 = 2
#define NOVA_SIMD_FP16_WITH_SSE 8 ///< float16:   128 / 16 = 8
#define NOVA_SIMD_BF16_WITH_SSE 8 ///< bfloat16:  128 / 16 = 8
#define NOVA_SIMD_S8_WITH_SSE 16  ///< int8:      128 / 8  = 16
#define NOVA_SIMD_U8_WITH_SSE 16  ///< uint8:     128 / 8  = 16
#define NOVA_SIMD_S32_WITH_SSE 4  ///< int32:     128 / 32 = 4
#define NOVA_SIMD_U32_WITH_SSE 4  ///< uint32:    128 / 32 = 4
#define NOVA_SIMD_S64_WITH_SSE 2  ///< int64:     128 / 64 = 2
#define NOVA_SIMD_U64_WITH_SSE 2  ///< uint64:    128 / 64 = 2
#define NOVA_SIMD_QS8_WITH_SSE 16 ///< qint8:     128 / 8  = 16
#define NOVA_SIMD_QU8_WITH_SSE 16 ///< quint8:    128 / 8  = 16
///@}

/** @name SIMD lane counts — AVX / AVX2
 *  256-bit register width.
 */
///@{
#define NOVA_SIMD_F32_WITH_AVX_AVX2 8   ///< float32:   256 / 32 = 8
#define NOVA_SIMD_F64_WITH_AVX_AVX2 4   ///< float64:   256 / 64 = 4
#define NOVA_SIMD_FP16_WITH_AVX_AVX2 16 ///< float16:  256 / 16 = 16
#define NOVA_SIMD_BF16_WITH_AVX_AVX2 16 ///< bfloat16: 256 / 16 = 16
#define NOVA_SIMD_S8_WITH_AVX_AVX2 32   ///< int8:      256 / 8  = 32
#define NOVA_SIMD_U8_WITH_AVX_AVX2 32   ///< uint8:     256 / 8  = 32
#define NOVA_SIMD_S32_WITH_AVX_AVX2 8   ///< int32:     256 / 32 = 8
#define NOVA_SIMD_U32_WITH_AVX_AVX2 8   ///< uint32:    256 / 32 = 8
#define NOVA_SIMD_S64_WITH_AVX_AVX2 4   ///< int64:     256 / 64 = 4
#define NOVA_SIMD_U64_WITH_AVX_AVX2 4   ///< uint64:    256 / 64 = 4
#define NOVA_SIMD_QS8_WITH_AVX_AVX2 32  ///< qint8:     256 / 8  = 32
#define NOVA_SIMD_QU8_WITH_AVX_AVX2 32  ///< quint8:    256 / 8  = 32
///@}

/** @name SIMD lane counts — AVX-512F
 *  512-bit register width.
 */
///@{
#define NOVA_SIMD_F32_WITH_AVX512F 16  ///< float32:   512 / 32 = 16
#define NOVA_SIMD_F64_WITH_AVX512F 8   ///< float64:   512 / 64 = 8
#define NOVA_SIMD_FP16_WITH_AVX512F 32 ///< float16:   512 / 16 = 32
#define NOVA_SIMD_BF16_WITH_AVX512F 32 ///< bfloat16:  512 / 16 = 32
#define NOVA_SIMD_S8_WITH_AVX512F 64   ///< int8:      512 / 8  = 64
#define NOVA_SIMD_U8_WITH_AVX512F 64   ///< uint8:     512 / 8  = 64
#define NOVA_SIMD_S32_WITH_AVX512F 16  ///< int32:     512 / 32 = 16
#define NOVA_SIMD_U32_WITH_AVX512F 16  ///< uint32:    512 / 32 = 16
#define NOVA_SIMD_S64_WITH_AVX512F 8   ///< int64:     512 / 64 = 8
#define NOVA_SIMD_U64_WITH_AVX512F 8   ///< uint64:    512 / 64 = 8
#define NOVA_SIMD_QS8_WITH_AVX512F 64  ///< qint8:     512 / 8  = 64
#define NOVA_SIMD_QU8_WITH_AVX512F 64  ///< quint8:    512 / 8  = 64
///@}

/** @} */

/**
 * @defgroup LOG_COLOURS Terminal colour codes
 * @{
 * @brief ANSI escape sequences for subtle, cmake-style terminal
 *        output.
 *
 * @details
 * Used by `print_*` functions in `device.c`, `alloc.c`, etc.
 * The colour palette is intentionally muted:
 * - **Green prefix** (`--`): status messages.
 * - **Cyan values**: highlighted data (device names, sizes).
 * - **Bold**: section headings or emphasis.
 * - **Reset**: restores default terminal colour.
 */

/**
 * @def NCORE_LOG_PREFIX
 * @brief Green `--` prefix for log messages.
 *
 * @code{.c}
 * printf(NCORE_LOG_PREFIX " Detecting CUDA devices\n");
 * // Output: -- Detecting CUDA devices  (green --)
 * @endcode
 */
#define NCORE_LOG_PREFIX "\033[32m--\033[0m"

/**
 * @def NCORE_LOG_BOLD
 * @brief Bold text start.  Must be paired with
 *        @ref NCORE_LOG_RESET.
 */
#define NCORE_LOG_BOLD "\033[1m"

/**
 * @def NCORE_LOG_VALUE
 * @brief Cyan colour for highlighted values.
 */
#define NCORE_LOG_VALUE "\033[36m"

/**
 * @def NCORE_LOG_RESET
 * @brief Reset to default terminal colour.
 */
#define NCORE_LOG_RESET "\033[0m"

/** @} */
