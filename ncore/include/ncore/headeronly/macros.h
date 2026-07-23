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
 * | C23 attributes      | `[[gnu::packed]]`, `[[gnu::constructor]]`, …       |
 * | Limits              | `NOVA_MAX_DIMS`, `NUM_DTYPES`, `NUM_BACKENDS`, …   |
 * | Assertion           | `NOVA_INTERNAL_ASSERT`                             |
 * | SIMD lane counts    | `NOVA_SIMD_*_WITH_SSE`, `…_WITH_AVX_AVX2`, `…_WITH_AVX512F` |
 * | Terminal colours    | `NCORE_LOG_PREFIX`, `NCORE_LOG_BOLD`, …            |
 * | CUDA/HIP qualifiers | `NCORE_HOST_DEVICE`, `NCORE_HOST`, `NCORE_DEVICE`  |
 * | Sanitizer           | `NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO`                |
 */
// clang-format on
/**
 * ## Design rules
 *
 * - **No function definitions** — This file is purely preprocessor
 *   constants and macros.
 * - **No dependencies** — Only standard C headers (`<stdio.h>`,
 *   `<stdlib.h>`, `<stdalign.h>`).
 * - **C and C++ compatible** — The `restrict` keyword and `fprintf()`
 *   function are mapped to `__restrict__` and `std::cerr <<` when
 *   compiling as C++.
 *
 * @see dtype.h   DType_ enum referenced by `NUM_DTYPES` etc.
 * @see simd.h    Runtime SIMD capability detection.
 * @see status.h  novaStatus_t enum referenced by `NUM_ERRORS`.
 */

#pragma once

#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <iostream>
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
#if defined(__cplusplus) && !defined(restrict)
#define restrict __restrict__
#endif

/**
 * @def ALIGN(N)
 * @brief Align a type or variable to @p N bytes.
 *
 * @details
 * Expands to `__attribute__((aligned(N)))` on GCC/Clang and
 * `[[align(N)]]` on MSVC.  Used extensively to enforce cache-line
 * alignment (64 bytes) for tensor hot fields and SIMD-friendly data structures.
 *
 * @param N  Alignment boundary in bytes.  Must be a power of two.
 *
 * @code{.c}
 * struct ALIGN(64) Tensor { ... };   // cache-line aligned
 * @endcode
 *
 * @see ATTR  Attribute macro.
 */

#ifdef _MSC_VER
#define ALIGN(N) [[align(N)]]
#elif defined(__clang__) || defined(__GNUC__)
#define ALIGN(N) __attribute__((aligned(N)))
#else
#define ALIGN(N) __attribute__((aligned(N)))
#endif

/**
 * @def ATTR(mode)
 * @brief Apply a GCC/Clang attribute.
 *
 * @details
 * Expands to `[[gnu::mode]]` on GCC/Clang using C23 attribute syntax,
 * or`[[mode]]` on MSVC. Used for all compiler-specific attributes throughout
 * the codebase, including:
 * - `ATTR(packed)` on packed enums
 * - `ATTR(constructor)` on init functions
 * - `ATTR(format(printf, k, n))` for printf-style format checks
 *
 * @param mode  Attribute name, optionally with arguments
 *              (e.g. `packed`, `constructor`, `format(printf, 2, 3)`).
 */
#ifdef _MSC_VER
#define ATTR(mode) [[mode]]
#elif defined(__clang__) || defined(__GNUC__)
#define ATTR(mode) [[gnu::mode]]
#else
#define ATTR(mode) __attribute__((mode))
#endif

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
 * Equals 21, covering Float32, Float64, Float16, BFloat16,
 * Float8E4M3fn, Float8E5M2, Float4E2M1fn, Signed8, UnSigned8,
 * QSigned8, QUnSigned8, Signed16, UnSigned16, QSigned16,
 * QUnSigned16, Signed32, UnSigned32, QSigned32, QUnSigned32,
 * Signed64, and UnSigned64.
 *
 * @see DType_ enum in dtype.h.
 */
#define NUM_DTYPES 21

/**
 * @def NUM_ERRORS
 * @brief Total number of type errors
 *
 */
#define NUM_ERRORS 33

/**
 * @def NUM_PARALLEL_GROUPS
 * @brief Total number of parallel groups
 * that have implemented their own thread pool.
 *
 */
#define NUM_PARALLEL_GROUPS 3

/**
 * @def NUM_FLOATS
 * @brief Number of floating-point data types (f32, f64, f16, bf16,
 *        fp8_e4m3fn, fp8_e5m2, fp4_e2m1fn_x2).
 */
#define NUM_FLOATS 7

/**
 * @def NUM_INTEGERS
 * @brief Total number of integer data types (signed + unsigned,
 *        including quantized).
 */
#define NUM_INTEGERS 14

/**
 * @def NUM_SIGNED_INTEGERS
 * @brief Number of signed integer data types (s8, s16, s32, s64,
 *        qs8, qs16, qs32).
 */
#define NUM_SIGNED_INTEGERS 7

/**
 * @def NUM_UNSIGNED_INTEGERS
 * @brief Number of unsigned integer data types (u8, u16, u32, u64,
 *        qu8, qu16, qu32).
 */
#define NUM_UNSIGNED_INTEGERS 7

/**
 * @def NUM_QUANTIZED_INTEGERS
 * @brief Total number of quantized integer data types.
 */
#define NUM_QUANTIZED_INTEGERS 6

/**
 * @def NUM_QUANTIZED_SIGNED_INTEGERS
 * @brief Number of signed quantized integer data types (qs8,
 *        qs16, qs32).
 */
#define NUM_QUANTIZED_SIGNED_INTEGERS 3

/**
 * @def NUM_QUANTIZED_UNSIGNED_INTEGERS
 * @brief Number of unsigned quantized integer data types (qu8,
 *        qu16, qu32).
 */
#define NUM_QUANTIZED_UNSIGNED_INTEGERS 3

/**
 * @def NUM_BACKENDS
 * @brief Number of supported compute backends.
 *
 * @details
 * Equals 7: CUDA, HIP, CPU, Meta, Miopen, OneDNN and Generic.
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
 * NOVA_INTERNAL_ASSERT(ptr != nullptr,
 *                      "[ALLOC] malloc returned nullptr\n");
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

#if defined(__CUDACC__) || defined(__HIPCC__)
/**
 * @def NCORE_HOST_DEVICE
 * @brief Qualifier for functions callable from both host and device.
 *
 * @details
 * Expands to `__host__ __device__` when compiling with CUDA or
 * HIP, otherwise empty.
 */
#define NCORE_HOST_DEVICE __host__ __device__
/**
 * @def NCORE_HOST
 * @brief Qualifier for host-only functions.
 *
 * @details
 * Expands to `__host__` when compiling with CUDA or HIP,
 * otherwise empty.
 */
#define NCORE_HOST __host__
/**
 * @def NCORE_DEVICE
 * @brief Qualifier for device-only functions.
 *
 * @details
 * Expands to `__device__` when compiling with CUDA or HIP,
 * otherwise empty.
 */
#define NCORE_DEVICE __device__
#else
#define NCORE_HOST_DEVICE
#define NCORE_HOST
#define NCORE_DEVICE
#endif

/**
 * @def NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
 * @brief Suppress UBSan float-divide-by-zero warnings.
 *
 * @details
 * Expands to `__attribute__((no_sanitize("float-divide-by-zero")))`
 * on Clang and GCC, otherwise empty.  Applied to division
 * operators in the dtype headers to suppress benign sanitizer
 * warnings on IEEE 754 division by zero (which yields +/-inf
 * rather than trapping).
 */
#if defined(__clang__)
#define NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO                                      \
  [[clang::no_sanitize("float-divide-by-zero")]]
#elif defined(__GNUC__)
#define NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO                                      \
  [[gnu::no_sanitize("float-divide-by-zero")]]
#else
#define NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO
#endif

/**
 * @def _GNUC_CLANG_
 * @brief Compiler detection macro for GNU/Clang-compatible toolchains.
 *
 * @details
 * Defined to `1` when the compiler is GCC (`__GNUC__`) or Clang (`__clang__`).
 * This macro is primarily used to dispatch between compiler-native
 * half-precision types (`_Float16`, `__bf16`) and the project's portable
 * soft-float implementations
 * (`float16`, `bfloat16`).
 *
 * @note
 * This macro acts as a feature-test for compiler extensions that allow
 * direct arithmetic on 16-bit floating-point types without library support.
 *
 * @code{.c}
 * #ifdef _GNUC_CLANG_
 *   // Use _Float16 with native casting
 *   float val = (float)half_value;
 * #else
 *   // Use soft-float conversion
 *   float val = fp16_to_float(half_value);
 * #endif
 * @endcode
 */
#if defined(__GNUC__) || defined(__clang__)
#define _GNUC_CLANG_ 1
#endif
