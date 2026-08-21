/**
 * @file macros.h
 * @brief Foundational preprocessor macros and compile-time constants.
 *
 * @details
 * Every public and internal header includes this file, so it must stay
 * free of function definitions and external dependencies. It provides
 * portability helpers (@c restrict, @c ALIGN, @c ATTR, @c INITIALIZE),
 * compile-time constants (@c NOVA_MAX_DIMS, @c NUM_DTYPES, @c NUM_FLOATS,
 * @c NUM_INTEGERS, @c NUM_ERRORS, @c NUM_BACKENDS), the
 * @c NOVA_INTERNAL_ASSERT diagnostic, SIMD lane counts (@c NOVA_SIMD_*),
 * terminal colour codes (@c NCORE_LOG_*), CUDA/HIP qualifiers
 * (@c NCORE_HOST_DEVICE, @c NCORE_HOST, @c NCORE_DEVICE) and compiler
 * detection macros (@c _GNUC_CLANG_, @c NCORE_UBSAN_IGNORE_DIVIDE_BY_ZERO).
 *
 * Three design rules keep this header usable everywhere:
 *
 * @li No function definitions — purely preprocessor constants and macros.
 * @li No dependencies — only standard C headers (@c stdio.h, @c stdlib.h,
 *     @c stdalign.h).
 * @li C and C++ compatible — @c restrict maps to @c __restrict__, and the
 *     assertion macro uses @c fprintf(stderr, ...) in C and @c std::cerr
 *     in C++.
 *
 * @see dtype.h   DType_ enum referenced by the @c NUM_DTYPES family.
 * @see simd.h    Runtime SIMD capability detection.
 * @see status.h  novaError_t enum referenced by @c NUM_ERRORS.
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
 * @brief Portable @c restrict qualifier for C++ compatibility.
 *
 * @details
 * C99 @c restrict is not a keyword in C++.  When compiling as C++
 * (detected via @c __cplusplus), this maps @c restrict to the
 * compiler-specific @c __restrict__ extension.  In C mode the
 * standard @c restrict is used unchanged.
 *
 * @see ALIGN(N)   Another portability macro.
 */
#if defined(__cplusplus) && !defined(restrict)
#define restrict __restrict__
#endif

/**
 * @def ALIGN(N)
 * @brief Align a type or variable to N bytes.
 *
 * @param N  Alignment boundary in bytes.  Must be a power of two.
 */
#define ALIGN(N) __attribute__((aligned(N)))

/**
 * @def ATTR(mode)
 * @brief Apply a GCC/Clang attribute.
 *
 * @details
 * Expands to @c [[gnu::mode]] on GCC/Clang using C23 attribute syntax,
 * Used for all compiler-specific attributes throughout the codebase,
 * including:
 * @li @c INITIALIZE(f) for constructor-like init functions
 * @li @c ATTR(format(printf, k, n)) for printf-style format checks
 *
 * @param mode  Attribute name, optionally with arguments
 *              (e.g. @c packed, @c constructor, @c format(printf, 2, 3)).
 */
#define ATTR(mode) [[gnu::mode]]

/**
 * @def INITIALIZE(f)
 * @brief Register a function to run automatically at library initialisation
 * time.
 *
 * @details
 * On GCC/Clang, expands to @c ATTR(constructor) static inline void f(void),
 * which causes the compiler to emit a @c .init_array entry that runs @p f
 * at library initialisation time.
 *
 * @param f  Name of the @c static inline function to register.  The function
 *           must take no arguments and return @c void.
 *
 * @note
 * The function body must follow immediately after the macro invocation.
 * Unlike a bare @c ATTR(constructor), the macro always defines the function
 * as @c static inline, so it is not visible outside the translation unit.
 *
 * @par Example
 * @code{.c}
 * INITIALIZE(init_my_table) {
 *   my_table[0] = "hello";
 * }
 * @endcode
 *
 * @warning
 * The init-order of multiple @c INITIALIZE calls across translation units is
 * not guaranteed.  Do not assume that @p f from file A runs before or
 * after @p f from file B.
 *
 * @see ATTR
 */
#define INITIALIZE(f) ATTR(constructor) static inline void f(void)

/**
 * @def NOVA_MAX_DIMS
 * @brief Maximum number of tensor dimensions supported.
 *
 * @details
 * Fixed at 64 — well beyond the typical 4–8 dimensions used in
 * deep learning.  Used to size the @c shape_t and @c strides_t
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
 * @brief Total number of @ref novaError_t enumerators.
 *
 * @details
 * Counts @c novaSuccess plus every error code across all categories
 * (parameters, memory, transfers, device/backend, OS, dtype/cast, GPU,
 * internal and general). Use it to size arrays indexed by error code.
 *
 * @see novaError_t in status.h.
 */
#define NUM_ERRORS 36

/**
 * @def NUM_PARALLEL_GROUPS
 * @brief Number of parallel subsystems that implement their own thread
 *        pool.
 *
 * @details
 * Used to size the registries of the parallel execution backends.
 */
#define NUM_PARALLEL_GROUPS 3

/**
 * @def NUM_FLOATS
 * @brief Number of floating-point data types (f32, f64, f16, bf16,
 *        fp8_e4m3fn, fp8_e5m2, fp4_e2m1fn_x2).
 *
 * @details
 * Complements @ref NUM_INTEGERS: @c NUM_FLOATS + @c NUM_INTEGERS ==
 * @c NUM_DTYPES (7 + 14 = 21).
 */
#define NUM_FLOATS 7

/**
 * @def NUM_INTEGERS
 * @brief Number of integer data types (signed + unsigned, including
 *        quantized).
 *
 * @details
 * The quantized types are counted inside their signed and unsigned
 * groups, not as a separate category. Satisfies @c NUM_INTEGERS ==
 * @c NUM_SIGNED_INTEGERS + @c NUM_UNSIGNED_INTEGERS (7 + 7 = 14).
 *
 * @see NUM_SIGNED_INTEGERS
 * @see NUM_UNSIGNED_INTEGERS
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
 * @brief Number of quantized integer data types (qs8, qs16, qs32,
 *        qu8, qu16, qu32).
 *
 * @details
 * Satisfies @c NUM_QUANTIZED_INTEGERS ==
 * @c NUM_QUANTIZED_SIGNED_INTEGERS + @c NUM_QUANTIZED_UNSIGNED_INTEGERS
 * (3 + 3 = 6).
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
 * @see Backend_ enum in backend.h.
 */
#define NUM_BACKENDS 7

/**
 * @def NOVA_INTERNAL_ASSERT(assertion, msg, ...)
 * @brief Fatal assertion for internal invariants.
 *
 * @details
 * If @p assertion evaluates to false, a formatted message is
 * written to @c stderr and the process exits with
 * @c EXIT_FAILURE.  Uses @c __VA_OPT__ for clean expansion when
 * no variadic arguments are provided.
 *
 * This macro is for debugging aid — it guards
 * invariants that, if violated, indicate a bug in NovaNN
 * itself (e.g., null storage after a successful allocation).
 *
 * @param assertion  Boolean expression to test.
 * @param msg        @c printf-style format string for the error
 *                   message.
 * @param ...        Optional format arguments.
 *
 * @code{.c}
 * NOVA_INTERNAL_ASSERT(ptr != nullptr,
 *                      "[ALLOC] malloc returned nullptr\n");
 * @endcode
 *
 * @note The message should include a module tag in brackets
 *       (e.g., @c [STORAGE], @c [CUDA]) for easy identification.
 */
#ifdef __cplusplus
[[deprecated("NOVA_INTERNAL_ASSERT is deprecated")]]
inline void nova_internal_assert_deprecated_marker() {}

#define NOVA_INTERNAL_ASSERT(assertion, msg, ...)                              \
  do {                                                                         \
    ::nova_internal_assert_deprecated_marker();                                \
    if (!(assertion)) {                                                        \
      std::cerr << msg __VA_OPT__(<< __VA_ARGS__);                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#else
#define NOVA_INTERNAL_ASSERT(assertion, msg, ...)                              \
  do {                                                                         \
    _Pragma("GCC warning \"NOVA_INTERNAL_ASSERT is deprecated\"") if (         \
        !(assertion)) {                                                        \
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
 * @li Loop unrolling and vectorization factor selection.
 * @li Buffer sizing for SIMD-aligned temporary storage.
 * @li Compile-time assertions that tensor sizes are multiples of
 *   the vector width.
 *
 * The naming convention is @c NOVA_SIMD_{TYPE}_WITH_{ISA}.
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
 * Used by @c print_* functions in @c device.c, @c alloc.c, etc.
 * The colour palette is intentionally muted:
 * @li Green prefix (@c --): status messages.
 * @li Cyan values: highlighted data (device names, sizes).
 * @li Bold: section headings or emphasis.
 * @li Reset: restores default terminal colour.
 */

/**
 * @def NCORE_LOG_PREFIX
 * @brief Green @c -- prefix for log messages.
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
 * Expands to @c __host__ @c __device__ when compiling with CUDA or
 * HIP, otherwise empty.
 */
#define NCORE_HOST_DEVICE __host__ __device__
/**
 * @def NCORE_HOST
 * @brief Qualifier for host-only functions.
 *
 * @details
 * Expands to @c __host__ when compiling with CUDA or HIP,
 * otherwise empty.
 */
#define NCORE_HOST __host__
/**
 * @def NCORE_DEVICE
 * @brief Qualifier for device-only functions.
 *
 * @details
 * Expands to @c __device__ when compiling with CUDA or HIP,
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
 * Expands to @c [[{clang,gnu}::no_sanitize("float-divide-by-zero")]]
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
 * Defined to @c 1 when the compiler is GCC (@c __GNUC__) or Clang (@c
 * __clang__). This macro is primarily used to dispatch between compiler-native
 * half-precision types (@c _Float16, @c __bf16) and the project's portable
 * soft-float implementations
 * (@c float16, @c bfloat16).
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
