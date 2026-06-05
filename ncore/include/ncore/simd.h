#pragma once

/**
 * @file simd.h
 * @brief CPU SIMD capability detection and runtime kernel selection interface.
 *
 * @details
 * Provides runtime detection of CPU SIMD instruction sets and a unified
 * interface for selecting optimized kernels based on available hardware.
 * This header defines the @ref Capabilities_ structure and the
 * @ref get_cpu_capabilities() accessor for querying SIMD features.
 *
 * ## SIMD Support Tiers
 * The detection follows a hierarchical model:
 *
 * | Tier        | Feature      | Width   | Typical Use          |
 * |-------------|--------------|---------|----------------------|
 * | SSE4.2      | sse4_2_      | 128-bit | Basic vectorization  |
 * | AVX         | avx_         | 256-bit | Float ops            |
 * | AVX2        | avx2_        | 256-bit | Integer SIMD         |
 * | AVX-512     | avx512_*     | 512-bit | High-throughput      |
 * | AMX         | amx_*        | Tile    | Matrix ops           |
 *
 * ## Usage
 * @code{.c}
 * const Capabilities_ *caps = get_cpu_capabilities();
 * if (caps->avx2_) {
 *     // Use AVX2 optimized kernel
 * } else if (caps->sse4_2_) {
 *     // Use SSE4.2 fallback
 * }
 * @endcode
 *
 * ## Thread Safety
 * Detection is performed once on first call using call_once.
 * The returned pointer is safe to use from any thread.
 *
 * ## Detection Process
 * 1. get_cpu_capabilities() is called
 * 2. CPUID instruction queries available features
 * 3. Results are cached in global static
 * 4. Subsequent calls return cached result
 *
 * @note AVX-512 features are detected only if AVX-512F is available.
 *       VNNI flags (avx2_vnni_, avx512_vnni_) enable neural network
 * optimizations.
 *
 * @see DetectSIMD.cmake Compile-time SIMD flag detection
 * @see tensor.h Tensor structure using SIMD alignment
 * @see simd.c Implementation of runtime detection
 */

#include <stdbool.h>

/**
 * @struct Capabilities_
 * @brief CPU SIMD capabilities detected at runtime.
 *
 * @details
 * Contains boolean flags indicating which SIMD instruction sets are available
 * on the current processor. These flags are populated by the first call to
 * @ref get_cpu_capabilities() and remain read-only thereafter.
 *
 * Flags are organized into logical groups:
 * - **Base SIMD**: sse4_2_, avx_, avx2_, f16c_, fma3_
 * - **AVX-512 Family**: avx512f_, avx512_bw_, avx512_dq_, avx512_vl_,
 *   avx512_vnni_, avx512_fp16_, avx512_bf16_
 * - **AMX Tiles**: amx_, amx_fp16_, amx_bf16_, amx_int8_
 * - **Neural Network**: vnni_, avx2_vnni_, avx2_int8_
 *
 * @note Composite flags (amx_, vnni_) are OR of their constituent features.
 *
 * @see get_cpu_capabilities()
 * @see detect_cpu_capabilities_()
 */
typedef struct {
  bool sse4_2_; ///< Streaming SIMD Extensions 4.2 (128-bit integer/vector ops).
  bool avx_;    ///< Advanced Vector Extensions (256-bit floating-point ops).
  bool avx2_;   ///< AVX2 (256-bit integer SIMD operations).
  bool avx2_int8_; ///< AVX2 VNNI INT8 support (INT8 dot product instructions).
  bool avx2_vnni_; ///< AVX2 VNNI support (Vector Neural Network Instructions).
  bool f16c_;      ///< F16C (FP16 <-> FP32 conversion instructions).

  bool vnni_; ///< Vector Neural Network Instructions (any VNNI support
              ///< available).

  bool fma3_; ///< Fused Multiply-Add 3 (FMA3 instructions).

  bool avx512f_;     ///< AVX-512 Foundation (512-bit vector operations).
  bool avx512_bw_;   ///< AVX-512 Byte/Word (8-bit and 16-bit integer ops).
  bool avx512_dq_;   ///< AVX-512 Doubleword/Quadword (32-bit and 64-bit integer
                     ///< ops).
  bool avx512_vl_;   ///< AVX-512 Vector Length extensions (128/256-bit
                     ///< compatibility).
  bool avx512_vnni_; ///< AVX-512 VNNI (Vector Neural Network Instructions).
  bool avx512_fp16_; ///< AVX-512 FP16 (half-precision floating-point ops).
  bool avx512_bf16_; ///< AVX-512 BF16 (bfloat16 dot product instructions).

  bool amx_;      ///< Any AMX tile support (composite flag).
  bool amx_fp16_; ///< AMX FP16 tile support (half-precision matrix ops).
  bool amx_bf16_; ///< AMX BF16 tile support (bfloat16 matrix ops).
  bool amx_int8_; ///< AMX INT8 tile support (int8 matrix multiply).
} Capabilities_;

/**
 * @brief Get CPU SIMD capabilities (thread-safe, singleton accessor).
 *
 * @details
 * Returns a pointer to the global @ref Capabilities_ structure containing
 * all detected SIMD features. The first call triggers detection via
 * @ref detect_cpu_capabilities_(); subsequent calls return the cached result.
 *
 * @return Pointer to the global Capabilities_ structure.
 *         The returned pointer is valid for the lifetime of the process
 *         and must not be freed by the caller.
 *
 * @note This function is thread-safe. The detection is performed at most
 *       once using C23 call_once().
 *
 * @par Example:
 * @code{.c}
 * const Capabilities_ *caps = get_cpu_capabilities();
 * if (caps->avx2_) {
 *     // Use AVX2-optimized code path
 * } else if (caps->sse4_2_) {
 *     // Use SSE4.2 fallback
 * }
 * @endcode
 *
 * @see Capabilities_
 * @see detect_cpu_capabilities_()
 */
const Capabilities_ *get_cpu_capabilities();
