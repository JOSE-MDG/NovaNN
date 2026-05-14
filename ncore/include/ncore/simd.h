#pragma once

/**
 * @file simd.h
 * @brief CPU SIMD capability detection and runtime selection.
 *
 * @details
 * Provides runtime detection of CPU SIMD instruction sets and a unified
 * interface for selecting optimized kernels based on available hardware.
 *
 * ## SIMD Support Tiers
 * The detection follows a hierarchical model:
 *
 * | Tier | Feature | Width | Typical Use |
 * |------|---------|-------|--------------|
 * | SSE4.2 | sse4_2 | 128-bit | Basic vectorization |
 * | AVX | avx | 256-bit | Float ops |
 * | AVX2 | avx2 | 256-bit | Integer SIMD |
 * | AVX-512 | avx512* | 512-bit | High-throughput |
 * | AMX | amx_* | Tile | Matrix ops |
 *
 * ## Usage
 * @code
 * const Capabilities_ *caps = get_cpu_capabilities();
 * if (caps->avx2_) {
 *   // Use AVX2 optimized kernel
 * } else if (caps->sse4_2_) {
 *   // Use SSE4.2 fallback
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
 * VNNI flags (avx2_vnni, avx512_vnni) enable neural network optimizations.
 *
 * @see DetectSIMD.cmake Compile-time SIMD flag detection
 * @see tensor.h Tensor structure using SIMD alignment
 */

#include <stdbool.h>

/**
 * @brief CPU SIMD capabilities detected at runtime.
 *
 * Contains flags indicating which SIMD instruction sets are available
 * on the current processor. These are used to select optimized kernels.
 */
typedef struct {
  bool sse4_2_;    ///< Streaming SIMD Extensions 4.2
  bool avx_;       ///< Advanced Vector Extensions
  bool avx2_;      ///< AVX2 (256-bit integer SIMD)
  bool avx2_int8_; ///< AVX2 VNNI INT8 support
  bool avx2_vnni_; ///< AVX2 VNNI support
  bool f16c_;      ///< F16C (FP16 conversion)

  bool vnni_;      ///< Vector Neural Network Instructions (any)

  bool fma3_;      ///< Fused Multiply-Add 3

  bool avx512f_;       ///< AVX-512 Foundation
  bool avx512_bw_;     ///< AVX-512 Byte/Word
  bool avx512_dq_;     ///< AVX-512 Doubleword/Quadword
  bool avx512_vl_;     ///< AVX-512 Vector Length
  bool avx512_vnni_;   ///< AVX-512 VNNI
  bool avx512_fp16_;   ///< AVX-512 FP16
  bool avx512_bf16_;   ///< AVX-512 BF16

  bool amx_;       ///< Any AMX tile support
  bool amx_fp16_;  ///< AMX FP16 tile support
  bool amx_bf16_;  ///< AMX BF16 tile support
  bool amx_int8_;  ///< AMX INT8 tile support
} Capabilities_;

/**
 * @brief Get CPU SIMD capabilities (thread-safe, singleton).
 * @return Pointer to global Capabilities_ structure.
 *
 * On first call, detects and caches CPU capabilities. Subsequent calls
 * return the cached result. Thread-safe via call_once.
 */
const Capabilities_ *get_cpu_capabilities();
