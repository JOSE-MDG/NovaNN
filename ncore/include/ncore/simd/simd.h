/**
 * @file simd.h
 * @brief CPU SIMD capability detection and runtime kernel selection interface.
 *
 * @details
 * Provides runtime detection of CPU SIMD instruction sets and a unified
 * interface for selecting optimized kernels based on available hardware.
 * This header defines the @ref SIMDCapabilities structure and the
 * @ref get_simd_capabilities() accessor for querying SIMD features.
 *
 * @section simd-support-tiers SIMD Support Tiers
 * The detection follows a hierarchical model:
 *
 * @li SSE4.2 — sse4_2_ — 128-bit — Basic vectorization
 * @li AVX — avx_ — 256-bit — Float ops
 * @li AVX2 — avx2_ — 256-bit — Integer SIMD
 * @li AVX-512 — avx512_* — 512-bit — High-throughput
 * @li AMX — amx_* — Tile — Matrix ops
 *
 * @section usage Usage
 * @code{.c}
 * const SIMDCapabilities *simd = get_simd_capabilities();
 * if (simd->avx512f_) {
 *     // Use AVX-512 optimized kernel
 * } else if (simd->avx2_) {
 *     // Use SSE4.2 fallback
 * }
 * @endcode
 *
 * @section thread-safety Thread Safety
 * Detection is performed once on first call.
 * The returned pointer is safe to use from any thread.
 *
 * @section detection-process Detection Process
 * @li 1. get_simd_capabilities() is called
 * @li 2. CPUID instruction queries available features
 * @li 3. Results are cached in global static
 * @li 4. Subsequent calls return cached result
 *
 * @note Every flag reflects exactly its own CPUID bit; no cross-feature
 *       gating is applied. VNNI flags (avx2_vnni_, avx512_vnni_) enable
 *       neural network optimizations.
 *
 * @see DetectSIMD.cmake Compile-time SIMD flag detection
 * @see tensor.h Tensor structure using SIMD alignment
 * @see simd.c Implementation of runtime detection
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * @struct SIMDCapabilities
 * @brief CPU SIMD capabilities detected at runtime.
 *
 * @details
 * Contains boolean flags indicating which SIMD instruction sets are available
 * on the current processor. These flags are populated by the first call to
 * @ref get_simd_capabilities() and remain read-only thereafter.
 *
 * Flags are organized into logical groups:
 * @li Base SIMD: sse4_2_, avx_, avx2_, f16c_, fma3_
 * @li AVX-512 Family: avx512f_, avx512_bw_, avx512_dq_, avx512_vl_,
 *   avx512_vnni_, avx512_fp16_, avx512_bf16_
 * @li AMX Tiles: amx_, amx_fp16_, amx_bf16_, amx_int8_
 * @li Neural Network: vnni_, avx2_vnni_, avx2_int8_
 *
 * @note Composite flags (amx_, vnni_) are OR of their constituent features.
 *
 * @see get_simd_capabilities()
 * @see detect_simd_capabilities()
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

  bool avx10_1_; ///< AVX-10.1 (converged AVX-512 baseline; version >= 1).
  bool avx10_2_; ///< AVX-10.2 (mandatory 512-bit width; version >= 2).

  bool amx_;      ///< Any AMX tile support (composite flag).
  bool amx_fp16_; ///< AMX FP16 tile support (half-precision matrix ops).
  bool amx_bf16_; ///< AMX BF16 tile support (bfloat16 matrix ops).
  bool amx_int8_; ///< AMX INT8 tile support (int8 matrix multiply).
} SIMDCapabilities;

/**
 * @struct SIMDCpuidSnapshot
 * @brief Raw CPUID register values consumed by
 *        @ref get_simd_capabilities_from_cpuid().
 *
 * @details
 * Each member holds the four output registers of one CPUID query in
 * hardware order:
 *
 * @li Index 0 — EAX
 * @li Index 1 — EBX
 * @li Index 2 — ECX
 * @li Index 3 — EDX
 *
 * The snapshot decouples the pure bit-to-flag mapping (unit-testable with
 * synthetic snapshots of processors that are not physically available)
 * from the platform-specific register reads performed by
 * @ref get_simd_capabilities().
 *
 * @see get_simd_capabilities_from_cpuid()
 */
typedef struct {
  uint32_t leaf1[4];    ///< Output registers of CPUID.(EAX=01H).
  uint32_t leaf7_0[4];  ///< Output registers of CPUID.(EAX=07H,ECX=00H).
  uint32_t leaf7_1[4];  ///< Output registers of CPUID.(EAX=07H,ECX=01H).
  uint32_t leaf24_0[4]; ///< Output registers of CPUID.(EAX=24H,ECX=00H).
} SIMDCpuidSnapshot;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get CPU capabilities (thread-safe singleton accessor).
 *
 * @details
 * Returns a pointer to the global @ref SIMDCapabilities structure containing
 * all detected SIMD features. The first call triggers detection via
 * @ref detect_simd_capabilities(); subsequent calls return the cached result.
 *
 * @return Pointer to the global SIMDCapabilities structure.
 *         The returned pointer is valid for the lifetime of the process
 *         and must not be freed by the caller.
 *
 * @note This function is thread-safe. The detection is performed at most
 *       once using C11 @c call_once on Linux or @c InitOnceExecuteOnce on
 *       Windows.
 *
 * @par Example:
 * @code{.c}
 *   const SIMDCapabilities *simd = get_simd_capabilities();
 *   if (simd->avx2_) {
 *       // Use AVX2-optimized code path
 *   } else if (simd->sse4_2_) {
 *       // Use SSE4.2 fallback
 *   }
 * @endcode
 *
 * @see SIMDCapabilities
 * @see detect_simd_capabilities()
 * @see init_once()
 */
const SIMDCapabilities *get_simd_capabilities();

/**
 * @brief Maps raw CPUID registers to SIMD capabilities.
 *
 * @details
 * Pure, deterministic translation of a @ref SIMDCpuidSnapshot into a
 * @ref SIMDCapabilities structure. It performs no hardware access, which
 * makes it directly unit-testable against synthetic snapshots of
 * processors that are not physically available to the build host.
 *
 * The mapping follows the Intel SDM layout:
 *
 * @li Leaf 1 ECX bits feed sse4_2_, fma3_, avx_ and f16c_.
 * @li Leaf 7 subleaf 0 feeds avx2_, avx512f_, avx512_dq_, avx512_bw_,
 *     avx512_vl_, avx512_vnni_, amx_bf16_, avx512_fp16_ and amx_int8_.
 * @li Leaf 7 subleaf 1 feeds avx2_vnni_, avx512_bf16_, amx_fp16_ and
 *     avx2_int8_; its EDX bit 19 carries the AVX10 presence flag.
 * @li Leaf 24H EBX[7:0] holds the AVX10 version number; it is consulted
 *     only when the AVX10 presence flag is set, since the leaf is
 *     architecturally undefined otherwise. avx10_1_ is set for a version
 *     of at least 1 and avx10_2_ for a version of at least 2.
 * @li Composite flags: amx_ = OR(amx_bf16_, amx_fp16_, amx_int8_) and
 *     vnni_ = OR(avx512_vnni_, avx2_vnni_).
 *
 * No cross-feature gating is applied: every flag is set exactly when its
 * source bit is set (e.g. avx512_fp16_ does not require avx512f_).
 *
 * @param[in]     snapshot  Raw CPUID registers. Must not be null.
 * @param[in,out] caps      Destination capabilities. Must not be null;
 *                          every flag is overwritten.
 *
 * @pre snapshot must point to a valid SIMDCpuidSnapshot structure.
 * @pre caps must point to a valid SIMDCapabilities structure.
 * @post All capability flags in caps reflect the bits in snapshot; the
 *       amx_ and vnni_ composite flags equal the OR of their constituents.
 *
 * @see get_simd_capabilities()
 */
void get_simd_capabilities_from_cpuid(const SIMDCpuidSnapshot *snapshot,
                                      SIMDCapabilities *caps);
#ifdef __cplusplus
}
#endif
