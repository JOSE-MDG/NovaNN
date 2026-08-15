/**
 * @file simd.c
 * @brief CPU SIMD capability detection and runtime feature detection
 * implementation.
 *
 * @details
 * This module provides runtime detection of CPU SIMD (Single Instruction,
 * Multiple Data) capabilities using the CPUID instruction. It supports
 * detection of SSE, AVX, AVX-512, FMA, F16C, VNNI, AMX, and AVX10
 * instruction sets on both x86_64 Linux and Windows platforms, giving the
 * rest of NovaNN a single, reliable source of truth for which vectorized
 * kernel variants are safe to dispatch to on the current machine.
 *
 * The detected capabilities are cached in a thread-safe singleton pattern
 * using platform-specific threading primitives (C11 @c call_once on Linux,
 * Windows @c InitOnceExecuteOnce on @c _WIN64) to ensure the detection is
 * performed only once, even when called from multiple threads.
 *
 * @section platform-support Platform Support
 * @li Windows (_WIN64): Uses __cpuid() and __cpuidex() from intrin.h;
 *   threading via @c INIT_ONCE + @c InitOnceExecuteOnce from @c <windows.h>.
 * @li Linux/Unix: Uses __get_cpuid() and __cpuid_count() from cpuid.h;
 *   threading via C11 @c once_flag + @c call_once from @c <threads.h>.
 *
 * @see simd.h                     Public interface and @ref SIMDCapabilities structure.
 * @see get_simd_capabilities()    Thread-safe singleton accessor.
 */

#ifdef __linux__
#include <threads.h>
#elif defined(_WIN64)
#include <windows.h>
#endif

#ifdef _WIN64
/** @brief Windows intrinsics for CPUID support. */
#include <intrin.h>
#else
/** @brief GCC/Clang CPUID intrinsic support. */
#include <cpuid.h>
#endif

#include <ncore/core/dtype.h>
#include <ncore/simd/simd.h>

/**
 * @var static SIMDCapabilities simd
 * @brief Global cached CPU capabilities structure.
 *
 * @details
 * Stores the detected SIMD features after the first call to
 * @ref get_simd_capabilities(). Once initialized, this structure is
 * read-only and can be safely accessed from multiple threads.
 *
 * @see get_simd_capabilities()
 * @see detect_simd_capabilities()
 */
static SIMDCapabilities simd = {};

#ifdef __linux__
/**
 * @var static once_flag init_flag
 * @brief Once-flag for thread-safe lazy initialization.
 *
 * @details
 * Ensures @ref init_once() is called exactly once, even when
 * @ref get_simd_capabilities() is called concurrently from multiple threads.
 * Initialized to the C23 empty initializer @c = {}.
 *
 * @see init_once()
 * @see get_simd_capabilities()
 */
static once_flag init_flag = {};
#elif defined(_WIN64)
/**
 * @var static INIT_ONCE init_flag
 * @brief Windows one-time initialisation guard for thread-safe lazy
 *        initialization.
 *
 * @details
 * Ensures the CPU capabilities detection callback is called exactly
 * once, even when @ref get_simd_capabilities() is called concurrently
 * from multiple threads.
 *
 * @see get_simd_capabilities()
 */
static INIT_ONCE init_flag = INIT_ONCE_STATIC_INIT;
#endif

/**
 * @brief Detect CPU SIMD capabilities via CPUID instruction.
 *
 * @details
 * Queries the relevant CPUID leaves to detect available SIMD features and
 * populates the provided @ref SIMDCapabilities structure.  Individual
 * leaves and bit positions are documented inline alongside the fields
 * they populate.
 *
 * @param[in,out] caps Pointer to the SIMDCapabilities structure to populate.
 *                  Must be zero-initialised by the caller, since fields
 *                  are only written when their CPUID leaf is available.
 *
 * @pre caps must point to a valid SIMDCapabilities structure.
 * @post All capability flags in caps are set to true/false based on
 *       CPU support. The amx_ and vnni_ composite flags are computed
 *       from their constituent features.
 *
 * @note On non-Windows platforms, if CPUID leaf 1 is not supported,
 *       the function returns early leaving caps untouched.  Callers
 *       should zero-initialise the structure (e.g., @c = @c {})
 *       beforehand.
 *
 * @note This function is platform-specific:
 *       @li Windows (_WIN64): Uses __cpuid() and __cpuidex() from intrin.h
 *       @li Linux/Unix: Uses __get_cpuid() and __cpuid_count() from cpuid.h
 *
 * @see get_simd_capabilities()  Public API accessor.
 * @see SIMDCapabilities         Structure definition.
 */
static inline void detect_simd_capabilities(SIMDCapabilities *restrict caps) {

  uint32 eax, ebx, ecx, edx;

  /* CPUID leaf 1: SSE4.2, FMA3, AVX, F16C detection */
#ifdef _WIN64
  int cpu_info[4];
  __cpuid(cpu_info, 1);
  eax = (uint32)cpu_info[0];
  ebx = (uint32)cpu_info[1];
  ecx = (uint32)cpu_info[2];
  edx = (uint32)cpu_info[3];
#else
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return;
  }
#endif

  caps->sse4_2_ = ((ecx & (1 << 20)) != 0); ///< SSE4.2
  caps->fma3_ = ((ecx & (1 << 12)) != 0);   ///< FMA3
  caps->avx_ = ((ecx & (1 << 28)) != 0);    ///< AVX
  caps->f16c_ = ((ecx & (1 << 29)) != 0);   ///< F16C

  /* CPUID leaf 7, subleaf 0: AVX2, AVX-512, AMX, AVX10 presence flag */
#ifdef _WIN64
  __cpuidex(cpu_info, 7, 0);
  eax = (uint32)cpu_info[0];
  ebx = (uint32)cpu_info[1];
  ecx = (uint32)cpu_info[2];
  edx = (uint32)cpu_info[3];
#else
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

  caps->avx2_ = ((ebx & (1 << 5)) != 0);       ///< AVX2
  caps->avx512f_ = ((ebx & (1 << 16)) != 0);   ///< AVX-512 Foundation
  caps->avx512_dq_ = ((ebx & (1 << 17)) != 0); ///< AVX-512 Doubleword/Quadword
  caps->avx512_bw_ = ((ebx & (1 << 30)) != 0); ///< AVX-512 Byte/Word
  caps->avx512_vl_ =
      ((ebx & (1U << 31)) != 0); ///< AVX-512 Vector Length extensions
  caps->avx512_vnni_ = ((ecx & (1 << 11)) != 0); ///< AVX-512 VNNI
  caps->amx_bf16_ = ((edx & (1 << 22)) != 0);    ///< AMX BF16
  caps->avx512_fp16_ = ((edx & (1 << 23)) != 0); ///< AVX-512 FP16
  caps->amx_int8_ = ((edx & (1 << 25)) != 0);    ///< AMX INT8

  /* CPUID.(EAX=07H,ECX=0H):EDX[19] — Intel AVX10 converged vector ISA
   * presence flag. Leaf 0x24 is only architecturally valid when this bit
   * is set; it must not be queried otherwise. */
  const bool avx10_present = ((edx & (1 << 19)) != 0);

  /* CPUID leaf 7, subleaf 1: AVX2 VNNI, AVX-512 BF16, AMX FP16 */
#ifdef _WIN64
  __cpuidex(cpu_info, 7, 1);
  eax = (uint32)cpu_info[0];
  ebx = (uint32)cpu_info[1];
  ecx = (uint32)cpu_info[2];
  edx = (uint32)cpu_info[3];
#else
  __cpuid_count(7, 1, eax, ebx, ecx, edx);
#endif

  caps->avx2_vnni_ = ((eax & (1 << 4)) != 0);   ///< AVX2 VNNI
  caps->avx512_bf16_ = ((eax & (1 << 5)) != 0); ///< AVX-512 BF16
  caps->amx_fp16_ = ((eax & (1 << 21)) != 0);   ///< AMX FP16
  caps->avx2_int8_ = ((edx & (1 << 4)) != 0);   ///< AVX2 INT8

  /* CPUID leaf 0x24, subleaf 0, EBX[7:0]: AVX10 version number (1 or 2).
   * Only queried when avx10_present is set, since the leaf is otherwise
   * undefined. EBX[18:16] (legacy per-width support bits) are reserved
   * on all shipped AVX10 parts and are not consulted here. */
  caps->avx10_1_ = false;
  caps->avx10_2_ = false;

  if (avx10_present) {
    uint32 eax24, ebx24, ecx24, edx24;
#ifdef _WIN64
    __cpuidex(cpu_info, 0x24, 0);
    eax24 = (uint32)cpu_info[0];
    ebx24 = (uint32)cpu_info[1];
    ecx24 = (uint32)cpu_info[2];
    edx24 = (uint32)cpu_info[3];
#else
    __cpuid_count(0x24, 0, eax24, ebx24, ecx24, edx24);
#endif
    (void)eax24;
    (void)ecx24;
    (void)edx24;

    const uint32 avx10_version = ebx24 & 0xFFu; ///< EBX[7:0]: AVX10 version

    caps->avx10_1_ = (avx10_version >= 1); ///< AVX10.1 or newer
    caps->avx10_2_ = (avx10_version >= 2); ///< AVX10.2 or newer
  }

  /* Composite flags */
  caps->amx_ = ((caps->amx_bf16_ || caps->amx_fp16_ || caps->amx_int8_) != 0);
  caps->vnni_ = ((caps->avx512_vnni_ || caps->avx2_vnni_) != 0);
}

#ifdef __linux__
/**
 * @brief One-time initializer for thread-safe lazy initialization.
 *
 * @details
 * This function is registered with call_once() and executes
 * @ref detect_simd_capabilities() to populate the global @ref simd structure.
 * It is guaranteed to be called exactly once, regardless of how many
 * threads invoke @ref get_simd_capabilities() concurrently.
 *
 * @see get_simd_capabilities()
 * @see init_flag
 * @see detect_simd_capabilities()
 */
static inline void init_once(void) { detect_simd_capabilities(&simd); }
#elif defined(_WIN64)
/**
 * @brief Windows one-time initialization callback.
 *
 * @details
 * This function matches the @c PINIT_ONCE_FN signature required by
 * Windows @c InitOnceExecuteOnce().  It delegates to
 * @ref detect_simd_capabilities() to populate the global @ref simd
 * structure.
 *
 * @param[in] once   Pointer to the one-time initialisation structure.
 * @param[in] param  Optional callback data (unused).
 * @param[in] ctx    Optional callback context (unused).
 *
 * @return Always @c TRUE (initialisation always succeeds).
 *
 * @see get_simd_capabilities()
 * @see init_flag
 * @see detect_simd_capabilities()
 */
static BOOL CALLBACK init_once_win(PINIT_ONCE once, PVOID param, PVOID *ctx) {
  (void)once;
  (void)param;
  (void)ctx;
  detect_simd_capabilities(&simd);
  return TRUE;
}
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
const SIMDCapabilities *get_simd_capabilities() {
#ifdef __linux__
  call_once(&init_flag, init_once);
#elif defined(_WIN64)
  InitOnceExecuteOnce(&init_flag, init_once_win, nullptr, nullptr);
#endif
  return &simd;
}
