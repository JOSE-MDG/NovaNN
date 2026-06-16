/**
 * @file simd.c
 * @brief CPU SIMD capability detection and runtime feature detection
 * implementation.
 *
 * @details
 * This module provides runtime detection of CPU SIMD (Single Instruction,
 * Multiple Data) capabilities using the CPUID instruction. It supports
 * detection of SSE, AVX, AVX-512, FMA, F16C, VNNI, and AMX instruction
 * sets on both x86_64 Linux and Windows platforms.
 *
 * The detected capabilities are cached in a thread-safe singleton pattern
 * using platform-specific threading primitives (C11 `call_once` on Linux,
 * Windows `InitOnceExecuteOnce` on `_WIN64`) to ensure the detection is
 * performed only once, even when called from multiple threads.
 *
 * ## Architecture
 * The detection follows a three-phase CPUID query strategy:
 * - **Phase 1**: Leaf 1 for base features (SSE4.2, FMA3, AVX, F16C)
 * - **Phase 2**: Leaf 7, subleaf 0 for AVX2, AVX-512 variants, AMX
 * - **Phase 3**: Leaf 7, subleaf 1 for VNNI, BF16, FP16 extensions
 *
 * ## Platform Support
 * - **Windows (_WIN64)**: Uses __cpuid() and __cpuidex() from intrin.h;
 *   threading via `INIT_ONCE` + `InitOnceExecuteOnce` from `<windows.h>`.
 * - **Linux/Unix**: Uses __get_cpuid() and __cpuid_count() from cpuid.h;
 *   threading via C11 `once_flag` + `call_once` from `<threads.h>`.
 *
 * @see simd.h Public interface and @ref SIMDCapabilities structure
 * @see get_simd_capabilities() Thread-safe singleton accessor
 */

#include <ncore/core/dtype.h>
#include <ncore/simd/simd.h>
#include <sys/types.h>
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

#include <string.h>

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
static SIMDCapabilities simd = {0};

#ifdef __linux__
/**
 * @var static once_flag init_flag
 * @brief Once-flag for thread-safe lazy initialization.
 *
 * @details
 * Ensures @ref init_once() is called exactly once, even when
 * @ref get_simd_capabilities() is called concurrently from multiple threads.
 * Initialized to the standard ONCE_FLAG_INIT.
 *
 * @see init_once()
 * @see get_simd_capabilities()
 */
static once_flag init_flag = ONCE_FLAG_INIT;
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
 * Queries CPUID leaves to detect available SIMD features and populates
 * the provided @ref SIMDCapabilities structure. The function clears the
 * structure before detection and sets each flag based on CPU support.
 *
 * @par CPUID Leaves Queried:
 * - **Leaf 1 (ECX/EDX):** SSE4.2, FMA3, AVX, F16C
 * - **Leaf 7, subleaf 0 (EBX/ECX/EDX):** AVX2, AVX-512 variants, AMX
 * - **Leaf 7, subleaf 1 (EAX/EDX):** AVX2 VNNI, AVX-512 BF16, AMX FP16
 *
 * @param[out] caps Pointer to the SIMDCapabilities structure to populate.
 *                  All fields are zeroed before detection.
 *
 * @pre caps must point to a valid SIMDCapabilities structure.
 * @post All capability flags in caps are set to true/false based on
 *       CPU support. The amx_ and vnni_ composite flags are computed
 *       from their constituent features.
 *
 * @warning On non-Windows platforms, if CPUID leaf 1 is not supported,
 *          the function returns early with caps zeroed.
 *
 * @note This function is platform-specific:
 *       - **Windows (_WIN64):** Uses __cpuid() and __cpuidex() from intrin.h
 *       - **Linux/Unix:** Uses __get_cpuid() and __cpuid_count() from cpuid.h
 *
 * @see get_simd_capabilities() for the public API.
 * @see SIMDCapabilities for the structure definition.
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

  caps->sse4_2_ = (bool)((ecx & (1 << 20)) != 0); ///< SSE4.2
  caps->fma3_ = (bool)((ecx & (1 << 12)) != 0);   ///< FMA3
  caps->avx_ = (bool)((ecx & (1 << 28)) != 0);    ///< AVX
  caps->f16c_ = (bool)((ecx & (1 << 29)) != 0);   ///< F16C

  /* CPUID leaf 7, subleaf 0: AVX2, AVX-512, AMX detection */
#ifdef _WIN64
  __cpuidex(cpu_info, 7, 0);
  eax = (uint32)cpu_info[0];
  ebx = (uint32)cpu_info[1];
  ecx = (uint32)cpu_info[2];
  edx = (uint32)cpu_info[3];
#else
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

  caps->avx2_ = (bool)((ebx & (1 << 5)) != 0);     ///< AVX2
  caps->avx512f_ = (bool)((ebx & (1 << 16)) != 0); ///< AVX-512 Foundation
  caps->avx512_dq_ =
      (bool)((ebx & (1 << 17)) != 0); ///< AVX-512 Doubleword/Quadword
  caps->avx512_bw_ = (bool)((ebx & (1 << 30)) != 0); ///< AVX-512 Byte/Word
  caps->avx512_vl_ =
      (bool)((ebx & (1U << 31)) != 0); ///< AVX-512 Vector Length extensions
  caps->avx512_vnni_ = (bool)((ecx & (1 << 11)) != 0); ///< AVX-512 VNNI
  caps->amx_bf16_ = (bool)((edx & (1 << 22)) != 0);    ///< AMX BF16
  caps->avx512_fp16_ = (bool)((edx & (1 << 23)) != 0); ///< AVX-512 FP16
  caps->amx_int8_ = (bool)((edx & (1 << 25)) != 0);    ///< AMX INT8

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

  caps->avx2_vnni_ = (bool)((eax & (1 << 4)) != 0);   ///< AVX2 VNNI
  caps->avx512_bf16_ = (bool)((eax & (1 << 5)) != 0); ///< AVX-512 BF16
  caps->amx_fp16_ = (bool)((eax & (1 << 21)) != 0);   ///< AMX FP16
  caps->avx2_int8_ = (bool)((edx & (1 << 4)) != 0);   ///< AVX2 INT8

  /* Composite flags */
  caps->amx_ = (bool)(caps->amx_bf16_ || caps->amx_fp16_ || caps->amx_int8_);
  caps->vnni_ = (bool)(caps->avx512_vnni_ || caps->avx2_vnni_);
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
 * This function matches the `PINIT_ONCE_FN` signature required by
 * Windows `InitOnceExecuteOnce()`.  It delegates to
 * @ref detect_simd_capabilities() to populate the global @ref simd
 * structure.
 *
 * @return Always `TRUE` (initialisation always succeeds).
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
 *       once using C11 `call_once` on Linux or `InitOnceExecuteOnce` on
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
  InitOnceExecuteOnce(&init_flag, init_once_win, NULL, NULL);
#endif
  return &simd;
}
